//! Official OpenAI Responses language model.

use std::{collections::VecDeque, sync::Arc, time::Duration};

use futures_util::StreamExt;
use oven_sdk::{
    AbortSignal, AdapterId, BoxFuture, BoxStream, CompactionRequest, CompactionResult, ErrorStage,
    JsonValue, LanguageModel, LanguageModelDescriptor, ModelConfig, ModelError, ModelId,
    ModelIdentity, NativeContextScope, ProviderMetadata, Request, RequestMetadata, StreamItem,
    StreamPart, StreamResponse,
};

use crate::{
    configuration::{
        OpenAiAuth, OpenAiResponsesSettings, build_client, canonical_endpoint,
        header_scope_component, official_headers, replay_resource_id,
        validate_responses_declaration, validate_routing_discriminator,
    },
    error::classify_error,
    responses::{compaction, request, state::State},
    transport::OpenAiTimeouts,
    wire::{
        OPENAI_RESPONSES_ADAPTER_ID,
        responses::{COMPACT_PATH, PATH},
    },
};
use reqwest::header::{CONTENT_TYPE, HeaderMap};

/// A configured official OpenAI Responses model.
#[derive(Clone)]
pub struct OpenAiResponsesModel {
    runtime: Arc<Runtime>,
}

impl OpenAiResponsesModel {
    /// Constructs one official Responses model from explicit configuration.
    pub fn new(
        config: ModelConfig<OpenAiAuth, OpenAiResponsesSettings>,
    ) -> Result<Self, ModelError> {
        config.validate()?;
        validate_responses_declaration(&config.model.capabilities, config.settings.compaction)?;
        validate_routing_discriminator(
            config.provider.headers.dynamic_headers.is_some(),
            config.settings.routing_discriminator.as_deref(),
        )?;
        let adapter_id = AdapterId::new(OPENAI_RESPONSES_ADAPTER_ID);
        let descriptor = LanguageModelDescriptor::new(
            ModelIdentity::new(config.provider.id.clone(), config.model.id.clone())?,
            adapter_id,
            config.model.capabilities.clone(),
        )?;
        let resource_id = replay_resource_id(&[
            &canonical_endpoint(config.provider.api.as_url()),
            OPENAI_RESPONSES_ADAPTER_ID,
            config.provider.auth.organization.as_deref().unwrap_or(""),
            config.provider.auth.project.as_deref().unwrap_or(""),
            config
                .settings
                .routing_discriminator
                .as_deref()
                .unwrap_or(""),
            &header_scope_component(&config.provider.headers),
            match config.settings.compaction {
                crate::configuration::OpenAiResponsesCompaction::Unsupported => {
                    "compaction_unsupported"
                }
                crate::configuration::OpenAiResponsesCompaction::V1 => "responses_compact_v1",
            },
        ])?;
        let scope = NativeContextScope::new(
            descriptor.identity.provider_id.clone(),
            descriptor.identity.model_id.clone(),
            resource_id,
        )?;
        let client = build_client(config.settings.client, &config.settings.timeouts)?;
        Ok(Self {
            runtime: Arc::new(Runtime {
                descriptor,
                scope,
                api: config.provider.api.as_url().to_string(),
                auth: config.provider.auth,
                headers: config.provider.headers,
                client,
                timeouts: config.settings.timeouts,
            }),
        })
    }

    /// Returns the configured model ID.
    #[must_use]
    pub fn model_id(&self) -> &ModelId {
        &self.runtime.descriptor.identity.model_id
    }
}

#[derive(Clone)]
struct Runtime {
    descriptor: LanguageModelDescriptor,
    scope: NativeContextScope,
    api: String,
    auth: OpenAiAuth,
    headers: oven_sdk::HeaderConfig,
    client: reqwest::Client,
    timeouts: OpenAiTimeouts,
}

impl LanguageModel for OpenAiResponsesModel {
    fn descriptor(&self) -> &LanguageModelDescriptor {
        &self.runtime.descriptor
    }

    fn validate_request(&self, request_value: &Request) -> Result<(), ModelError> {
        request::validate_request(
            request_value,
            &self.runtime.descriptor.capabilities,
            &self.runtime.descriptor,
            &self.runtime.scope,
        )
    }

    fn supports_request(&self, request: &Request) -> bool {
        self.validate_request(request).is_ok()
    }

    fn validate_compaction(&self, compaction: &CompactionRequest) -> Result<(), ModelError> {
        compaction.validate_for(&self.runtime.descriptor.capabilities)?;
        request::validate_request(
            &compaction.request,
            &self.runtime.descriptor.capabilities,
            &self.runtime.descriptor,
            &self.runtime.scope,
        )?;
        let options = crate::options::responses_compaction_options(compaction)?;
        compaction::validate_options(&options)?;
        Ok(())
    }

    fn supports_compaction(&self, request: &CompactionRequest) -> bool {
        self.validate_compaction(request).is_ok()
    }

    fn stream<'a>(
        &'a self,
        request_value: Request,
        abort: AbortSignal,
    ) -> BoxFuture<'a, Result<StreamResponse, ModelError>> {
        Box::pin(async move {
            self.validate_request(&request_value)?;
            if abort.is_aborted() {
                return Err(ModelError::abort("request was aborted before dispatch")
                    .with_stage(ErrorStage::Connect));
            }
            let descriptor = self.runtime.descriptor.clone();
            let policy = descriptor.capabilities.replay.policy;
            let encoded =
                request::encode_request(&request_value, &descriptor, &self.runtime.scope, policy)?;
            let headers = official_headers(&self.runtime.auth, &self.runtime.headers)?;
            let send = self
                .runtime
                .client
                .post(format!("{}/{PATH}", self.runtime.api.trim_end_matches('/')))
                .headers(headers)
                .json(&encoded.body)
                .send();
            let response = tokio::select! {
                value = tokio::time::timeout(self.runtime.timeouts.headers, send) => value
                    .map_err(|_| ModelError::timeout("response headers timed out").with_stage(ErrorStage::ResponseHeaders))?
                    .map_err(|_| ModelError::transport("OpenAI Responses request failed").with_stage(ErrorStage::Connect))?,
                _ = abort.aborted() => return Err(ModelError::abort("request was aborted before response headers").with_stage(ErrorStage::Connect)),
            };
            let mut head = crate::transport::response_head(&response, &["x-request-id".into()]);
            if !response.status().is_success() {
                let status = response.status().as_u16();
                let error_headers = response.headers().clone();
                let request_id = head.request_id.clone();
                let (body, bytes) = crate::transport::read_error_body(
                    response,
                    &abort,
                    self.runtime.timeouts.stream_idle,
                )
                .await?;
                return Err(classify_error(
                    status,
                    &body,
                    request_id,
                    ErrorStage::ResponseBody,
                    bytes,
                    &error_headers,
                ));
            }
            let response_headers = response.headers().clone();
            if !response
                .headers()
                .get(reqwest::header::CONTENT_TYPE)
                .and_then(|value| value.to_str().ok())
                .is_some_and(|value| value.starts_with("text/event-stream"))
            {
                return Err(ModelError::invalid_response(
                    "OpenAI Responses response is not an SSE content type",
                )
                .with_stage(ErrorStage::ResponseHeaders));
            }
            let mut live = LiveState {
                bytes: Box::pin(response.bytes_stream()),
                parser: crate::sse::Parser::new("OpenAI SSE contains invalid UTF-8")
                    .clear_name_on_empty_event(),
                state: State::new(
                    descriptor.adapter_id.clone(),
                    self.runtime.scope.clone(),
                    policy,
                ),
                queue: VecDeque::from([Ok(StreamPart::StreamStart {
                    warnings: encoded.warnings,
                })]),
                pending_events: VecDeque::new(),
                pending_error: None,
                abort,
                idle: self.runtime.timeouts.stream_idle,
                count: 0,
                eof: false,
                request_id: head.request_id.clone(),
                response_headers,
                saw_provider_event: false,
            };
            early_peek(&mut live).await?;
            head.response_metadata
                .extend(live.state.response_metadata().clone());
            let stream = futures_util::stream::unfold(live, |mut live| async move {
                loop {
                    if let Some(item) = live.queue.pop_front() {
                        return Some((item, live));
                    }
                    if let Some(error) = live.pending_error.take() {
                        live.eof = true;
                        return Some((Err(error), live));
                    }
                    if live.eof {
                        return None;
                    }
                    if let Err(error) = read_live(&mut live, false).await {
                        live.pending_error = Some(error);
                    }
                }
            });
            Ok(StreamResponse::new(Box::pin(stream))
                .with_request(RequestMetadata {
                    replay: encoded.replay,
                    provider_metadata: ProviderMetadata::new(),
                })
                .with_response(head))
        })
    }

    fn compact<'a>(
        &'a self,
        compaction_request: CompactionRequest,
        abort: AbortSignal,
    ) -> BoxFuture<'a, Result<CompactionResult, ModelError>> {
        Box::pin(async move {
            self.validate_compaction(&compaction_request)?;
            if abort.is_aborted() {
                return Err(
                    ModelError::abort("native compaction was aborted before encoding")
                        .with_stage(ErrorStage::NativeContextEncode),
                );
            }
            let descriptor = self.runtime.descriptor.clone();
            let encoded = request::encode_compaction_request(
                &compaction_request,
                &descriptor,
                &self.runtime.scope,
                descriptor.capabilities.replay.policy,
            )?;
            if !encoded.warnings.is_empty() {
                return Err(ModelError::native_context(
                    "OpenAI Responses compaction cannot omit normalized reasoning state",
                )
                .with_stage(ErrorStage::NativeContextEncode));
            }
            let body = serde_json::to_vec(&encoded.body).map_err(|_| {
                ModelError::invalid_request("could not encode OpenAI compaction request")
                    .with_stage(ErrorStage::NativeContextEncode)
            })?;
            compaction::validate_request_size(body.len())?;
            let headers = official_headers(&self.runtime.auth, &self.runtime.headers)?;
            let send = self
                .runtime
                .client
                .post(format!(
                    "{}/{COMPACT_PATH}",
                    self.runtime.api.trim_end_matches('/')
                ))
                .headers(headers)
                .header(CONTENT_TYPE, "application/json")
                .body(body)
                .send();
            let response = tokio::select! {
                value = tokio::time::timeout(self.runtime.timeouts.headers, send) => value
                    .map_err(|_| ModelError::timeout("compaction response headers timed out").with_stage(ErrorStage::ResponseHeaders))?
                    .map_err(|_| ModelError::transport("OpenAI Responses compaction request failed").with_stage(ErrorStage::Connect))?,
                _ = abort.aborted() => return Err(ModelError::abort("native compaction was aborted before response headers").with_stage(ErrorStage::Connect)),
            };
            let mut head = crate::transport::response_head(&response, &["x-request-id".into()]);
            if !response.status().is_success() {
                let status = response.status().as_u16();
                let error_headers = response.headers().clone();
                let request_id = head.request_id.clone();
                let (body, bytes) = crate::transport::read_error_body(
                    response,
                    &abort,
                    self.runtime.timeouts.stream_idle,
                )
                .await?;
                return Err(classify_error(
                    status,
                    &body,
                    request_id,
                    ErrorStage::ResponseBody,
                    bytes,
                    &error_headers,
                ));
            }
            if !response
                .headers()
                .get(CONTENT_TYPE)
                .and_then(|value| value.to_str().ok())
                .is_some_and(|value| value.starts_with("application/json"))
            {
                return Err(ModelError::invalid_response(
                    "OpenAI Responses compaction response is not JSON",
                )
                .with_stage(ErrorStage::ResponseHeaders));
            }
            compaction::validate_response_length(response.content_length())?;
            let (body, bytes) = crate::transport::read_bounded_body(
                response,
                &abort,
                self.runtime.timeouts.stream_idle,
                compaction::MAX_COMPACTION_RESPONSE_BYTES,
            )
            .await?;
            let value = serde_json::from_slice(&body).map_err(|_| {
                ModelError::invalid_response("OpenAI Responses compaction response is invalid JSON")
                    .with_stage(ErrorStage::NativeContextDecode)
                    .with_bytes_received(bytes)
            })?;
            let decoded = compaction::decode_response(
                value,
                descriptor.adapter_id,
                self.runtime.scope.clone(),
                bytes,
            )?;
            head.response_metadata.extend(decoded.response_metadata);
            Ok(CompactionResult {
                native_context: decoded.native_context,
                usage: decoded.usage,
                request: RequestMetadata {
                    replay: encoded.replay,
                    provider_metadata: ProviderMetadata::new(),
                },
                response: head,
            })
        })
    }
}

struct LiveState {
    bytes: BoxStream<'static, Result<bytes::Bytes, reqwest::Error>>,
    parser: crate::sse::Parser,
    state: State,
    queue: VecDeque<StreamItem>,
    pending_events: VecDeque<crate::sse::Event>,
    pending_error: Option<ModelError>,
    abort: AbortSignal,
    idle: Duration,
    count: u64,
    eof: bool,
    request_id: Option<String>,
    response_headers: HeaderMap,
    saw_provider_event: bool,
}

async fn early_peek(live: &mut LiveState) -> Result<(), ModelError> {
    loop {
        if read_live(live, true).await? {
            return Ok(());
        }
        if live.eof {
            return Err(ModelError::unexpected_eof(
                "OpenAI Responses stream ended before a semantic event",
            )
            .with_bytes_received(live.count));
        }
    }
}

async fn read_live(live: &mut LiveState, stop_after_event: bool) -> Result<bool, ModelError> {
    if !live.pending_events.is_empty() {
        return process_events(live, stop_after_event);
    }
    let next = tokio::select! {
        value = tokio::time::timeout(live.idle, live.bytes.next()) => value
            .map_err(|_| ModelError::timeout("stream idle timeout").with_stage(ErrorStage::StreamRead).with_bytes_received(live.count))?,
        _ = live.abort.aborted() => return Err(ModelError::abort("stream was aborted").with_stage(ErrorStage::StreamRead).with_bytes_received(live.count)),
    };
    match next {
        Some(Ok(chunk)) => {
            live.count = live.count.saturating_add(chunk.len() as u64);
            live.parser
                .feed_into(&chunk, &mut live.pending_events)
                .map_err(|error| error.with_bytes_received(live.count))?;
        }
        Some(Err(_)) => {
            return Err(ModelError::transport("OpenAI Responses stream read failed")
                .with_stage(ErrorStage::StreamRead)
                .with_bytes_received(live.count));
        }
        None => {
            live.eof = true;
            live.pending_events.extend(
                live.parser
                    .finish()
                    .map_err(|error| error.with_bytes_received(live.count))?,
            );
        }
    }
    let semantic = process_events(live, stop_after_event)?;
    if live.eof && !live.state.done() {
        return Err(ModelError::unexpected_eof(
            "OpenAI Responses stream ended before a terminal response event",
        )
        .with_bytes_received(live.count));
    }
    Ok(semantic)
}

fn process_events(live: &mut LiveState, stop_after_event: bool) -> Result<bool, ModelError> {
    let mut semantic = false;
    while let Some(event) = live.pending_events.pop_front() {
        if event.data.is_empty() {
            continue;
        }
        semantic = true;
        let value: JsonValue = serde_json::from_str(&event.data).map_err(|_| {
            ModelError::invalid_response("OpenAI Responses SSE event is invalid JSON")
                .with_stage(ErrorStage::StreamDecode)
                .with_bytes_received(live.count)
        })?;
        let kind = value
            .get("type")
            .and_then(JsonValue::as_str)
            .unwrap_or("")
            .to_owned();
        if kind == "response.failed" || kind == "error" || value.get("error").is_some() {
            let error_value = value
                .pointer("/response/error")
                .cloned()
                .map(|error| serde_json::json!({"error":error}))
                .unwrap_or_else(|| value.clone());
            let status = stream_error_status(&error_value);
            let error = classify_error(
                status,
                error_value.to_string().as_bytes(),
                live.request_id.clone(),
                ErrorStage::StreamEvent,
                live.count,
                &live.response_headers,
            );
            if !live.saw_provider_event {
                return Err(error);
            }
            let mut parts = Vec::new();
            live.state.in_band_error(error, &mut parts)?;
            live.queue.extend(parts.into_iter().map(Ok));
            live.eof = true;
            live.pending_events.clear();
            return Ok(true);
        }
        live.saw_provider_event = true;
        let mut parts = Vec::new();
        live.state.apply(value, &mut parts, live.count)?;
        live.queue.extend(parts.into_iter().map(Ok));
        if live.state.done() {
            live.eof = true;
            live.pending_events.clear();
            return Ok(true);
        }
        if stop_after_event {
            return Ok(true);
        }
    }
    Ok(semantic)
}

fn stream_error_status(value: &JsonValue) -> u16 {
    if let Some(status) = value.pointer("/error/code").and_then(JsonValue::as_u64) {
        return u16::try_from(status).unwrap_or(500);
    }
    let text = value.to_string().to_lowercase();
    if text.contains("rate_limit") || text.contains("insufficient_quota") {
        429
    } else if text.contains("auth") {
        401
    } else if text.contains("permission") {
        403
    } else if text.contains("not_found") {
        404
    } else if text.contains("invalid") || text.contains("context") {
        400
    } else if text.contains("overload") {
        503
    } else if text.contains("timeout") {
        504
    } else {
        500
    }
}
