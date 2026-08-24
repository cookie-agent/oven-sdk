//! Azure OpenAI Responses language model.

use std::{collections::VecDeque, sync::Arc, time::Duration};

use futures_util::StreamExt;
use oven_sdk::{
    AbortSignal, BoxFuture, BoxStream, CompactionRequest, CompactionResult, ErrorStage, JsonValue,
    LanguageModel, LanguageModelDescriptor, ModelError, ModelId, NativeContextScope,
    ProviderMetadata, Request, RequestMetadata, StreamItem, StreamPart, StreamResponse,
};
use reqwest::header::HeaderMap;

use crate::{
    configuration::{AzureOpenAiAuth, AzureOpenAiResponsesConfig, Config, build_responses},
    error::classify_error,
    responses::{compaction, request, state::State},
    wire::responses::{COMPACT_PATH, PATH},
};

/// A configured Azure OpenAI Responses deployment.
#[derive(Clone)]
pub struct AzureOpenAiResponsesModel {
    pub(crate) config: Arc<Config>,
    descriptor: LanguageModelDescriptor,
}

impl AzureOpenAiResponsesModel {
    /// Constructs one registry-free Azure Responses model.
    pub fn new(config: AzureOpenAiResponsesConfig) -> Result<Self, ModelError> {
        let (config, descriptor) = build_responses(config)?;
        Ok(Self { config, descriptor })
    }

    /// Returns the configured deployment-name model ID.
    #[must_use]
    pub fn model_id(&self) -> &ModelId {
        &self.descriptor.identity.model_id
    }

    /// Returns the exact configured native-context scope when V1 compaction is enabled.
    #[must_use]
    pub fn native_context_scope(&self) -> Option<&NativeContextScope> {
        self.config.configured_native_context_scope()
    }
}

impl LanguageModel for AzureOpenAiResponsesModel {
    fn descriptor(&self) -> &LanguageModelDescriptor {
        &self.descriptor
    }

    fn validate_request(&self, request_value: &Request) -> Result<(), ModelError> {
        request::validate_request(
            request_value,
            &self.descriptor().capabilities,
            &self.descriptor,
            self.native_context_scope(),
        )
    }

    fn supports_request(&self, request: &Request) -> bool {
        self.validate_request(request).is_ok()
    }

    fn validate_compaction(&self, request_value: &CompactionRequest) -> Result<(), ModelError> {
        let (binding, scope) = self.config.configured_native_context()?.ok_or_else(|| {
            ModelError::unsupported("Azure Responses V1 native compaction is not configured")
        })?;
        compaction::validate(request_value, &self.descriptor, &binding, &scope)
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
            let descriptor = self.descriptor();
            let replay_policy = self.config.capabilities.replay.policy;
            let (headers, replay_binding, replay_scope) =
                configured_headers(&self.config, &abort).await?;
            let encoded = request::encode_request(
                &request_value,
                descriptor,
                replay_policy,
                &replay_binding,
                &replay_scope,
            )?;
            let mut url = reqwest::Url::parse(&format!("{}/{PATH}", self.config.base_url))
                .map_err(|_| ModelError::invalid_request("invalid Azure Responses URL"))?;
            if !self.config.query.is_empty() {
                url.query_pairs_mut().extend_pairs(self.config.query.iter());
            }
            let send = self
                .config
                .client
                .post(url)
                .headers(headers)
                .json(&encoded.body)
                .send();
            let response = tokio::select! {
                value = tokio::time::timeout(self.config.timeouts.headers, send) => value
                    .map_err(|_| ModelError::timeout("response headers timed out").with_stage(ErrorStage::ResponseHeaders))?
                    .map_err(|_| ModelError::transport("Azure OpenAI Responses request failed").with_stage(ErrorStage::Connect))?,
                _ = abort.aborted() => return Err(ModelError::abort("request was aborted before response headers").with_stage(ErrorStage::Connect)),
            };
            let mut head = crate::transport::response_head(
                &response,
                &["apim-request-id".into(), "x-request-id".into()],
            );
            if !response.status().is_success() {
                let status = response.status().as_u16();
                let error_headers = response.headers().clone();
                let request_id = head.request_id.clone();
                let (body, bytes) = crate::transport::read_error_body(
                    response,
                    &abort,
                    self.config.timeouts.stream_idle,
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
                    "Azure OpenAI Responses response is not an SSE content type",
                )
                .with_stage(ErrorStage::ResponseHeaders));
            }
            let mut live = LiveState {
                bytes: Box::pin(response.bytes_stream()),
                parser: crate::sse::Parser::new("Azure OpenAI SSE contains invalid UTF-8")
                    .clear_name_on_empty_event(),
                state: State::new(
                    descriptor.adapter_id.clone(),
                    replay_policy,
                    replay_binding,
                    replay_scope,
                ),
                queue: VecDeque::from([Ok(StreamPart::StreamStart {
                    warnings: encoded.warnings,
                })]),
                pending_events: VecDeque::new(),
                pending_error: None,
                abort,
                idle: self.config.timeouts.stream_idle,
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
        request_value: CompactionRequest,
        abort: AbortSignal,
    ) -> BoxFuture<'a, Result<CompactionResult, ModelError>> {
        Box::pin(async move {
            self.validate_compaction(&request_value)?;
            if abort.is_aborted() {
                return Err(
                    ModelError::abort("Azure compaction was aborted before dispatch")
                        .with_stage(ErrorStage::NativeContextEncode),
                );
            }
            let (replay_binding, scope) = self
                .config
                .configured_native_context()?
                .expect("native compaction validation requires a configured scope");
            let encoded =
                compaction::encode(&request_value, &self.descriptor, &replay_binding, &scope)?;
            let (headers, resolved_binding, request_scope) =
                configured_headers(&self.config, &abort).await?;
            if request_scope != scope || resolved_binding != replay_binding {
                return Err(ModelError::native_context(
                    "Azure compaction request scope changed during configuration resolution",
                ));
            }
            let url = reqwest::Url::parse(&format!("{}/{COMPACT_PATH}", self.config.base_url))
                .map_err(|_| {
                    ModelError::invalid_request("invalid Azure Responses compact URL")
                        .with_stage(ErrorStage::NativeContextEncode)
                })?;
            let send = self
                .config
                .client
                .post(url)
                .headers(headers)
                .json(&encoded.body)
                .send();
            let response = tokio::select! {
                value = tokio::time::timeout(self.config.timeouts.headers, send) => value
                    .map_err(|_| ModelError::timeout("Azure compaction response headers timed out").with_stage(ErrorStage::ResponseHeaders))?
                    .map_err(|_| ModelError::transport("Azure compaction request failed").with_stage(ErrorStage::Connect))?,
                _ = abort.aborted() => return Err(ModelError::abort("Azure compaction was aborted before response headers").with_stage(ErrorStage::Connect)),
            };
            let head = crate::transport::response_head(
                &response,
                &["apim-request-id".into(), "x-request-id".into()],
            );
            if !response.status().is_success() {
                let status = response.status().as_u16();
                let error_headers = response.headers().clone();
                let request_id = head.request_id.clone();
                let (body, bytes) = crate::transport::read_error_body(
                    response,
                    &abort,
                    self.config.timeouts.stream_idle,
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
                .get(reqwest::header::CONTENT_TYPE)
                .and_then(|value| value.to_str().ok())
                .is_some_and(|value| value.starts_with("application/json"))
            {
                return Err(
                    ModelError::invalid_response("Azure compaction response is not JSON")
                        .with_stage(ErrorStage::ResponseHeaders),
                );
            }
            let (body, bytes) =
                compaction::read_body(response, &abort, self.config.timeouts.stream_idle).await?;
            let value = serde_json::from_slice(&body).map_err(|_| {
                ModelError::invalid_response("Azure compaction response is invalid JSON")
                    .with_stage(ErrorStage::NativeContextDecode)
                    .with_bytes_received(bytes)
            })?;
            let (native_context, usage) =
                compaction::parse_response(value, &self.descriptor, &scope, bytes)?;
            Ok(CompactionResult {
                native_context,
                usage,
                request: encoded.request,
                response: head,
            })
        })
    }
}

async fn configured_headers(
    config: &Config,
    abort: &AbortSignal,
) -> Result<(HeaderMap, JsonValue, NativeContextScope), ModelError> {
    let caller_headers = config.caller_headers()?;
    let mut headers = HeaderMap::new();
    match &config.auth {
        AzureOpenAiAuth::ApiKey(key) => {
            insert_header(&mut headers, "api-key", key.expose_secret())?;
        }
        AzureOpenAiAuth::Entra(provider) => {
            let token = tokio::select! {
                result = tokio::time::timeout(config.timeouts.credentials, provider()) => result
                    .map_err(|_| ModelError::timeout("Azure Entra credential timed out").with_stage(ErrorStage::Connect))??,
                _ = abort.aborted() => return Err(ModelError::abort("request was aborted during Azure credential resolution").with_stage(ErrorStage::Connect)),
            };
            if token.is_empty() {
                return Err(ModelError::new(
                    oven_sdk::ModelErrorKind::Auth,
                    "Azure Entra token provider returned an empty token",
                )
                .with_stage(ErrorStage::Connect));
            }
            insert_header(&mut headers, "authorization", &format!("Bearer {token}"))?;
        }
    }
    headers.extend(caller_headers);
    let (binding, scope) = config.native_context(&headers)?;
    Ok((headers, binding, scope))
}

fn insert_header(headers: &mut HeaderMap, name: &str, value: &str) -> Result<(), ModelError> {
    let name = reqwest::header::HeaderName::from_bytes(name.as_bytes())
        .map_err(|_| ModelError::invalid_request("invalid Azure OpenAI header name"))?;
    let value = reqwest::header::HeaderValue::from_str(value)
        .map_err(|_| ModelError::invalid_request("invalid Azure OpenAI header value"))?;
    headers.insert(name, value);
    Ok(())
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
                "Azure Responses stream ended before a semantic event",
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
            return Err(ModelError::transport("Azure Responses stream read failed")
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
            "Azure Responses stream ended before a terminal response event",
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
            ModelError::invalid_response("Azure Responses SSE event is invalid JSON")
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
