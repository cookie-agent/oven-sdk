//! Azure OpenAI Chat Completions language model.

use std::{collections::VecDeque, sync::Arc, time::Duration};

use oven_sdk::{
    AbortSignal, BoxFuture, BoxStream, ErrorStage, JsonValue, LanguageModel,
    LanguageModelDescriptor, ModelError, ModelId, NativeContextScope, ProviderMetadata,
    ReplayPolicy, Request, RequestMetadata, StreamItem, StreamPart, StreamResponse,
};
use reqwest::{Client, header::HeaderMap};

use crate::{
    chat::{request, state::State},
    configuration::{
        AzureMaxTokensField, AzureOpenAiAuth, AzureOpenAiChatConfig, AzureReasoningField,
        AzureStructuredOutputSupport, AzureSystemMessageRole, Config, build_chat,
    },
    error::classify_error,
    transport::AzureOpenAiTimeouts,
    wire::chat::PATH,
};

/// A configured Azure OpenAI Chat Completions deployment.
#[derive(Clone)]
pub struct AzureOpenAiChatModel {
    pub(crate) config: Arc<Config>,
    descriptor: LanguageModelDescriptor,
}

impl AzureOpenAiChatModel {
    /// Constructs one registry-free Azure Chat Completions model.
    pub fn new(config: AzureOpenAiChatConfig) -> Result<Self, ModelError> {
        let (config, descriptor) = build_chat(config)?;
        Ok(Self { config, descriptor })
    }

    /// Returns the configured deployment-name model ID.
    #[must_use]
    pub fn model_id(&self) -> &ModelId {
        &self.descriptor.identity.model_id
    }
}

#[derive(Clone)]
struct OwnedProfile {
    system_role: AzureSystemMessageRole,
    max_tokens_field: AzureMaxTokensField,
    stream_usage: bool,
    structured_output: AzureStructuredOutputSupport,
    reasoning_field: AzureReasoningField,
    omit_reasoning_sampling: bool,
}

impl OwnedProfile {
    fn wire(&self) -> request::ChatWireProfile {
        request::ChatWireProfile {
            system_role: self.system_role,
            max_tokens_field: self.max_tokens_field,
            stream_usage: self.stream_usage,
            structured_output: self.structured_output,
            reasoning_field: self.reasoning_field,
            omit_reasoning_sampling: self.omit_reasoning_sampling,
        }
    }
}

#[derive(Clone)]
struct Settings {
    client: Client,
    base_url: String,
    headers: HeaderMap,
    query: Vec<(String, String)>,
    timeouts: AzureOpenAiTimeouts,
    replay_policy: ReplayPolicy,
    request_id_headers: Vec<String>,
    strict_sse_content_type: bool,
    profile: OwnedProfile,
    replay_binding: JsonValue,
    replay_scope: NativeContextScope,
}

impl LanguageModel for AzureOpenAiChatModel {
    fn descriptor(&self) -> &LanguageModelDescriptor {
        &self.descriptor
    }

    fn validate_request(&self, request_value: &Request) -> Result<(), ModelError> {
        let profile = configured_owned(&self.config);
        let options = crate::options::chat_options(request_value)?;
        request::validate_request(
            request_value,
            &options,
            &self.descriptor().capabilities,
            &profile.wire(),
        )
    }

    fn supports_request(&self, request: &Request) -> bool {
        self.validate_request(request).is_ok()
    }

    fn stream<'a>(
        &'a self,
        request_value: Request,
        abort: AbortSignal,
    ) -> BoxFuture<'a, Result<StreamResponse, ModelError>> {
        Box::pin(async move {
            let profile = configured_owned(&self.config);
            let options = crate::options::chat_options(&request_value)?;
            request::validate_request(
                &request_value,
                &options,
                &self.descriptor().capabilities,
                &profile.wire(),
            )?;
            let settings =
                configured_settings(&self.config, &request_value.header_context, &abort).await?;
            start_stream(
                self.descriptor().clone(),
                request_value,
                options,
                abort,
                settings,
            )
            .await
        })
    }
}

fn configured_owned(config: &Config) -> OwnedProfile {
    OwnedProfile {
        system_role: config.completions.system_role,
        max_tokens_field: config.completions.max_tokens_field,
        stream_usage: config.completions.stream_usage,
        structured_output: config.completions.structured_output,
        reasoning_field: config.completions.reasoning_field,
        omit_reasoning_sampling: config.completions.omit_reasoning_sampling,
    }
}

async fn configured_settings(
    config: &Config,
    context: &oven_sdk::HeaderContext,
    abort: &AbortSignal,
) -> Result<Settings, ModelError> {
    let mut headers = config.caller_headers(context)?;
    if !oven_sdk::contains_auth_owned_header(&headers) {
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
    }
    let (replay_binding, replay_scope) = config.native_context(&headers)?;
    Ok(Settings {
        client: config.client.clone(),
        base_url: config.base_url.clone(),
        headers,
        query: config.query.clone(),
        timeouts: config.timeouts.clone(),
        replay_policy: config.capabilities.replay.policy,
        request_id_headers: vec!["apim-request-id".into(), "x-request-id".into()],
        strict_sse_content_type: true,
        profile: configured_owned(config),
        replay_binding,
        replay_scope,
    })
}

fn insert_header(headers: &mut HeaderMap, name: &str, value: &str) -> Result<(), ModelError> {
    let name = reqwest::header::HeaderName::from_bytes(name.as_bytes())
        .map_err(|_| ModelError::invalid_request("invalid Azure OpenAI header name"))?;
    let value = reqwest::header::HeaderValue::from_str(value)
        .map_err(|_| ModelError::invalid_request("invalid Azure OpenAI header value"))?;
    headers.insert(name, value);
    Ok(())
}

async fn start_stream(
    descriptor: LanguageModelDescriptor,
    request_value: Request,
    options: crate::options::AzureOpenAiChatOptions,
    abort: AbortSignal,
    settings: Settings,
) -> Result<StreamResponse, ModelError> {
    if abort.is_aborted() {
        return Err(ModelError::abort("request was aborted before dispatch")
            .with_stage(ErrorStage::Connect));
    }
    let encoded = request::encode_request(
        &request_value,
        &options,
        &descriptor,
        settings.replay_policy,
        &settings.profile.wire(),
        &settings.replay_binding,
        &settings.replay_scope,
    )?;
    let mut url = reqwest::Url::parse(&format!("{}/{PATH}", settings.base_url))
        .map_err(|_| ModelError::invalid_request("invalid Chat endpoint URL"))?;
    if !settings.query.is_empty() {
        url.query_pairs_mut().extend_pairs(settings.query.iter());
    }
    let send = settings
        .client
        .post(url)
        .headers(settings.headers)
        .json(&encoded.body)
        .send();
    let response = tokio::select! {
        value = tokio::time::timeout(settings.timeouts.headers, send) => value
            .map_err(|_| ModelError::timeout("response headers timed out").with_stage(ErrorStage::ResponseHeaders))?
            .map_err(|_| ModelError::transport("Azure OpenAI Chat request failed").with_stage(ErrorStage::Connect))?,
        _ = abort.aborted() => return Err(ModelError::abort("request was aborted before response headers").with_stage(ErrorStage::Connect)),
    };
    let mut head = crate::transport::response_head(&response, &settings.request_id_headers);
    if !response.status().is_success() {
        let status = response.status().as_u16();
        let headers = response.headers().clone();
        let request_id = head.request_id.clone();
        let (body, bytes) =
            crate::transport::read_error_body(response, &abort, settings.timeouts.stream_idle)
                .await?;
        return Err(classify_error(
            status,
            &body,
            request_id,
            ErrorStage::ResponseBody,
            bytes,
            &headers,
        ));
    }
    let response_headers = response.headers().clone();
    if settings.strict_sse_content_type
        && !response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .is_some_and(|value| value.starts_with("text/event-stream"))
    {
        return Err(ModelError::invalid_response(
            "Azure OpenAI Chat response is not an SSE content type",
        )
        .with_stage(ErrorStage::ResponseHeaders));
    }
    let mut live = LiveState {
        bytes: Box::pin(response.bytes_stream()),
        parser: crate::sse::Parser::new("Azure OpenAI SSE contains invalid UTF-8")
            .clear_name_on_empty_event(),
        state: State::new(
            descriptor.adapter_id.clone(),
            settings.replay_policy,
            settings.profile.reasoning_field,
            settings.replay_binding,
            settings.replay_scope,
        ),
        queue: VecDeque::from([Ok(StreamPart::StreamStart {
            warnings: encoded.warnings,
        })]),
        pending_events: VecDeque::new(),
        pending_error: None,
        deadline: oven_sdk::provider_support::StreamReadDeadline::new(
            tokio::time::sleep(settings.timeouts.stream_idle),
            &abort,
        ),
        idle: settings.timeouts.stream_idle,
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
}

struct LiveState {
    bytes: BoxStream<'static, Result<bytes::Bytes, reqwest::Error>>,
    parser: crate::sse::Parser,
    state: State,
    queue: VecDeque<StreamItem>,
    pending_events: VecDeque<crate::sse::Event>,
    pending_error: Option<ModelError>,
    deadline: oven_sdk::provider_support::StreamReadDeadline<tokio::time::Sleep>,
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
                "Azure Chat stream ended before a semantic event",
            )
            .with_bytes_received(live.count));
        }
    }
}

async fn read_live(live: &mut LiveState, stop_after_event: bool) -> Result<bool, ModelError> {
    if !live.pending_events.is_empty() {
        return process_events(live, stop_after_event);
    }
    let next = match live
        .deadline
        .next(live.bytes.as_mut(), |timer| {
            timer.reset(tokio::time::Instant::now() + live.idle);
        })
        .await
    {
        oven_sdk::provider_support::StreamRead::Aborted => {
            return Err(ModelError::abort("stream was aborted")
                .with_stage(ErrorStage::StreamRead)
                .with_bytes_received(live.count));
        }
        oven_sdk::provider_support::StreamRead::TimedOut => {
            return Err(ModelError::timeout("stream idle timeout")
                .with_stage(ErrorStage::StreamRead)
                .with_bytes_received(live.count));
        }
        oven_sdk::provider_support::StreamRead::Item(value) => value,
    };
    match next {
        Some(Ok(chunk)) => {
            live.count = live.count.saturating_add(chunk.len() as u64);
            live.parser
                .feed_into(&chunk, &mut live.pending_events)
                .map_err(|error| error.with_bytes_received(live.count))?;
        }
        Some(Err(_)) => {
            return Err(ModelError::transport("Azure Chat stream read failed")
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
        let mut parts = Vec::new();
        live.state.finish(false, &mut parts, live.count)?;
        live.queue.extend(parts.into_iter().map(Ok));
        return Ok(true);
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
        if event.data.trim() == "[DONE]" {
            let mut parts = Vec::new();
            live.state.finish(true, &mut parts, live.count)?;
            live.queue.extend(parts.into_iter().map(Ok));
            live.eof = true;
            live.pending_events.clear();
            return Ok(true);
        }
        let value: JsonValue = serde_json::from_str(&event.data).map_err(|_| {
            ModelError::invalid_response("Azure Chat SSE event is invalid JSON")
                .with_stage(ErrorStage::StreamDecode)
                .with_bytes_received(live.count)
        })?;
        if value.get("error").is_some() {
            let status = stream_error_status(&value);
            let error = classify_error(
                status,
                event.data.as_bytes(),
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
