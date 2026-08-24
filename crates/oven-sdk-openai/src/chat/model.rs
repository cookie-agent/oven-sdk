//! Official and compatible Chat Completions language models.

use std::{collections::VecDeque, sync::Arc, time::Duration};

use oven_sdk::{
    AbortSignal, AdapterId, BoxFuture, BoxStream, ErrorStage, JsonValue, LanguageModel,
    LanguageModelDescriptor, ModelConfig, ModelError, ModelId, ModelIdentity, NativeContextScope,
    ProviderMetadata, Request, RequestMetadata, StreamItem, StreamPart, StreamResponse,
};
use reqwest::{Client, header::HeaderMap};

use crate::{
    chat::{request, state::State},
    configuration::{
        MaxTokensField, OpenAiAuth, OpenAiChatSettings, OpenAiCompatibleAuth,
        OpenAiCompatibleChatSettings, ReasoningField, StructuredOutputSupport, SystemMessageRole,
        build_client, canonical_endpoint, compatible_base_headers, compatible_headers,
        header_scope_component, official_base_headers, official_headers, replay_resource_id,
        validate_chat_declaration, validate_routing_discriminator,
    },
    error::classify_error,
    transport::OpenAiTimeouts,
    wire::{OPENAI_CHAT_ADAPTER_ID, OPENAI_RESPONSES_ADAPTER_ID, chat::PATH},
};

/// A configured official OpenAI Chat Completions model.
#[derive(Clone)]
pub struct OpenAiChatModel {
    runtime: Arc<Runtime>,
}

/// A configured OpenAI-compatible Chat Completions model.
#[derive(Clone)]
pub struct OpenAiCompatibleChatModel {
    runtime: Arc<Runtime>,
}

impl OpenAiChatModel {
    /// Constructs one official Chat Completions model from explicit configuration.
    pub fn new(config: ModelConfig<OpenAiAuth, OpenAiChatSettings>) -> Result<Self, ModelError> {
        config.validate()?;
        validate_chat_declaration(
            &config.model.capabilities,
            config.settings.max_tokens_field,
            config.settings.structured_output,
            config.settings.reasoning_field,
            config.settings.stream_usage,
        )?;
        validate_routing_discriminator(
            config.provider.headers.dynamic_headers.is_some(),
            config.settings.routing_discriminator.as_deref(),
        )?;
        let adapter_id = AdapterId::new(OPENAI_CHAT_ADAPTER_ID);
        let descriptor = LanguageModelDescriptor::new(
            ModelIdentity::new(config.provider.id.clone(), config.model.id.clone())?,
            adapter_id.clone(),
            config.model.capabilities.clone(),
        )?;
        let profile = OwnedProfile {
            compatible: false,
            system_role: config.settings.system_message_role,
            max_tokens_field: config.settings.max_tokens_field,
            stream_usage: config.settings.stream_usage,
            structured_output: config.settings.structured_output,
            reasoning_field: config.settings.reasoning_field,
        };
        let scope = replay_scope(
            &descriptor,
            ScopeInputs {
                api: &canonical_endpoint(config.provider.api.as_url()),
                adapter_id: &adapter_id,
                profile: &profile,
                query: &[],
                request_id_headers: &["x-request-id".into()],
                strict_sse_content_type: true,
                organization: config.provider.auth.organization.as_deref(),
                project: config.provider.auth.project.as_deref(),
                routing_discriminator: config.settings.routing_discriminator.as_deref(),
                static_headers: &header_scope_component(&config.provider.headers),
            },
        )?;
        let client = build_client(config.settings.client, &config.settings.timeouts)?;
        let base_headers = official_base_headers(&config.provider.auth, &config.provider.headers)?;
        Ok(Self {
            runtime: Arc::new(Runtime {
                descriptor,
                scope,
                api: config.provider.api.as_url().to_string(),
                auth: Authentication::Official(config.provider.auth),
                headers: config.provider.headers,
                base_headers,
                client,
                query: Vec::new(),
                timeouts: config.settings.timeouts,
                request_id_headers: vec!["x-request-id".into()],
                strict_sse_content_type: true,
                profile,
            }),
        })
    }

    /// Returns the configured model ID.
    #[must_use]
    pub fn model_id(&self) -> &ModelId {
        &self.runtime.descriptor.identity.model_id
    }
}

impl OpenAiCompatibleChatModel {
    /// Constructs one OpenAI-compatible Chat model from explicit configuration.
    pub fn new(
        config: ModelConfig<OpenAiCompatibleAuth, OpenAiCompatibleChatSettings>,
    ) -> Result<Self, ModelError> {
        config.validate()?;
        config.settings.adapter_id.validate()?;
        if matches!(
            config.settings.adapter_id.as_str(),
            OPENAI_CHAT_ADAPTER_ID | OPENAI_RESPONSES_ADAPTER_ID
        ) {
            return Err(ModelError::invalid_request(
                "official oven.openai adapter IDs are reserved",
            ));
        }
        validate_chat_declaration(
            &config.model.capabilities,
            config.settings.max_tokens_field,
            config.settings.structured_output,
            config.settings.reasoning_field,
            config.settings.stream_usage,
        )?;
        validate_routing_discriminator(
            config.provider.headers.dynamic_headers.is_some()
                || config.provider.auth.header_provider.is_some(),
            config.settings.routing_discriminator.as_deref(),
        )?;
        let adapter_id = config.settings.adapter_id.clone();
        let descriptor = LanguageModelDescriptor::new(
            ModelIdentity::new(config.provider.id.clone(), config.model.id.clone())?,
            adapter_id.clone(),
            config.model.capabilities.clone(),
        )?;
        let profile = OwnedProfile {
            compatible: true,
            system_role: config.settings.system_message_role,
            max_tokens_field: config.settings.max_tokens_field,
            stream_usage: config.settings.stream_usage,
            structured_output: config.settings.structured_output,
            reasoning_field: config.settings.reasoning_field,
        };
        let scope = replay_scope(
            &descriptor,
            ScopeInputs {
                api: &canonical_endpoint(config.provider.api.as_url()),
                adapter_id: &adapter_id,
                profile: &profile,
                query: &config.settings.query,
                request_id_headers: &config.settings.request_id_headers,
                strict_sse_content_type: config.settings.strict_sse_content_type,
                organization: None,
                project: None,
                routing_discriminator: config.settings.routing_discriminator.as_deref(),
                static_headers: &header_scope_component(&config.provider.headers),
            },
        )?;
        let client = build_client(config.settings.client, &config.settings.timeouts)?;
        let base_headers = compatible_base_headers(&config.provider.headers)?;
        Ok(Self {
            runtime: Arc::new(Runtime {
                descriptor,
                scope,
                api: config.provider.api.as_url().to_string(),
                auth: Authentication::Compatible(config.provider.auth),
                headers: config.provider.headers,
                base_headers,
                client,
                query: config.settings.query,
                timeouts: config.settings.timeouts,
                request_id_headers: config.settings.request_id_headers,
                strict_sse_content_type: config.settings.strict_sse_content_type,
                profile,
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
struct OwnedProfile {
    compatible: bool,
    system_role: SystemMessageRole,
    max_tokens_field: MaxTokensField,
    stream_usage: bool,
    structured_output: StructuredOutputSupport,
    reasoning_field: ReasoningField,
}

impl OwnedProfile {
    fn wire(&self) -> request::ChatWireProfile {
        request::ChatWireProfile {
            compatible: self.compatible,
            system_role: self.system_role,
            max_tokens_field: self.max_tokens_field,
            stream_usage: self.stream_usage,
            structured_output: self.structured_output,
            reasoning_field: self.reasoning_field,
        }
    }
}

#[derive(Clone)]
struct Runtime {
    descriptor: LanguageModelDescriptor,
    scope: NativeContextScope,
    api: String,
    auth: Authentication,
    headers: oven_sdk::HeaderConfig,
    base_headers: HeaderMap,
    client: Client,
    query: Vec<(String, String)>,
    timeouts: OpenAiTimeouts,
    request_id_headers: Vec<String>,
    strict_sse_content_type: bool,
    profile: OwnedProfile,
}

#[derive(Clone)]
enum Authentication {
    Official(OpenAiAuth),
    Compatible(OpenAiCompatibleAuth),
}

impl LanguageModel for OpenAiChatModel {
    fn descriptor(&self) -> &LanguageModelDescriptor {
        &self.runtime.descriptor
    }

    fn validate_request(&self, request_value: &Request) -> Result<(), ModelError> {
        let options = request::parse_options(request_value)?;
        request::validate_request(
            request_value,
            &options,
            &self.runtime.descriptor.capabilities,
            &self.runtime.profile.wire(),
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
            let options = request::parse_options(&request_value)?;
            request::validate_request(
                &request_value,
                &options,
                &self.runtime.descriptor.capabilities,
                &self.runtime.profile.wire(),
            )?;
            start_stream(Arc::clone(&self.runtime), request_value, options, abort).await
        })
    }
}

impl LanguageModel for OpenAiCompatibleChatModel {
    fn descriptor(&self) -> &LanguageModelDescriptor {
        &self.runtime.descriptor
    }

    fn validate_request(&self, request_value: &Request) -> Result<(), ModelError> {
        let options = request::parse_options(request_value)?;
        request::validate_request(
            request_value,
            &options,
            &self.runtime.descriptor.capabilities,
            &self.runtime.profile.wire(),
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
            let options = request::parse_options(&request_value)?;
            request::validate_request(
                &request_value,
                &options,
                &self.runtime.descriptor.capabilities,
                &self.runtime.profile.wire(),
            )?;
            start_stream(Arc::clone(&self.runtime), request_value, options, abort).await
        })
    }
}

struct ScopeInputs<'a> {
    api: &'a str,
    adapter_id: &'a AdapterId,
    profile: &'a OwnedProfile,
    query: &'a [(String, String)],
    request_id_headers: &'a [String],
    strict_sse_content_type: bool,
    organization: Option<&'a str>,
    project: Option<&'a str>,
    routing_discriminator: Option<&'a str>,
    static_headers: &'a str,
}

fn replay_scope(
    descriptor: &LanguageModelDescriptor,
    inputs: ScopeInputs<'_>,
) -> Result<NativeContextScope, ModelError> {
    let mut query = inputs.query.to_vec();
    query.sort();
    let query = serde_json::to_string(&query)
        .map_err(|_| ModelError::invalid_request("could not encode compatible query settings"))?;
    let mut request_id_headers = inputs.request_id_headers.to_vec();
    request_id_headers.sort();
    let request_id_headers = serde_json::to_string(&request_id_headers)
        .map_err(|_| ModelError::invalid_request("could not encode request-ID settings"))?;
    let resource_id = replay_resource_id(&[
        inputs.api,
        inputs.adapter_id.as_str(),
        if inputs.profile.compatible {
            "compatible"
        } else {
            "official"
        },
        match inputs.profile.system_role {
            SystemMessageRole::System => "system",
            SystemMessageRole::Developer => "developer",
            SystemMessageRole::Omit => "omit",
        },
        match inputs.profile.max_tokens_field {
            MaxTokensField::MaxTokens => "max_tokens",
            MaxTokensField::MaxCompletionTokens => "max_completion_tokens",
            MaxTokensField::Omit => "omit",
        },
        match inputs.profile.structured_output {
            StructuredOutputSupport::Unsupported => "structured_none",
            StructuredOutputSupport::JsonObject => "json_object",
            StructuredOutputSupport::JsonSchema => "json_schema",
        },
        match inputs.profile.reasoning_field {
            ReasoningField::None => "reasoning_none",
            ReasoningField::ReasoningContent => "reasoning_content",
            ReasoningField::Reasoning => "reasoning",
        },
        if inputs.profile.stream_usage {
            "stream_usage"
        } else {
            "no_stream_usage"
        },
        if inputs.strict_sse_content_type {
            "strict_sse"
        } else {
            "lenient_sse"
        },
        &query,
        &request_id_headers,
        inputs.organization.unwrap_or(""),
        inputs.project.unwrap_or(""),
        inputs.routing_discriminator.unwrap_or(""),
        inputs.static_headers,
    ])?;
    NativeContextScope::new(
        descriptor.identity.provider_id.clone(),
        descriptor.identity.model_id.clone(),
        resource_id,
    )
}

async fn start_stream(
    runtime: Arc<Runtime>,
    request_value: Request,
    options: request::ParsedOptions,
    abort: AbortSignal,
) -> Result<StreamResponse, ModelError> {
    if abort.is_aborted() {
        return Err(ModelError::abort("request was aborted before dispatch")
            .with_stage(ErrorStage::Connect));
    }
    let policy = runtime.descriptor.capabilities.replay.policy;
    let encoded = request::encode_request(
        &request_value,
        &options,
        &runtime.descriptor,
        &runtime.scope,
        policy,
        &runtime.profile.wire(),
    )?;
    let mut url = reqwest::Url::parse(&format!("{}/{PATH}", runtime.api.trim_end_matches('/')))
        .map_err(|_| ModelError::invalid_request("invalid Chat endpoint URL"))?;
    url.query_pairs_mut().extend_pairs(runtime.query.iter());
    let headers = match &runtime.auth {
        Authentication::Official(auth) => {
            official_headers(auth, &runtime.base_headers, &runtime.headers)?
        }
        Authentication::Compatible(auth) => {
            compatible_headers(auth, &runtime.base_headers, &runtime.headers)?
        }
    };
    let send = runtime
        .client
        .post(url)
        .headers(headers)
        .json(&encoded.body)
        .send();
    let response = tokio::select! {
        value = tokio::time::timeout(runtime.timeouts.headers, send) => value
            .map_err(|_| ModelError::timeout("response headers timed out").with_stage(ErrorStage::ResponseHeaders))?
            .map_err(|_| ModelError::transport("OpenAI Chat request failed").with_stage(ErrorStage::Connect))?,
        _ = abort.aborted() => return Err(ModelError::abort("request was aborted before response headers").with_stage(ErrorStage::Connect)),
    };
    let mut head = crate::transport::response_head(&response, &runtime.request_id_headers);
    if !response.status().is_success() {
        let status = response.status().as_u16();
        let headers = response.headers().clone();
        let request_id = head.request_id.clone();
        let (body, bytes) =
            crate::transport::read_error_body(response, &abort, runtime.timeouts.stream_idle)
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
    if runtime.strict_sse_content_type
        && !response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .is_some_and(|value| value.starts_with("text/event-stream"))
    {
        return Err(ModelError::invalid_response(
            "OpenAI Chat response is not an SSE content type",
        )
        .with_stage(ErrorStage::ResponseHeaders));
    }
    let mut live = LiveState {
        bytes: Box::pin(response.bytes_stream()),
        parser: crate::sse::Parser::new("OpenAI SSE contains invalid UTF-8")
            .clear_name_on_empty_event(),
        state: State::new(
            runtime.descriptor.adapter_id.clone(),
            runtime.scope.clone(),
            policy,
            runtime.profile.reasoning_field,
        ),
        queue: VecDeque::from([Ok(StreamPart::StreamStart {
            warnings: encoded.warnings,
        })]),
        pending_events: VecDeque::new(),
        pending_error: None,
        deadline: oven_sdk::provider_support::StreamReadDeadline::new(
            tokio::time::sleep(runtime.timeouts.stream_idle),
            &abort,
        ),
        idle: runtime.timeouts.stream_idle,
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
                "OpenAI Chat stream ended before a semantic event",
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
            return Err(ModelError::transport("OpenAI Chat stream read failed")
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
            ModelError::invalid_response("OpenAI Chat SSE event is invalid JSON")
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
