//! Bedrock language-model implementation and resource URL construction.

use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
};

use oven_sdk::{
    AbortSignal, AdapterId, BoxFuture, Capability, CompactionCapability, CompleteResult,
    ErrorStage, HeaderConfig, LanguageModel, LanguageModelDescriptor, MediaSourceSupport,
    ModelCapabilities, ModelConfig, ModelError, ModelId, ModelIdentity, NativeContextScope,
    ProviderMetadata, ReplayCapability, Request, RequestMetadata, ResourceId, StreamPart,
    StreamResponse,
};
use reqwest::header::{ACCEPT, AUTHORIZATION, CONTENT_TYPE, HeaderValue};
use sha2::{Digest, Sha256};
use url::Url;

use crate::{
    AwsCredentials, AwsCredentialsProvider, BEDROCK_CONVERSE_ADAPTER_ID, BEDROCK_PROVIDER_ID,
    BedrockAuth, BedrockConverseSettings, BedrockReasoningWireFormat, BedrockStructuredOutput,
};

#[derive(Clone)]
struct Config {
    region: String,
    base_url: String,
    credentials: Arc<dyn AwsCredentialsProvider>,
    client: reqwest::Client,
    headers: HeaderConfig,
    timeouts: crate::BedrockTimeouts,
    capabilities: ModelCapabilities,
    reasoning_wire_format: BedrockReasoningWireFormat,
    signed_reasoning: bool,
    structured_output: BedrockStructuredOutput,
    max_event_bytes: usize,
    native_context_scope: NativeContextScope,
    descriptor: LanguageModelDescriptor,
}

/// A configured Amazon Bedrock Runtime model.
#[derive(Clone)]
pub struct BedrockModel {
    config: Arc<Config>,
}

impl BedrockModel {
    /// Constructs one registry-free Bedrock Converse model from explicit configuration.
    pub fn new(
        config: ModelConfig<BedrockAuth, BedrockConverseSettings>,
    ) -> Result<Self, ModelError> {
        config.validate()?;
        if config.provider.id.as_str() != BEDROCK_PROVIDER_ID {
            return Err(ModelError::invalid_request(format!(
                "Bedrock provider ID must be {BEDROCK_PROVIDER_ID}"
            )));
        }
        validate_model_id(&config.model.id)?;
        validate_adapter_declaration(&config.model.capabilities)?;
        validate_settings(&config.model.capabilities, &config.settings)?;
        validate_headers(config.provider.headers.static_headers.as_map())?;
        if config.provider.api.as_url().query().is_some() {
            return Err(ModelError::invalid_request(
                "Bedrock API endpoint must not contain a query string",
            ));
        }
        let credentials: Arc<dyn AwsCredentialsProvider> = match config.provider.auth {
            BedrockAuth::Static(credentials) => {
                validate_credentials(&credentials)?;
                Arc::new(move || {
                    let credentials = credentials.clone();
                    async move { Ok(credentials) }
                })
            }
            BedrockAuth::Provider(provider) => provider,
        };
        let client = match config.settings.client.clone() {
            Some(client) => client,
            None => reqwest::Client::builder()
                .connect_timeout(config.settings.timeouts.connect)
                .build()
                .map_err(|_| {
                    ModelError::transport("could not construct Bedrock HTTP client")
                        .with_stage(ErrorStage::Connect)
                })?,
        };
        let base_url = config
            .provider
            .api
            .as_url()
            .as_str()
            .trim_end_matches('/')
            .to_owned();
        let native_context_scope = native_context_scope(
            &config.provider.id,
            &config.model.id,
            &base_url,
            &config.settings,
        )?;
        let identity = ModelIdentity::new(config.provider.id, config.model.id)?;
        let mut provider_metadata = ProviderMetadata::new();
        provider_metadata.insert(
            "bedrock.region".into(),
            serde_json::Value::String(config.settings.region.clone()),
        );
        let descriptor = LanguageModelDescriptor::new(
            identity,
            AdapterId::new(BEDROCK_CONVERSE_ADAPTER_ID),
            config.model.capabilities.clone(),
        )?
        .with_provider_metadata(provider_metadata);
        Ok(Self {
            config: Arc::new(Config {
                region: config.settings.region,
                base_url,
                credentials,
                client,
                headers: config.provider.headers,
                timeouts: config.settings.timeouts,
                capabilities: config.model.capabilities,
                reasoning_wire_format: config.settings.reasoning_wire_format,
                signed_reasoning: config.settings.signed_reasoning,
                structured_output: config.settings.structured_output,
                max_event_bytes: config.settings.event_stream.max_message_bytes,
                native_context_scope,
                descriptor,
            }),
        })
    }

    /// Returns the selected model ID or resource ARN.
    #[must_use]
    pub fn model_id(&self) -> &ModelId {
        &self.config.descriptor.identity.model_id
    }

    /// Returns the exact scope required for native replay compatibility.
    #[must_use]
    pub fn native_context_scope(&self) -> &NativeContextScope {
        &self.config.native_context_scope
    }

    /// Returns the exact non-streaming `Converse` resource URL.
    pub fn converse_url(&self) -> Result<Url, ModelError> {
        self.endpoint(false)
    }

    /// Returns the exact streaming `ConverseStream` resource URL.
    pub fn converse_stream_url(&self) -> Result<Url, ModelError> {
        self.endpoint(true)
    }

    /// Calls Bedrock's non-streaming `Converse` operation and drains it through
    /// the same strict normalized collector used by [`LanguageModel::complete`].
    pub fn converse<'a>(
        &'a self,
        request: Request,
        abort: AbortSignal,
    ) -> BoxFuture<'a, Result<CompleteResult, ModelError>> {
        Box::pin(async move {
            let response = self.execute(request, abort, false).await?;
            collect_direct(response, self.descriptor().clone()).await
        })
    }

    fn endpoint(&self, streaming: bool) -> Result<Url, ModelError> {
        let operation = if streaming {
            "converse-stream"
        } else {
            "converse"
        };
        let encoded = encode_path_segment(self.model_id().as_str());
        Url::parse(&format!(
            "{}/model/{encoded}/{operation}",
            self.config.base_url
        ))
        .map_err(|_| {
            ModelError::invalid_request("Bedrock resource URL is invalid")
                .with_stage(ErrorStage::RequestEncoding)
        })
    }

    fn execute<'a>(
        &'a self,
        request: Request,
        abort: AbortSignal,
        streaming: bool,
    ) -> BoxFuture<'a, Result<StreamResponse, ModelError>> {
        Box::pin(async move {
            let options = crate::request::options(&request)?;
            crate::request::validate_request(
                &request,
                &options,
                &self.config.capabilities,
                self.config.reasoning_wire_format,
                self.config.signed_reasoning,
                self.config.structured_output,
            )?;
            if abort.is_aborted() {
                return Err(ModelError::abort("request was aborted before dispatch")
                    .with_stage(ErrorStage::Connect));
            }
            let descriptor = self.descriptor();
            let encoded = crate::request::encode_request(
                &request,
                &options,
                descriptor,
                &self.config.native_context_scope,
                crate::request::EncodeSettings {
                    reasoning_wire_format: self.config.reasoning_wire_format,
                    signed_reasoning: self.config.signed_reasoning,
                    structured_output: self.config.structured_output,
                    streaming,
                },
            )?;
            let body = serde_json::to_vec(&encoded.body).map_err(|_| {
                ModelError::invalid_request("could not serialize Bedrock Converse request")
                    .with_stage(ErrorStage::RequestEncoding)
            })?;
            crate::request::validate_serialized_body(&request, body.len())?;
            let mut headers = self.config.headers.static_headers.as_map().clone();
            if let Some(provider) = &self.config.headers.dynamic_headers {
                let dynamic = provider.headers(&request.header_context)?;
                let dynamic = dynamic.as_map();
                validate_headers(dynamic)?;
                headers.extend(dynamic.clone());
            }
            headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
            if streaming {
                headers.insert(
                    ACCEPT,
                    HeaderValue::from_static("application/vnd.amazon.eventstream"),
                );
            } else {
                headers.insert(ACCEPT, HeaderValue::from_static("application/json"));
            }
            let url = self.endpoint(streaming)?;
            if !headers.contains_key(AUTHORIZATION) {
                let credentials = self.resolve_credentials(&abort).await?;
                crate::sigv4::sign(
                    "POST",
                    &url,
                    &body,
                    &mut headers,
                    &self.config.region,
                    &credentials,
                )?;
            }
            let send = self
                .config
                .client
                .post(url)
                .headers(headers)
                .body(body)
                .send();
            let response = tokio::select! {
                value = tokio::time::timeout(self.config.timeouts.headers, send) => value
                    .map_err(|_| ModelError::timeout("Bedrock response headers timed out").with_stage(ErrorStage::ResponseHeaders))?
                    .map_err(|_| ModelError::transport("Bedrock Converse request failed").with_stage(ErrorStage::Connect))?,
                _ = abort.aborted() => return Err(ModelError::abort("request was aborted before Bedrock response headers").with_stage(ErrorStage::ResponseHeaders)),
            };
            let mut head = crate::transport::response_head(&response);
            if !response.status().is_success() {
                let status = response.status().as_u16();
                let request_id = head.request_id.clone();
                let error_headers = response.headers().clone();
                let (body, count) = crate::transport::read_body(
                    response,
                    &abort,
                    self.config.timeouts.stream_idle,
                    oven_sdk::SanitizedBody::MAX_BYTES,
                )
                .await?;
                return Err(crate::error::classify_error(
                    status,
                    &body,
                    request_id,
                    ErrorStage::ResponseBody,
                    count,
                    &error_headers,
                ));
            }
            let request_metadata = RequestMetadata {
                replay: encoded.replay,
                provider_metadata: ProviderMetadata::new(),
            };
            if !streaming {
                const MAX_SUCCESS_BODY: usize = 32 * 1024 * 1024;
                let request_id = head.request_id.clone();
                let (body, count) = crate::transport::read_body(
                    response,
                    &abort,
                    self.config.timeouts.stream_idle,
                    MAX_SUCCESS_BODY,
                )
                .await?;
                if count > MAX_SUCCESS_BODY as u64 {
                    return Err(ModelError::invalid_response(
                        "Bedrock Converse response exceeds 32 MiB",
                    )
                    .with_stage(ErrorStage::ResponseBody)
                    .with_bytes_received(count));
                }
                let value: serde_json::Value = serde_json::from_slice(&body).map_err(|_| {
                    ModelError::invalid_response("Bedrock Converse response is invalid JSON")
                        .with_stage(ErrorStage::ResponseBody)
                        .with_bytes_received(count)
                })?;
                let (mut parts, metadata) = crate::stream::normalize_single(
                    value,
                    self.stream_configuration(),
                    request_id,
                    request.stream_options.include_raw,
                    count,
                )?;
                if let Some(StreamPart::StreamStart { warnings }) = parts.first_mut() {
                    *warnings = encoded.warnings;
                }
                head.response_metadata.extend(metadata);
                return Ok(StreamResponse::new(Box::pin(futures_util::stream::iter(
                    parts.into_iter().map(Ok),
                )))
                .with_request(request_metadata)
                .with_response(head));
            }
            let mut live = crate::stream::LiveState {
                bytes: Box::pin(response.bytes_stream()),
                decoder: crate::eventstream::Decoder::new(self.config.max_event_bytes),
                state: crate::stream::State::new(self.stream_configuration()),
                queue: VecDeque::from([Ok(StreamPart::StreamStart {
                    warnings: encoded.warnings,
                })]),
                pending_messages: VecDeque::new(),
                terminal_queue: VecDeque::new(),
                pending_error: None,
                deadline: oven_sdk::provider_support::StreamReadDeadline::new(
                    tokio::time::sleep(self.config.timeouts.stream_idle),
                    &abort,
                ),
                idle: self.config.timeouts.stream_idle,
                count: 0,
                eof: false,
                include_raw: request.stream_options.include_raw,
            };
            live.state.set_request_id(head.request_id.clone());
            crate::stream::early_peek(&mut live).await?;
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
                    if let Err(error) = crate::stream::read_live(&mut live, false).await {
                        live.pending_error = Some(error);
                    }
                }
            });
            Ok(StreamResponse::new(Box::pin(stream))
                .with_request(request_metadata)
                .with_response(head))
        })
    }

    async fn resolve_credentials(
        &self,
        abort: &AbortSignal,
    ) -> Result<crate::AwsCredentials, ModelError> {
        let credentials = tokio::select! {
            value = tokio::time::timeout(self.config.timeouts.credentials, self.config.credentials.credentials()) => value
                .map_err(|_| ModelError::timeout("Bedrock credential resolution timed out").with_stage(ErrorStage::RequestEncoding))?,
            _ = abort.aborted() => Err(ModelError::abort("request was aborted while resolving Bedrock credentials").with_stage(ErrorStage::RequestEncoding)),
        }?;
        validate_credentials(&credentials)
            .map_err(|error| error.with_stage(ErrorStage::RequestEncoding))?;
        Ok(credentials)
    }

    fn stream_configuration(&self) -> crate::stream::StreamConfiguration {
        crate::stream::StreamConfiguration {
            policy: self.config.capabilities.replay.policy,
            native_context_scope: self.config.native_context_scope.clone(),
            reasoning: self
                .config
                .capabilities
                .features
                .contains(Capability::REASONING),
            signed_reasoning: self.config.signed_reasoning,
        }
    }
}

impl LanguageModel for BedrockModel {
    fn descriptor(&self) -> &LanguageModelDescriptor {
        &self.config.descriptor
    }

    fn validate_request(&self, request: &Request) -> Result<(), ModelError> {
        let options = crate::request::options(request)?;
        crate::request::validate_request(
            request,
            &options,
            &self.config.capabilities,
            self.config.reasoning_wire_format,
            self.config.signed_reasoning,
            self.config.structured_output,
        )
    }

    fn supports_request(&self, request: &Request) -> bool {
        self.validate_request(request).is_ok()
    }

    fn stream<'a>(
        &'a self,
        request: Request,
        abort: AbortSignal,
    ) -> BoxFuture<'a, Result<StreamResponse, ModelError>> {
        self.execute(request, abort, true)
    }
}

struct DirectResponseCollector {
    response: Mutex<Option<StreamResponse>>,
    descriptor: LanguageModelDescriptor,
}

impl LanguageModel for DirectResponseCollector {
    fn descriptor(&self) -> &LanguageModelDescriptor {
        &self.descriptor
    }

    fn validate_request(&self, _request: &Request) -> Result<(), ModelError> {
        Ok(())
    }

    fn supports_request(&self, _request: &Request) -> bool {
        true
    }

    fn stream<'a>(
        &'a self,
        _request: Request,
        _abort: AbortSignal,
    ) -> BoxFuture<'a, Result<StreamResponse, ModelError>> {
        let response = self
            .response
            .lock()
            .map_err(|_| ModelError::invalid_response("Bedrock direct response collector poisoned"))
            .and_then(|mut response| {
                response.take().ok_or_else(|| {
                    ModelError::invalid_response("Bedrock direct response was already collected")
                })
            });
        Box::pin(std::future::ready(response))
    }
}

async fn collect_direct(
    response: StreamResponse,
    descriptor: LanguageModelDescriptor,
) -> Result<CompleteResult, ModelError> {
    let collector = DirectResponseCollector {
        response: Mutex::new(Some(response)),
        descriptor,
    };
    collector
        .complete(Request::new(Vec::new()), AbortSignal::default())
        .await
}

fn encode_path_segment(value: &str) -> String {
    let mut output = String::new();
    for byte in value.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b'~') {
            output.push(char::from(byte));
        } else {
            use std::fmt::Write as _;
            write!(&mut output, "%{byte:02X}").expect("writing to String cannot fail");
        }
    }
    output
}

fn validate_model_id(model_id: &ModelId) -> Result<(), ModelError> {
    let value = model_id.as_str();
    if value.len() > 2048 || value.chars().any(char::is_whitespace) {
        return Err(ModelError::invalid_request("Bedrock model ID is invalid"));
    }
    Ok(())
}

fn validate_credentials(credentials: &AwsCredentials) -> Result<(), ModelError> {
    if credentials.access_key_id.is_empty()
        || credentials.secret_access_key.is_empty()
        || credentials
            .session_token
            .as_deref()
            .is_some_and(str::is_empty)
    {
        return Err(ModelError::invalid_request(
            "Bedrock static credentials must not be empty",
        ));
    }
    Ok(())
}

fn validate_adapter_declaration(capabilities: &ModelCapabilities) -> Result<(), ModelError> {
    if capabilities.compaction != CompactionCapability::Unsupported {
        return Err(ModelError::invalid_request(
            "Bedrock Converse does not support provider-native compaction",
        ));
    }
    let supported_features = Capability::TOOL_CALLING
        | Capability::PARALLEL_TOOLS
        | Capability::TOOL_INPUT_DELTAS
        | Capability::REASONING
        | Capability::STRUCTURED_OUTPUT
        | Capability::TEMPERATURE
        | Capability::TOP_P
        | Capability::MAX_OUTPUT_TOKENS
        | Capability::PROMPT_CACHING
        | Capability::USAGE
        | Capability::SOURCES;
    if capabilities.features.bits() & !supported_features.bits() != 0 {
        return Err(ModelError::invalid_request(
            "Bedrock declaration contains unsupported capabilities",
        ));
    }
    if [
        capabilities.limits.context,
        capabilities.limits.input,
        capabilities.limits.output,
    ]
    .into_iter()
    .flatten()
    .any(|limit| limit == 0)
    {
        return Err(ModelError::invalid_request(
            "Bedrock token limits must be positive when declared",
        ));
    }
    if !capabilities
        .modalities
        .input
        .iter()
        .any(|modality| modality.as_str() == "text")
        || capabilities.modalities.output.len() != 1
        || !capabilities
            .modalities
            .output
            .iter()
            .all(|modality| modality.as_str() == "text")
    {
        return Err(ModelError::invalid_request(
            "Bedrock declarations require text input and text-only output",
        ));
    }
    for modality in &capabilities.modalities.input {
        if !matches!(modality.as_str(), "text" | "image" | "video" | "pdf") {
            return Err(ModelError::invalid_request(format!(
                "Bedrock input modality `{}` is unsupported",
                modality.as_str()
            )));
        }
        if modality.as_str() != "text" && !capabilities.media.input.contains_key(modality) {
            return Err(ModelError::invalid_request(format!(
                "Bedrock input modality `{}` requires explicit media rules",
                modality.as_str()
            )));
        }
    }
    for (modality, support) in &capabilities.media.input {
        let (media_types, sources) = media_ceiling(modality.as_str()).ok_or_else(|| {
            ModelError::invalid_request(format!(
                "Bedrock media modality `{}` is unsupported",
                modality.as_str()
            ))
        })?;
        if support
            .media_types
            .iter()
            .any(|media_type| !media_types.contains(&media_type.as_str()))
        {
            return Err(ModelError::invalid_request(format!(
                "Bedrock media declaration for `{}` contains an unsupported MIME type",
                modality.as_str()
            )));
        }
        if support.sources.bits() & !sources.bits() != 0 {
            return Err(ModelError::invalid_request(format!(
                "Bedrock media declaration for `{}` contains an unsupported source form",
                modality.as_str()
            )));
        }
    }
    Ok(())
}

fn media_ceiling(modality: &str) -> Option<(&'static [&'static str], MediaSourceSupport)> {
    const IMAGE: &[&str] = &["image/png", "image/jpeg", "image/gif", "image/webp"];
    const PDF: &[&str] = &["application/pdf"];
    const VIDEO: &[&str] = &[
        "video/x-matroska",
        "video/quicktime",
        "video/mp4",
        "video/webm",
        "video/x-flv",
        "video/mpeg",
        "video/mpg",
        "video/wmv",
        "video/3gpp",
    ];
    match modality {
        "image" => Some((IMAGE, MediaSourceSupport::INLINE_BYTES)),
        "pdf" => Some((PDF, MediaSourceSupport::INLINE_BYTES)),
        "video" => Some((VIDEO, MediaSourceSupport::INLINE_BYTES)),
        _ => None,
    }
}

fn validate_settings(
    capabilities: &ModelCapabilities,
    settings: &BedrockConverseSettings,
) -> Result<(), ModelError> {
    let region = settings.region.as_str();
    if region.is_empty()
        || !region
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
    {
        return Err(ModelError::invalid_request("Bedrock region is invalid"));
    }
    if settings.event_stream.max_message_bytes < 16 {
        return Err(ModelError::invalid_request(
            "Bedrock EventStream frame limit must be at least 16 bytes",
        ));
    }
    if capabilities.cancellation != oven_sdk::CancellationCapability::LocalOnly {
        return Err(ModelError::invalid_request(
            "Bedrock Converse implements local-only cancellation",
        ));
    }
    let reasoning = capabilities.features.contains(Capability::REASONING);
    if reasoning != (settings.reasoning_wire_format != BedrockReasoningWireFormat::Unsupported) {
        return Err(ModelError::invalid_request(
            "Bedrock reasoning capability and wire format must agree",
        ));
    }
    if settings.signed_reasoning
        && (!reasoning
            || settings.reasoning_wire_format != BedrockReasoningWireFormat::AnthropicThinking
            || !capabilities.replay.reasoning
            || capabilities.replay.capability != ReplayCapability::Required)
    {
        return Err(ModelError::invalid_request(
            "Bedrock signed reasoning requires reasoning and required reasoning replay",
        ));
    }
    if reasoning
        && capabilities.replay.policy != oven_sdk::ReplayPolicy::Never
        && !capabilities.replay.reasoning
    {
        return Err(ModelError::invalid_request(
            "Bedrock reasoning replay must be declared when native replay is enabled",
        ));
    }
    let structured = capabilities
        .features
        .contains(Capability::STRUCTURED_OUTPUT);
    if structured != (settings.structured_output == BedrockStructuredOutput::JsonSchema) {
        return Err(ModelError::invalid_request(
            "Bedrock structured-output capability and wire setting must agree",
        ));
    }
    Ok(())
}

fn native_context_scope(
    provider_id: &oven_sdk::ProviderId,
    model_id: &ModelId,
    endpoint: &str,
    settings: &BedrockConverseSettings,
) -> Result<NativeContextScope, ModelError> {
    let mut hasher = Sha256::new();
    hasher.update(b"bedrock-converse-native-context-v2\0");
    hasher.update(endpoint.as_bytes());
    hasher.update(b"\0");
    hasher.update(settings.region.as_bytes());
    hasher.update(b"\0");
    hasher.update([match settings.reasoning_wire_format {
        BedrockReasoningWireFormat::Unsupported => 0,
        BedrockReasoningWireFormat::AnthropicThinking => 1,
        BedrockReasoningWireFormat::OpenAiReasoningEffort => 2,
        BedrockReasoningWireFormat::BedrockReasoningConfig => 3,
    }]);
    hasher.update([u8::from(settings.signed_reasoning)]);
    hasher.update([match settings.structured_output {
        BedrockStructuredOutput::Unsupported => 0,
        BedrockStructuredOutput::JsonSchema => 1,
    }]);
    let resource = ResourceId::new(format!("bedrock-converse:{:x}", hasher.finalize()))?;
    NativeContextScope::new(provider_id.clone(), model_id.clone(), resource)
}

pub(crate) fn validate_headers(headers: &reqwest::header::HeaderMap) -> Result<(), ModelError> {
    const PROTECTED: &[&str] = &[
        "host",
        "content-type",
        "x-amz-date",
        "x-amz-security-token",
        "x-amz-content-sha256",
    ];
    if let Some(name) = PROTECTED.iter().find(|name| headers.contains_key(**name)) {
        return Err(ModelError::invalid_request(format!(
            "Bedrock caller headers cannot set protected header {name}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BedrockEventStreamLimits, BedrockStructuredOutput};
    use oven_sdk::{
        ApiEndpoint, CancellationCapability, HeaderConfig, Modalities, Modality, ModelDeclaration,
        ModelLimits, ProviderConfig, ProviderId, ReplayDeclaration, ReplayPolicy,
    };

    fn model(id: &str) -> BedrockModel {
        let provider = ProviderConfig::new(
            ProviderId::new(BEDROCK_PROVIDER_ID),
            ApiEndpoint::parse("https://bedrock-runtime.us-east-1.amazonaws.com").unwrap(),
            BedrockAuth::Static(AwsCredentials {
                access_key_id: "key".into(),
                secret_access_key: "secret".into(),
                session_token: None,
            }),
            HeaderConfig::empty(),
        )
        .unwrap();
        let capabilities = ModelCapabilities {
            features: Capability::USAGE,
            limits: ModelLimits::new(None, None, None),
            modalities: Modalities::new([Modality::text()], [Modality::text()]),
            media: Default::default(),
            cancellation: CancellationCapability::LocalOnly,
            compaction: CompactionCapability::Unsupported,
            replay: ReplayDeclaration {
                policy: ReplayPolicy::IfValid,
                capability: ReplayCapability::Optional,
                reasoning: false,
            },
        };
        let declaration =
            ModelDeclaration::new(ModelId::new(id), capabilities).expect("declaration");
        let settings = BedrockConverseSettings::new(
            "us-east-1",
            BedrockReasoningWireFormat::Unsupported,
            false,
            BedrockStructuredOutput::Unsupported,
            BedrockEventStreamLimits::new(1024),
        );
        BedrockModel::new(ModelConfig::new(provider, declaration, settings)).unwrap()
    }

    #[test]
    fn model_ids_and_arns_are_exact_single_path_segments() {
        assert_eq!(
            model("anthropic.claude-haiku-4-5-20251001-v1:0")
                .converse_stream_url()
                .unwrap()
                .path(),
            "/model/anthropic.claude-haiku-4-5-20251001-v1%3A0/converse-stream"
        );
        assert_eq!(
            model("arn:aws:bedrock:us-east-1:123:inference-profile/x")
                .converse_url()
                .unwrap()
                .path(),
            "/model/arn%3Aaws%3Abedrock%3Aus-east-1%3A123%3Ainference-profile%2Fx/converse"
        );
    }

    #[test]
    fn credentials_and_regions_are_explicit() {
        assert!(
            validate_settings(
                &model("opaque").config.capabilities,
                &BedrockConverseSettings::new(
                    "",
                    BedrockReasoningWireFormat::Unsupported,
                    false,
                    BedrockStructuredOutput::Unsupported,
                    BedrockEventStreamLimits::new(1024),
                )
            )
            .is_err()
        );
        assert!(
            validate_credentials(&AwsCredentials {
                access_key_id: String::new(),
                secret_access_key: "secret".into(),
                session_token: None,
            })
            .is_err()
        );
    }
}
