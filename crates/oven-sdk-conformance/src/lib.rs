#![warn(missing_docs)]
//! Runtime-neutral contract tests for [`oven_sdk::LanguageModel`] adapters.
//!
//! Use [`assert_stream_contract`] for parser-level stream validation,
//! [`assert_replay_round_trip`] for compatible replay checks,
//! [`assert_compaction_round_trip`] for provider-native context checks, and
//! [`sse`] to construct byte-level SSE parser fixtures.
//!
//! # Example
//!
//! ```no_run
//! use oven_sdk::Request;
//! use oven_sdk_conformance::{assert_complete_drain, assert_stream_lifecycle};
//!
//! async fn check(model: &dyn oven_sdk::LanguageModel) -> Result<(), Box<dyn std::error::Error>> {
//! let request = Request::new(Vec::new());
//! let streamed = assert_stream_lifecycle(model, request.clone()).await?;
//! let completed = assert_complete_drain(model, request).await?;
//! let _replay_decisions = &streamed.request.replay.decisions;
//! let _turn = &completed.turn;
//! Ok(())
//! }
//! ```

use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    error::Error,
    fmt,
    panic::{AssertUnwindSafe, UnwindSafe},
    pin::Pin,
    task::{Context, Poll},
};

use futures_core::Stream;
use oven_sdk::{
    AbortSignal, AdapterId, AssistantPart, BoxFuture, BoxStream, CancellationCapability,
    Capability, CompactionCapability, CompactionRequest, CompactionResult, CompleteResult,
    CompletedTurn, FilePart, FileSource, Finish, FinishReason, HistoryTurn, InferenceOptions,
    InputPart, LanguageModel, LanguageModelDescriptor, MediaSourceSupport, Modality,
    ModelCapabilities, ModelError, ModelErrorKind, ModelId, ModelIdentity, NativeContextScope,
    NativeContextWindow, NativeReplayArtifact, ProviderId, ReplayCapability, ReplayDecision,
    ReplayDeclaration, ReplayDisposition, ReplayOutcome, ReplayPolicy, Request, RequestMetadata,
    ResourceId, ResponseFormat, ResponseHead, StreamItem, StreamPart, StreamResponse, TextPart,
    ToolDefinition, UserMessage,
};

/// A structured conformance failure with a diagnostic suitable for test output.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ConformanceError {
    kind: ConformanceErrorKind,
    message: String,
}

/// The broad category of a conformance failure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ConformanceErrorKind {
    /// A model stream or result violated a contract invariant.
    ContractViolation,
    /// A stream ended without its mandatory terminal finish part.
    UnexpectedEof,
    /// The model returned a [`ModelError`] while running a conformance probe.
    ModelError,
}

impl ConformanceError {
    /// Creates a conformance failure from a diagnostic message.
    #[must_use]
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            kind: ConformanceErrorKind::ContractViolation,
            message: message.into(),
        }
    }

    /// Creates an EOF conformance failure for a stream missing `Finish`.
    #[must_use]
    pub fn unexpected_eof(message: impl Into<String>) -> Self {
        Self {
            kind: ConformanceErrorKind::UnexpectedEof,
            message: message.into(),
        }
    }

    /// Creates a failure wrapping a model-produced error.
    #[must_use]
    pub fn model_error(message: impl Into<String>) -> Self {
        Self {
            kind: ConformanceErrorKind::ModelError,
            message: message.into(),
        }
    }

    /// Returns the category of this failure.
    #[must_use]
    pub const fn kind(&self) -> ConformanceErrorKind {
        self.kind
    }

    /// Returns the diagnostic message.
    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for ConformanceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl Error for ConformanceError {}

/// The successful stream facts captured by [`assert_stream_lifecycle`].
#[derive(Clone, Debug, PartialEq)]
pub struct StreamReport {
    /// All successful stream parts, including the terminal finish part.
    pub parts: Vec<StreamPart>,
    /// The mandatory terminal finish data.
    pub finish: Finish,
    /// Warnings emitted by the first stream part.
    pub warnings: Vec<String>,
}

/// Successful lifecycle facts from a [`LanguageModel::stream`] call.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelStreamReport {
    /// Validated normalized stream facts.
    pub stream: StreamReport,
    /// Request metadata returned with the stream.
    pub request: RequestMetadata,
    /// Initial response metadata returned with the stream.
    pub response: ResponseHead,
}

/// A reusable script of normalized stream items.
#[derive(Clone, Debug, Default)]
pub struct StreamFixture {
    items: Vec<StreamItem>,
}

impl StreamFixture {
    /// Creates a fixture from successful stream parts.
    #[must_use]
    pub fn from_parts(parts: Vec<StreamPart>) -> Self {
        Self {
            items: parts.into_iter().map(Ok).collect(),
        }
    }

    /// Creates a fixture from complete stream items, including terminal errors.
    #[must_use]
    pub fn from_items(items: Vec<StreamItem>) -> Self {
        Self { items }
    }

    /// Returns a valid single-text-block stream.
    #[must_use]
    pub fn valid_text(text: impl Into<String>) -> Self {
        let text = text.into();
        Self::from_parts(vec![
            StreamPart::StreamStart {
                warnings: Vec::new(),
            },
            StreamPart::TextStart {
                id: "text-0".into(),
                metadata: None,
            },
            StreamPart::TextDelta {
                id: "text-0".into(),
                delta: text,
                metadata: None,
            },
            StreamPart::TextEnd {
                id: "text-0".into(),
                metadata: None,
            },
            StreamPart::Finish {
                finish: Finish::new(Default::default(), FinishReason::Stop),
            },
        ])
    }

    /// Returns a stream that ends before its mandatory finish part.
    #[must_use]
    pub fn eof_before_finish() -> Self {
        Self::from_parts(vec![StreamPart::StreamStart {
            warnings: Vec::new(),
        }])
    }

    /// Returns a valid in-band provider-error terminal sequence.
    #[must_use]
    pub fn in_band_error(error: ModelError) -> Self {
        Self::from_parts(vec![
            StreamPart::StreamStart {
                warnings: Vec::new(),
            },
            StreamPart::Error { error },
            StreamPart::Finish {
                finish: Finish::new(Default::default(), FinishReason::Error),
            },
        ])
    }

    /// Returns the scripted items for use by a model implementation.
    pub fn items(&self) -> &[StreamItem] {
        &self.items
    }

    /// Appends a successful stream part.
    #[must_use]
    pub fn push_part(mut self, part: StreamPart) -> Self {
        self.items.push(Ok(part));
        self
    }
}

/// A configurable in-memory [`LanguageModel`] suitable for conformance fixtures.
#[derive(Clone, Debug)]
pub struct MockLanguageModel {
    descriptor: LanguageModelDescriptor,
    native_context_scope: NativeContextScope,
    script: StreamFixture,
    request_metadata: RequestMetadata,
    response_head: ResponseHead,
    replay_outcome: Option<ReplayOutcome>,
    compaction_results:
        std::sync::Arc<std::sync::Mutex<VecDeque<Result<CompactionResult, ModelError>>>>,
    captured_compactions: std::sync::Arc<std::sync::Mutex<Vec<CompactionRequest>>>,
}

impl MockLanguageModel {
    /// Creates a builder for a mock model with a valid single-text-block script.
    #[must_use]
    pub fn builder() -> MockLanguageModelBuilder {
        MockLanguageModelBuilder::default()
    }

    /// Returns the script used for every mock stream invocation.
    #[must_use]
    pub fn script(&self) -> &StreamFixture {
        &self.script
    }

    /// Returns all compaction requests captured by this mock.
    #[must_use]
    pub fn captured_compactions(&self) -> Vec<CompactionRequest> {
        self.captured_compactions
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .clone()
    }
}

/// Builder for [`MockLanguageModel`].
#[derive(Clone, Debug)]
pub struct MockLanguageModelBuilder {
    descriptor: LanguageModelDescriptor,
    native_context_scope: NativeContextScope,
    script: StreamFixture,
    native_replay: Option<NativeReplayArtifact>,
    request_metadata: RequestMetadata,
    response_head: ResponseHead,
    replay_outcome: Option<ReplayOutcome>,
    compaction_results: VecDeque<Result<CompactionResult, ModelError>>,
}

impl Default for MockLanguageModelBuilder {
    fn default() -> Self {
        let identity = ModelIdentity::new(
            ProviderId::new("conformance.mock"),
            ModelId::new("scripted"),
        )
        .expect("valid mock identity");
        let native_context_scope = NativeContextScope::new(
            identity.provider_id.clone(),
            identity.model_id.clone(),
            ResourceId::new("conformance.mock.resource").expect("valid mock resource"),
        )
        .expect("valid mock scope");
        Self {
            descriptor: LanguageModelDescriptor::new(
                identity,
                AdapterId::new("conformance.mock"),
                ModelCapabilities::conservative(),
            )
            .expect("valid mock descriptor"),
            native_context_scope,
            script: StreamFixture::valid_text("mock"),
            native_replay: None,
            request_metadata: RequestMetadata::default(),
            response_head: ResponseHead::default(),
            replay_outcome: None,
            compaction_results: VecDeque::new(),
        }
    }
}

impl MockLanguageModelBuilder {
    /// Replaces the configured descriptor.
    #[must_use]
    pub fn descriptor(mut self, descriptor: LanguageModelDescriptor) -> Self {
        self.descriptor = descriptor;
        self
    }

    /// Replaces the declared capabilities.
    #[must_use]
    pub fn capabilities(mut self, capabilities: ModelCapabilities) -> Self {
        self.descriptor.capabilities = capabilities;
        self
    }

    /// Replaces the native-context scope expected by replay and compaction.
    #[must_use]
    pub fn native_context_scope(mut self, native_context_scope: NativeContextScope) -> Self {
        self.native_context_scope = native_context_scope;
        self
    }

    /// Appends one result to the mock compaction queue.
    #[must_use]
    pub fn compaction_result(mut self, result: Result<CompactionResult, ModelError>) -> Self {
        self.compaction_results.push_back(result);
        self
    }

    /// Replaces the script replayed by each stream invocation.
    #[must_use]
    pub fn script(mut self, script: StreamFixture) -> Self {
        self.script = script;
        self
    }

    /// Attaches this artifact to the script's terminal finish part when built.
    #[must_use]
    pub fn native_replay(mut self, artifact: NativeReplayArtifact) -> Self {
        self.native_replay = Some(artifact);
        self
    }

    /// Replaces safe request metadata returned by mock streams.
    #[must_use]
    pub fn request_metadata(mut self, request_metadata: RequestMetadata) -> Self {
        self.request_metadata = request_metadata;
        self
    }

    /// Replaces initial response metadata returned by mock streams.
    #[must_use]
    pub fn response_head(mut self, response_head: ResponseHead) -> Self {
        self.response_head = response_head;
        self
    }

    /// Replaces derived replay decisions returned by mock streams.
    #[must_use]
    pub fn replay_outcome(mut self, replay_outcome: ReplayOutcome) -> Self {
        self.replay_outcome = Some(replay_outcome);
        self
    }

    /// Builds the configured in-memory model.
    #[must_use]
    pub fn build(mut self) -> MockLanguageModel {
        if let Some(artifact) = self.native_replay {
            for part in self.script.items.iter_mut().flatten() {
                if let StreamPart::Finish { finish } = part {
                    finish.native_replay = Some(artifact.clone());
                    break;
                }
            }
        }
        MockLanguageModel {
            descriptor: self.descriptor,
            native_context_scope: self.native_context_scope,
            script: self.script,
            request_metadata: self.request_metadata,
            response_head: self.response_head,
            replay_outcome: self.replay_outcome,
            compaction_results: std::sync::Arc::new(std::sync::Mutex::new(self.compaction_results)),
            captured_compactions: std::sync::Arc::new(std::sync::Mutex::new(Vec::new())),
        }
    }
}

struct FixtureStream(VecDeque<StreamItem>);

impl Stream for FixtureStream {
    type Item = StreamItem;

    fn poll_next(mut self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Poll::Ready(self.0.pop_front())
    }
}

fn replay_outcome_for(
    expected_adapter: &AdapterId,
    expected_scope: &NativeContextScope,
    replay: &ReplayDeclaration,
    history: &[HistoryTurn],
) -> ReplayOutcome {
    let mut decisions = Vec::new();
    for (history_index, turn) in history.iter().enumerate() {
        let HistoryTurn::Assistant(turn) = turn else {
            continue;
        };
        if replay.policy == ReplayPolicy::Never {
            decisions.push(ReplayDecision {
                history_index,
                disposition: ReplayDisposition::ReconstructedNormalized,
            });
            continue;
        }
        let Some(artifact) = &turn.finish.native_replay else {
            decisions.push(ReplayDecision {
                history_index,
                disposition: ReplayDisposition::NoArtifact,
            });
            decisions.push(ReplayDecision {
                history_index,
                disposition: ReplayDisposition::ReconstructedNormalized,
            });
            continue;
        };
        if artifact.adapter_id() != expected_adapter {
            decisions.push(ReplayDecision {
                history_index,
                disposition: ReplayDisposition::DiscardedForeignAdapter {
                    found: artifact.adapter_id().clone(),
                    expected: expected_adapter.clone(),
                },
            });
            decisions.push(ReplayDecision {
                history_index,
                disposition: ReplayDisposition::ReconstructedNormalized,
            });
        } else if artifact.scope() != expected_scope {
            decisions.push(ReplayDecision {
                history_index,
                disposition: ReplayDisposition::DiscardedForeignScope {
                    found: artifact.scope().clone(),
                    expected: expected_scope.clone(),
                },
            });
            decisions.push(ReplayDecision {
                history_index,
                disposition: ReplayDisposition::ReconstructedNormalized,
            });
        } else if artifact.payload() == &serde_json::Value::String("garbage".into()) {
            decisions.push(ReplayDecision {
                history_index,
                disposition: ReplayDisposition::DiscardedInvalidPayload {
                    reason: "mock replay decoder rejected payload".into(),
                },
            });
            decisions.push(ReplayDecision {
                history_index,
                disposition: ReplayDisposition::ReconstructedNormalized,
            });
        } else {
            decisions.push(ReplayDecision {
                history_index,
                disposition: ReplayDisposition::Replayed,
            });
        }
    }
    ReplayOutcome { decisions }
}

impl LanguageModel for MockLanguageModel {
    fn descriptor(&self) -> &LanguageModelDescriptor {
        &self.descriptor
    }

    fn validate_request(&self, request: &Request) -> Result<(), ModelError> {
        request.validate_for(&self.descriptor.capabilities)?;
        if let Some(context) = &request.native_context {
            if context.adapter_id() != &self.descriptor.adapter_id {
                return Err(ModelError::native_context(
                    "native context belongs to a foreign adapter",
                ));
            }
            if context.scope() != &self.native_context_scope {
                return Err(ModelError::native_context(
                    "native context belongs to a foreign scope",
                ));
            }
        }
        Ok(())
    }

    fn supports_request(&self, request: &Request) -> bool {
        self.validate_request(request).is_ok()
    }

    fn validate_compaction(&self, request: &CompactionRequest) -> Result<(), ModelError> {
        request.validate_for(&self.descriptor.capabilities)?;
        self.validate_request(&request.request)
    }

    fn stream<'a>(
        &'a self,
        request: Request,
        _: AbortSignal,
    ) -> BoxFuture<'a, Result<StreamResponse, ModelError>> {
        let validation = self.validate_request(&request);
        let items = self.script.items.clone();
        let mut request_metadata = self.request_metadata.clone();
        request_metadata.replay = self.replay_outcome.clone().unwrap_or_else(|| {
            replay_outcome_for(
                &self.descriptor.adapter_id,
                &self.native_context_scope,
                &self.descriptor.capabilities.replay,
                &request.history,
            )
        });
        let response_head = self.response_head.clone();
        Box::pin(async move {
            validation?;
            let stream: BoxStream<'static, StreamItem> =
                Box::pin(FixtureStream(VecDeque::from(items)));
            Ok(StreamResponse::new(stream)
                .with_request(request_metadata)
                .with_response(response_head))
        })
    }

    fn compact<'a>(
        &'a self,
        request: CompactionRequest,
        abort: AbortSignal,
    ) -> BoxFuture<'a, Result<CompactionResult, ModelError>> {
        let validation = self.validate_compaction(&request);
        let captured = std::sync::Arc::clone(&self.captured_compactions);
        let results = std::sync::Arc::clone(&self.compaction_results);
        Box::pin(async move {
            validation?;
            if abort.is_aborted() {
                return Err(ModelError::abort("mock compaction aborted"));
            }
            captured
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .push(request);
            let result = results
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .pop_front();
            match result {
                Some(result) => result,
                None => Err(ModelError::native_context("mock compaction queue is empty")),
            }
        })
    }
}

/// Drains a model stream and verifies the public stream lifecycle contract.
pub async fn assert_stream_lifecycle(
    model: &dyn LanguageModel,
    request: Request,
) -> Result<ModelStreamReport, ConformanceError> {
    let StreamResponse {
        stream,
        request,
        response,
    } = model
        .stream(request, AbortSignal::default())
        .await
        .map_err(model_error)?;
    Ok(ModelStreamReport {
        stream: assert_stream_contract(stream).await?,
        request,
        response,
    })
}

/// Drains any normalized stream and verifies the public lifecycle contract.
///
/// This is the parser-level variant of [`assert_stream_lifecycle`]: adapter
/// tests can validate the output of an SSE/transport parser directly, without
/// wrapping it in a [`LanguageModel`] implementation. It accepts any normalized
/// stream and pins it internally.
pub async fn assert_stream_contract<S>(stream: S) -> Result<StreamReport, ConformanceError>
where
    S: Stream<Item = StreamItem>,
{
    let mut stream = Box::pin(stream);
    let mut parts = Vec::new();
    while let Some(item) = std::future::poll_fn(|context| stream.as_mut().poll_next(context)).await
    {
        parts.push(item.map_err(model_error)?);
    }
    validate_lifecycle(parts)
}

/// Verifies that [`LanguageModel::complete`] agrees with the model's stream.
pub async fn assert_complete_drain(
    model: &dyn LanguageModel,
    request: Request,
) -> Result<CompleteResult, ConformanceError> {
    let report = assert_stream_lifecycle(model, request.clone()).await?;
    let completed = model
        .complete(request, AbortSignal::default())
        .await
        .map_err(model_error)?;
    if completed.turn.finish != report.stream.finish {
        return Err(ConformanceError::new(
            "complete() finish differs from stream Finish",
        ));
    }
    if report.stream.warnings != completed.turn.warnings {
        return Err(ConformanceError::new(format!(
            "complete() warnings differ: stream {:?}, complete {:?}",
            report.stream.warnings, completed.turn.warnings
        )));
    }
    let expected = assembled_content(&report.stream.parts)?;
    if completed.turn.message.content != expected {
        return Err(ConformanceError::new(format!(
            "complete() ordered content mismatch: expected {expected:?}, got {:?}",
            completed.turn.message.content
        )));
    }
    if completed.response != report.response {
        return Err(ConformanceError::new(format!(
            "complete() response head differs: stream {:?}, complete {:?}",
            report.response, completed.response
        )));
    }
    if completed.request != report.request {
        return Err(ConformanceError::new(
            "complete() request metadata differs from stream request metadata",
        ));
    }
    Ok(completed)
}

/// An adapter-supplied request that exercises one capability.
#[derive(Clone, Debug)]
pub struct CapabilityProbe {
    /// Capability exercised by this request.
    pub capability: Capability,
    /// Stable diagnostic capability name.
    pub name: &'static str,
    /// Request exercising the capability.
    pub request: Request,
}

/// Verifies request-level honesty for normalized capability probes.
pub fn assert_capability_honesty(model: &dyn LanguageModel) -> Result<(), ConformanceError> {
    assert_capability_honesty_with(model, std::iter::empty())
}

/// Verifies request-level honesty including adapter-supplied capability probes.
///
/// An adapter-supplied probe replaces the generic probe for the same capability,
/// allowing protocol-specific valid minima such as output-token limits.
pub fn assert_capability_honesty_with(
    model: &dyn LanguageModel,
    additional: impl IntoIterator<Item = CapabilityProbe>,
) -> Result<(), ConformanceError> {
    let capabilities = model.capabilities().features;
    let additional = additional.into_iter().collect::<Vec<_>>();
    let overridden = additional
        .iter()
        .map(|probe| probe.capability.bits())
        .collect::<BTreeSet<_>>();
    let probes = capability_probe_requests()
        .into_iter()
        .filter(|(capability, _, _)| !overridden.contains(&capability.bits()))
        .map(|(capability, request, name)| CapabilityProbe {
            capability,
            name,
            request,
        })
        .chain(additional);
    for CapabilityProbe {
        capability,
        request,
        name,
    } in probes
    {
        let result = model.validate_request(&request);
        if capabilities.contains(capability) {
            if result
                .as_ref()
                .is_err_and(|error| error.is_kind(ModelErrorKind::Unsupported))
            {
                return Err(ConformanceError::new(format!(
                    "model claims {name} but rejects its normalized probe as Unsupported"
                )));
            }
            if let Err(error) = result {
                return Err(ConformanceError::new(format!(
                    "model claims {name} but rejects its normalized probe: {error}"
                )));
            }
            if !model.supports_request(&request) {
                return Err(ConformanceError::new(format!(
                    "model claims {name} but supports_request returned false"
                )));
            }
        } else {
            if !result
                .as_ref()
                .is_err_and(|error| error.is_kind(ModelErrorKind::Unsupported))
            {
                return Err(ConformanceError::new(format!(
                    "model does not claim {name} but its probe was not rejected as Unsupported"
                )));
            }
            if model.supports_request(&request) {
                return Err(ConformanceError::new(format!(
                    "model does not claim {name} but supports_request returned true"
                )));
            }
        }
    }
    Ok(())
}

/// Verifies that a descriptor is complete, internally valid, and returned consistently.
pub fn assert_declaration_honesty(model: &dyn LanguageModel) -> Result<(), ConformanceError> {
    let descriptor = model.descriptor();
    descriptor.identity.validate().map_err(model_error)?;
    descriptor.adapter_id.validate().map_err(model_error)?;
    descriptor.capabilities.validate().map_err(model_error)?;
    if model.capabilities() != &descriptor.capabilities {
        return Err(ConformanceError::new(
            "LanguageModel::capabilities differs from descriptor capabilities",
        ));
    }
    let compaction = CompactionRequest::new(Request::new(Vec::new()));
    let validation = model.validate_compaction(&compaction);
    match descriptor.capabilities.compaction {
        CompactionCapability::Native => {
            validation.map_err(|error| {
                ConformanceError::new(format!(
                    "model declares native compaction but rejects its baseline request: {error}"
                ))
            })?;
            if !model.supports_compaction(&compaction) {
                return Err(ConformanceError::new(
                    "model declares native compaction but supports_compaction returned false",
                ));
            }
        }
        CompactionCapability::Unsupported => {
            if !validation
                .as_ref()
                .is_err_and(|error| error.is_kind(ModelErrorKind::Unsupported))
            {
                return Err(ConformanceError::new(
                    "model declares compaction unsupported but baseline validation was not Unsupported",
                ));
            }
            if model.supports_compaction(&compaction) {
                return Err(ConformanceError::new(
                    "model declares compaction unsupported but supports_compaction returned true",
                ));
            }
        }
    }
    Ok(())
}

/// Verifies one provider-native context window against the configured model contract.
pub fn assert_native_context_window(
    descriptor: &LanguageModelDescriptor,
    expected_scope: &NativeContextScope,
    window: &NativeContextWindow,
) -> Result<(), ConformanceError> {
    if descriptor.capabilities.compaction != CompactionCapability::Native {
        return Err(ConformanceError::new(
            "native context window was produced while compaction is declared unsupported",
        ));
    }
    if window.adapter_id() != &descriptor.adapter_id {
        return Err(ConformanceError::new(
            "native context adapter ID differs from the configured adapter",
        ));
    }
    if window.scope() != expected_scope {
        return Err(ConformanceError::new(
            "native context scope differs from the exact configured scope",
        ));
    }
    if expected_scope.provider_id != descriptor.identity.provider_id
        || expected_scope.model_id != descriptor.identity.model_id
    {
        return Err(ConformanceError::new(
            "expected native context scope differs from descriptor identity",
        ));
    }
    let payload_size = serde_json::to_vec(window.payload())
        .map_err(|error| ConformanceError::new(format!("could not size native context: {error}")))?
        .len();
    if payload_size > NativeContextWindow::MAX_PAYLOAD_BYTES {
        return Err(ConformanceError::new(
            "native context exceeds the 32 MiB cap",
        ));
    }
    if format!("{window:?}").contains(&window.payload().to_string()) {
        return Err(ConformanceError::new(
            "native context Debug output exposed its payload",
        ));
    }
    let encoded = serde_json::to_vec(window).map_err(|error| {
        ConformanceError::new(format!("could not encode native context: {error}"))
    })?;
    let decoded: NativeContextWindow = serde_json::from_slice(&encoded).map_err(|error| {
        ConformanceError::new(format!("could not decode native context: {error}"))
    })?;
    if &decoded != window {
        return Err(ConformanceError::new(
            "native context did not round-trip through its current serde shape",
        ));
    }
    Ok(())
}

/// Runs provider-native compaction and validates its returned context window.
pub async fn assert_native_compaction(
    model: &dyn LanguageModel,
    expected_scope: &NativeContextScope,
    request: CompactionRequest,
) -> Result<CompactionResult, ConformanceError> {
    model.validate_compaction(&request).map_err(model_error)?;
    if !model.supports_compaction(&request) {
        return Err(ConformanceError::new(
            "validate_compaction accepted but supports_compaction returned false",
        ));
    }
    let result = model
        .compact(request, AbortSignal::default())
        .await
        .map_err(model_error)?;
    assert_native_context_window(model.descriptor(), expected_scope, &result.native_context)?;
    Ok(result)
}

/// Verifies that an already-aborted native compaction does not succeed.
pub async fn assert_compaction_cancellation(
    model: &dyn LanguageModel,
    request: CompactionRequest,
) -> Result<(), ConformanceError> {
    if model.descriptor().capabilities.cancellation == CancellationCapability::Unsupported {
        return Err(ConformanceError::new(
            "compaction cancellation assertion requires declared cancellation support",
        ));
    }
    let (signal, registration) = AbortSignal::new();
    registration.abort();
    match model.compact(request, signal).await {
        Err(error) if error.is_kind(ModelErrorKind::Abort) => Ok(()),
        Err(error) => Err(ConformanceError::new(format!(
            "cancelled compaction returned {:?}, expected Abort",
            error.kind()
        ))),
        Ok(_) => Err(ConformanceError::new(
            "cancelled compaction unexpectedly succeeded",
        )),
    }
}

/// Verifies native compaction followed by use of its context in a model request.
pub async fn assert_compaction_round_trip(
    model: &dyn LanguageModel,
    expected_scope: &NativeContextScope,
    compaction: CompactionRequest,
    mut continuation: Request,
) -> Result<(CompactionResult, ModelStreamReport), ConformanceError> {
    let result = assert_native_compaction(model, expected_scope, compaction).await?;
    continuation.native_context = Some(result.native_context.clone());
    model.validate_request(&continuation).map_err(model_error)?;
    if !model.supports_request(&continuation) {
        return Err(ConformanceError::new(
            "native-context continuation validation succeeded but supports_request returned false",
        ));
    }
    let report = assert_stream_lifecycle(model, continuation).await?;
    Ok((result, report))
}

/// Verifies that an unsupported compaction request fails before adapter work is observable.
pub async fn assert_compaction_unsupported_before_io(
    model: &dyn LanguageModel,
    request: CompactionRequest,
) -> Result<(), ConformanceError> {
    if model.descriptor().capabilities.compaction != CompactionCapability::Unsupported {
        return Err(ConformanceError::new(
            "unsupported compaction assertion requires an Unsupported declaration",
        ));
    }
    if !model
        .validate_compaction(&request)
        .as_ref()
        .is_err_and(|error| error.is_kind(ModelErrorKind::Unsupported))
        || model.supports_compaction(&request)
    {
        return Err(ConformanceError::new(
            "unsupported compaction validation/support matrix is dishonest",
        ));
    }
    match model.compact(request, AbortSignal::default()).await {
        Err(error) if error.is_kind(ModelErrorKind::Unsupported) => Ok(()),
        Err(error) => Err(ConformanceError::new(format!(
            "unsupported compaction returned {:?}, expected Unsupported",
            error.kind()
        ))),
        Ok(_) => Err(ConformanceError::new(
            "unsupported compaction unexpectedly succeeded",
        )),
    }
}

/// One request used to prove that model IDs do not select behavior.
#[derive(Clone, Debug)]
pub struct ModelIdIndependenceProbe {
    /// Diagnostic probe name.
    pub name: String,
    /// Request exercised against both configured models.
    pub request: Request,
    /// Whether accepted requests should also compare normalized stream/complete output.
    pub compare_execution: bool,
}

impl ModelIdIndependenceProbe {
    /// Creates a probe that compares validation and normalized execution.
    #[must_use]
    pub fn new(name: impl Into<String>, request: Request) -> Self {
        Self {
            name: name.into(),
            request,
            compare_execution: true,
        }
    }

    /// Restricts this probe to validation and support checks.
    #[must_use]
    pub fn validation_only(mut self) -> Self {
        self.compare_execution = false;
        self
    }
}

/// Runs the default comprehensive model-ID independence probe suite.
pub async fn assert_model_id_independence(
    first: &dyn LanguageModel,
    second: &dyn LanguageModel,
) -> Result<(), ConformanceError> {
    assert_model_id_independence_with(first, second, std::iter::empty()).await
}

/// Runs the default suite plus caller-supplied provider-specific probes.
pub async fn assert_model_id_independence_with(
    first: &dyn LanguageModel,
    second: &dyn LanguageModel,
    additional: impl IntoIterator<Item = ModelIdIndependenceProbe>,
) -> Result<(), ConformanceError> {
    let first_descriptor = first.descriptor();
    let second_descriptor = second.descriptor();
    if first_descriptor.identity.provider_id != second_descriptor.identity.provider_id
        || first_descriptor.adapter_id != second_descriptor.adapter_id
        || first_descriptor.capabilities != second_descriptor.capabilities
        || first_descriptor.provider_metadata != second_descriptor.provider_metadata
    {
        return Err(ConformanceError::new(
            "model-ID independence requires otherwise identical descriptors",
        ));
    }
    if first_descriptor.identity.model_id == second_descriptor.identity.model_id {
        return Err(ConformanceError::new(
            "model-ID independence requires distinct model IDs",
        ));
    }
    let compaction = CompactionRequest::new(Request::new(Vec::new()));
    let first_compaction = first.validate_compaction(&compaction);
    let second_compaction = second.validate_compaction(&compaction);
    if first_compaction.as_ref().map_err(ModelError::kind)
        != second_compaction.as_ref().map_err(ModelError::kind)
        || first.supports_compaction(&compaction) != second.supports_compaction(&compaction)
    {
        return Err(ConformanceError::new(
            "model ID changed native compaction validation behavior",
        ));
    }
    let additional = additional.into_iter().collect::<Vec<_>>();
    let overridden = additional
        .iter()
        .map(|probe| probe.name.as_str())
        .collect::<BTreeSet<_>>();
    let mut probes = vec![ModelIdIndependenceProbe::new(
        "baseline",
        Request::new(Vec::new()),
    )];
    probes.extend(
        capability_probe_requests()
            .into_iter()
            .filter(|(_, _, name)| !overridden.contains(name))
            .map(|(_, request, name)| ModelIdIndependenceProbe::new(name, request)),
    );
    if first_descriptor
        .capabilities
        .features
        .contains(Capability::MAX_OUTPUT_TOKENS)
        && let Some(limit) = first_descriptor
            .capabilities
            .limits
            .output
            .filter(|limit| *limit > 0)
    {
        let mut inference = InferenceOptions::new();
        inference.max_output_tokens = Some(limit);
        probes.push(ModelIdIndependenceProbe::new(
            "MAX_OUTPUT_TOKENS_LIMIT",
            Request::new(Vec::new()).with_inference(inference),
        ));
    }
    probes.extend(
        media_probe_requests(&first_descriptor.capabilities)?
            .into_iter()
            .map(|probe| {
                let name = format!(
                    "media:{}:{}:{:?}",
                    probe.modality.as_str(),
                    probe.declared_media_type,
                    probe.source
                );
                if probe.expected_supported {
                    ModelIdIndependenceProbe::new(name, probe.request)
                } else {
                    ModelIdIndependenceProbe::new(name, probe.request).validation_only()
                }
            }),
    );
    probes.push(ModelIdIndependenceProbe::new(
        "replay:no-artifact",
        Request::new(vec![HistoryTurn::assistant(CompletedTurn::new(
            oven_sdk::AssistantMessage::new(Vec::new()),
            Finish::new(Default::default(), FinishReason::Stop),
        ))]),
    ));
    probes.extend(additional);

    for ModelIdIndependenceProbe {
        name,
        request,
        compare_execution,
    } in probes
    {
        let first_result = first.validate_request(&request);
        let second_result = second.validate_request(&request);
        if first_result.as_ref().map_err(ModelError::kind)
            != second_result.as_ref().map_err(ModelError::kind)
            || first.supports_request(&request) != second.supports_request(&request)
        {
            return Err(ConformanceError::new(format!(
                "model ID changed validation behavior for {name}"
            )));
        }
        if compare_execution && first_result.is_ok() {
            let first_complete = assert_complete_drain(first, request.clone()).await?;
            let second_complete = assert_complete_drain(second, request).await?;
            compare_model_independent_results(&name, first_complete, second_complete)?;
        }
    }
    Ok(())
}

fn compare_model_independent_results(
    name: &str,
    mut first: CompleteResult,
    mut second: CompleteResult,
) -> Result<(), ConformanceError> {
    first.turn.finish.native_replay = None;
    second.turn.finish.native_replay = None;
    if first.turn != second.turn {
        return Err(ConformanceError::new(format!(
            "model ID changed normalized completion for {name}"
        )));
    }
    if first.request.provider_metadata != second.request.provider_metadata
        || replay_shapes(&first.request.replay) != replay_shapes(&second.request.replay)
    {
        return Err(ConformanceError::new(format!(
            "model ID changed normalized request encoding metadata for {name}"
        )));
    }
    if first.response.http_status != second.response.http_status
        || first.response.response_metadata != second.response.response_metadata
    {
        return Err(ConformanceError::new(format!(
            "model ID changed normalized response metadata for {name}"
        )));
    }
    Ok(())
}

fn replay_shapes(outcome: &ReplayOutcome) -> Vec<(usize, &'static str)> {
    outcome
        .decisions
        .iter()
        .map(|decision| {
            let kind = match decision.disposition {
                ReplayDisposition::Replayed => "replayed",
                ReplayDisposition::NoArtifact => "no_artifact",
                ReplayDisposition::DiscardedForeignAdapter { .. } => "foreign_adapter",
                ReplayDisposition::DiscardedForeignScope { .. } => "foreign_scope",
                ReplayDisposition::DiscardedInvalidPayload { .. } => "invalid_payload",
                ReplayDisposition::ReconstructedNormalized => "reconstructed",
            };
            (decision.history_index, kind)
        })
        .collect()
}

/// Verifies that a model's validation result agrees with core request validation.
pub fn assert_validate_for_consistency(
    model: &dyn LanguageModel,
    request: &Request,
) -> Result<(), ConformanceError> {
    let core = request.validate_for(model.capabilities());
    let adapter = model.validate_request(request);
    match (core, adapter) {
        (Ok(()), Ok(())) if model.supports_request(request) => Ok(()),
        (Ok(()), Ok(())) => Err(ConformanceError::new(
            "validate_request accepted request but supports_request returned false",
        )),
        (Ok(()), Err(error)) => Err(ConformanceError::new(format!(
            "core validation accepted request but adapter rejected it: {error}"
        ))),
        (Err(_), Err(_)) if !model.supports_request(request) => Ok(()),
        (Err(_), Err(_)) => Err(ConformanceError::new(
            "core and adapter rejected request but supports_request returned true",
        )),
        (Err(error), Ok(())) => Err(ConformanceError::new(format!(
            "core validation rejected request ({error}) but adapter accepted it"
        ))),
    }
}

/// Verifies basic error taxonomy invariants for adapter-produced errors.
pub fn assert_error_taxonomy(errors: &[ModelError]) -> Result<(), ConformanceError> {
    for error in errors {
        if error.message.trim().is_empty() {
            return Err(ConformanceError::new(
                "ModelError message must not be empty",
            ));
        }
        if error.retryable
            && matches!(
                error.kind,
                ModelErrorKind::Auth
                    | ModelErrorKind::PermissionDenied
                    | ModelErrorKind::InvalidRequest
                    | ModelErrorKind::Unsupported
                    | ModelErrorKind::NativeContext
                    | ModelErrorKind::Abort
            )
        {
            return Err(ConformanceError::new(format!(
                "non-retryable error category was marked retryable: {error}"
            )));
        }
    }
    Ok(())
}

/// Runs a malformed-payload parser probe and fails if it panics or succeeds.
pub fn assert_malformed_payload_returns_error<F>(probe: F) -> Result<(), ConformanceError>
where
    F: FnOnce() -> Result<(), ModelError> + UnwindSafe,
{
    match std::panic::catch_unwind(AssertUnwindSafe(probe)) {
        Ok(Err(_)) => Ok(()),
        Ok(Ok(())) => Err(ConformanceError::new(
            "malformed-payload probe unexpectedly succeeded",
        )),
        Err(_) => Err(ConformanceError::new(
            "malformed-payload probe panicked instead of returning ModelError",
        )),
    }
}

/// Verifies terminal replay capture against the complete declaration matrix.
pub fn assert_replay_artifact(
    descriptor: &LanguageModelDescriptor,
    expected_scope: &NativeContextScope,
    turn: &CompletedTurn,
) -> Result<(), ConformanceError> {
    validate_replay_declaration(&descriptor.capabilities.replay)?;
    let artifact = match (
        descriptor.capabilities.replay.policy,
        descriptor.capabilities.replay.capability,
        turn.finish.native_replay.as_ref(),
    ) {
        (ReplayPolicy::Never, ReplayCapability::Unsupported, None) => return Ok(()),
        (ReplayPolicy::Never, ReplayCapability::Unsupported, Some(_)) => {
            return Err(ConformanceError::new(
                "Never/Unsupported replay must not capture an artifact",
            ));
        }
        (ReplayPolicy::IfValid, ReplayCapability::Optional, None) => return Ok(()),
        (ReplayPolicy::IfValid, ReplayCapability::Optional, Some(artifact))
        | (ReplayPolicy::IfValid, ReplayCapability::Required, Some(artifact))
        | (ReplayPolicy::Always, ReplayCapability::Required, Some(artifact)) => artifact,
        (ReplayPolicy::IfValid, ReplayCapability::Required, None)
        | (ReplayPolicy::Always, ReplayCapability::Required, None) => {
            return Err(ConformanceError::new(
                "Required replay must capture an artifact",
            ));
        }
        _ => {
            return Err(ConformanceError::new(
                "invalid replay policy/capability combination",
            ));
        }
    };
    if artifact.adapter_id() != &descriptor.adapter_id {
        return Err(ConformanceError::new(format!(
            "replay artifact adapter ID {} differs from descriptor {}",
            artifact.adapter_id(),
            descriptor.adapter_id
        )));
    }
    if artifact.scope() != expected_scope {
        return Err(ConformanceError::new(
            "replay artifact scope differs from the exact configured replay scope",
        ));
    }
    if expected_scope.provider_id != descriptor.identity.provider_id
        || expected_scope.model_id != descriptor.identity.model_id
    {
        return Err(ConformanceError::new(
            "expected replay scope differs from descriptor identity",
        ));
    }
    let payload_size = serde_json::to_vec(artifact.payload())
        .map_err(|error| ConformanceError::new(format!("could not size replay payload: {error}")))?
        .len();
    if payload_size > NativeReplayArtifact::MAX_PAYLOAD_BYTES {
        return Err(ConformanceError::new(
            "replay artifact exceeds the 2 MiB cap",
        ));
    }
    if format!("{artifact:?}").contains(&artifact.payload().to_string()) {
        return Err(ConformanceError::new(
            "replay artifact Debug output exposed its payload",
        ));
    }
    Ok(())
}

fn validate_replay_declaration(replay: &ReplayDeclaration) -> Result<(), ConformanceError> {
    if matches!(
        (replay.policy, replay.capability),
        (ReplayPolicy::Never, ReplayCapability::Unsupported)
            | (ReplayPolicy::IfValid, ReplayCapability::Optional)
            | (ReplayPolicy::IfValid, ReplayCapability::Required)
            | (ReplayPolicy::Always, ReplayCapability::Required)
    ) {
        Ok(())
    } else {
        Err(ConformanceError::new(
            "invalid replay policy/capability combination",
        ))
    }
}

/// Verifies that same-adapter artifacts produce explicit replay decisions.
pub async fn assert_replay_round_trip(
    model: &dyn LanguageModel,
    expected_scope: &NativeContextScope,
    request: Request,
) -> Result<ModelStreamReport, ConformanceError> {
    let history = request.history.clone();
    let report = assert_stream_lifecycle(model, request).await?;
    let descriptor = model.descriptor();
    validate_replay_log(
        &history,
        &report.request.replay.decisions,
        &descriptor.adapter_id,
        expected_scope,
        &descriptor.capabilities.replay,
    )?;
    if !report
        .request
        .replay
        .decisions
        .iter()
        .any(|decision| decision.disposition == ReplayDisposition::Replayed)
    {
        return Err(ConformanceError::new(
            "replay round-trip request had no valid same-adapter artifact",
        ));
    }
    Ok(report)
}

/// Verifies invalid replay payload reporting followed by reconstruction.
pub async fn assert_invalid_replay_reconstructs(
    model: &dyn LanguageModel,
    expected_scope: &NativeContextScope,
    request: Request,
) -> Result<ModelStreamReport, ConformanceError> {
    let history = request.history.clone();
    let report = assert_stream_lifecycle(model, request).await?;
    let descriptor = model.descriptor();
    validate_replay_log(
        &history,
        &report.request.replay.decisions,
        &descriptor.adapter_id,
        expected_scope,
        &descriptor.capabilities.replay,
    )?;
    if !report.request.replay.decisions.windows(2).any(|decisions| {
        matches!(
            decisions,
            [
                ReplayDecision {
                    disposition: ReplayDisposition::DiscardedInvalidPayload { .. },
                    ..
                },
                ReplayDecision {
                    disposition: ReplayDisposition::ReconstructedNormalized,
                    ..
                }
            ]
        )
    }) {
        return Err(ConformanceError::new(
            "invalid replay payload was not followed by normalized reconstruction",
        ));
    }
    Ok(report)
}

/// Verifies foreign replay reporting followed by reconstruction.
pub async fn assert_foreign_replay_is_reported(
    model: &dyn LanguageModel,
    expected_scope: &NativeContextScope,
    request: Request,
) -> Result<ModelStreamReport, ConformanceError> {
    let history = request.history.clone();
    let report = assert_stream_lifecycle(model, request).await?;
    let descriptor = model.descriptor();
    validate_replay_log(
        &history,
        &report.request.replay.decisions,
        &descriptor.adapter_id,
        expected_scope,
        &descriptor.capabilities.replay,
    )?;
    if !report.request.replay.decisions.windows(2).any(|decisions| {
        matches!(
            decisions,
            [
                ReplayDecision {
                    disposition: ReplayDisposition::DiscardedForeignAdapter { .. },
                    ..
                },
                ReplayDecision {
                    disposition: ReplayDisposition::ReconstructedNormalized,
                    ..
                }
            ]
        )
    }) {
        return Err(ConformanceError::new(
            "foreign replay artifact was not followed by normalized reconstruction",
        ));
    }
    Ok(report)
}

/// Verifies foreign-scope replay reporting followed by normalized reconstruction.
pub async fn assert_foreign_replay_scope_is_reported(
    model: &dyn LanguageModel,
    expected_scope: &NativeContextScope,
    request: Request,
) -> Result<ModelStreamReport, ConformanceError> {
    let history = request.history.clone();
    let report = assert_stream_lifecycle(model, request).await?;
    let descriptor = model.descriptor();
    validate_replay_log(
        &history,
        &report.request.replay.decisions,
        &descriptor.adapter_id,
        expected_scope,
        &descriptor.capabilities.replay,
    )?;
    if !report.request.replay.decisions.windows(2).any(|decisions| {
        matches!(
            decisions,
            [
                ReplayDecision {
                    disposition: ReplayDisposition::DiscardedForeignScope { .. },
                    ..
                },
                ReplayDecision {
                    disposition: ReplayDisposition::ReconstructedNormalized,
                    ..
                }
            ]
        )
    }) {
        return Err(ConformanceError::new(
            "foreign replay scope was not followed by normalized reconstruction",
        ));
    }
    Ok(report)
}

fn validate_replay_log(
    history: &[HistoryTurn],
    decisions: &[ReplayDecision],
    adapter_id: &AdapterId,
    expected_scope: &NativeContextScope,
    replay: &ReplayDeclaration,
) -> Result<(), ConformanceError> {
    validate_replay_declaration(replay)?;
    let mut cursor = 0;
    for (index, turn) in history.iter().enumerate() {
        let HistoryTurn::Assistant(turn) = turn else {
            continue;
        };
        if replay.policy == ReplayPolicy::Never {
            require_replay_decision(
                decisions,
                &mut cursor,
                index,
                |disposition| matches!(disposition, ReplayDisposition::ReconstructedNormalized),
                "ReconstructedNormalized",
            )?;
            continue;
        }

        match &turn.finish.native_replay {
            None => {
                require_replay_decision(
                    decisions,
                    &mut cursor,
                    index,
                    |disposition| matches!(disposition, ReplayDisposition::NoArtifact),
                    "NoArtifact",
                )?;
                require_reconstruction(decisions, &mut cursor, index)?;
            }
            Some(artifact) if artifact.adapter_id() != adapter_id => {
                require_replay_decision(
                    decisions,
                    &mut cursor,
                    index,
                    |disposition| {
                        matches!(
                            disposition,
                            ReplayDisposition::DiscardedForeignAdapter { found, expected }
                                if found == artifact.adapter_id() && expected == adapter_id
                        )
                    },
                    "DiscardedForeignAdapter with exact IDs",
                )?;
                require_reconstruction(decisions, &mut cursor, index)?;
            }
            Some(artifact) if artifact.scope() != expected_scope => {
                require_replay_decision(
                    decisions,
                    &mut cursor,
                    index,
                    |disposition| {
                        matches!(
                            disposition,
                            ReplayDisposition::DiscardedForeignScope { found, expected }
                                if found == artifact.scope() && expected == expected_scope
                        )
                    },
                    "DiscardedForeignScope with exact scopes",
                )?;
                require_reconstruction(decisions, &mut cursor, index)?;
            }
            Some(_) => {
                let decision = decisions.get(cursor).ok_or_else(|| {
                    ConformanceError::new(format!(
                        "missing replay decision for assistant history entry {index}"
                    ))
                })?;
                if decision.history_index != index {
                    return Err(ConformanceError::new(format!(
                        "replay decision {cursor} refers to history entry {}, expected {index}",
                        decision.history_index
                    )));
                }
                match &decision.disposition {
                    ReplayDisposition::Replayed => cursor += 1,
                    ReplayDisposition::DiscardedInvalidPayload { .. } => {
                        cursor += 1;
                        require_reconstruction(decisions, &mut cursor, index)?;
                    }
                    disposition => {
                        return Err(ConformanceError::new(format!(
                            "same-adapter/scope artifact produced invalid disposition {disposition:?}"
                        )));
                    }
                }
            }
        }
    }
    if cursor != decisions.len() {
        return Err(ConformanceError::new(format!(
            "replay log has {} extra decision(s)",
            decisions.len() - cursor
        )));
    }
    Ok(())
}

fn require_reconstruction(
    decisions: &[ReplayDecision],
    cursor: &mut usize,
    history_index: usize,
) -> Result<(), ConformanceError> {
    require_replay_decision(
        decisions,
        cursor,
        history_index,
        |disposition| matches!(disposition, ReplayDisposition::ReconstructedNormalized),
        "ReconstructedNormalized",
    )
}

fn require_replay_decision(
    decisions: &[ReplayDecision],
    cursor: &mut usize,
    history_index: usize,
    matches: impl FnOnce(&ReplayDisposition) -> bool,
    expected: &str,
) -> Result<(), ConformanceError> {
    let decision = decisions.get(*cursor).ok_or_else(|| {
        ConformanceError::new(format!(
            "missing replay decision for assistant history entry {history_index}"
        ))
    })?;
    if decision.history_index != history_index || !matches(&decision.disposition) {
        return Err(ConformanceError::new(format!(
            "replay decision {} was {:?} for history entry {}, expected {expected} for entry {history_index}",
            *cursor, decision.disposition, decision.history_index
        )));
    }
    *cursor += 1;
    Ok(())
}

/// Verifies that a completed turn can be placed in a new request history.
pub fn assert_history_round_trip(
    model: &dyn LanguageModel,
    turn: CompletedTurn,
) -> Result<(), ConformanceError> {
    let request = Request::new(vec![HistoryTurn::assistant(turn)]);
    request
        .validate_for(model.capabilities())
        .map_err(model_error)?;
    Ok(())
}

/// Transport-level SSE fixtures and byte chunking for adapter parser tests.
///
/// Transport conformance requires incremental UTF-8/SSE parsing with no
/// assumption that network chunks align with event boundaries. These helpers
/// encode provider-shaped SSE documents and re-chunk them at arbitrary byte
/// boundaries (including one-byte chunks that split multi-byte UTF-8
/// sequences) so each adapter can feed its parser pathological input without
/// reimplementing the harness.
pub mod sse {
    /// Line-ending style used when encoding SSE fixtures.
    #[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
    pub enum LineEnding {
        /// `\n` line endings.
        #[default]
        Lf,
        /// `\r\n` line endings.
        Crlf,
    }

    impl LineEnding {
        fn as_str(self) -> &'static str {
            match self {
                Self::Lf => "\n",
                Self::Crlf => "\r\n",
            }
        }
    }

    /// A single server-sent event fixture.
    #[derive(Clone, Debug, Default, Eq, PartialEq)]
    pub struct SseEvent {
        /// Optional event name, emitted as an `event:` field.
        pub event: Option<String>,
        /// Data payload lines; each line becomes its own `data:` field.
        pub data: Vec<String>,
        /// Comment lines emitted before the event fields as `: comment`.
        pub comments: Vec<String>,
    }

    impl SseEvent {
        /// Creates an unnamed data-only event.
        #[must_use]
        pub fn data(data: impl Into<String>) -> Self {
            Self {
                event: None,
                data: vec![data.into()],
                comments: Vec::new(),
            }
        }

        /// Creates a named data event.
        #[must_use]
        pub fn named(event: impl Into<String>, data: impl Into<String>) -> Self {
            Self {
                event: Some(event.into()),
                data: vec![data.into()],
                comments: Vec::new(),
            }
        }

        /// Creates a comment-only line group.
        #[must_use]
        pub fn comment(comment: impl Into<String>) -> Self {
            Self {
                event: None,
                data: Vec::new(),
                comments: vec![comment.into()],
            }
        }

        /// Adds a comment line to this event.
        #[must_use]
        pub fn with_comment(mut self, comment: impl Into<String>) -> Self {
            self.comments.push(comment.into());
            self
        }

        /// Adds another data line, producing multiline `data:` fields.
        #[must_use]
        pub fn with_data_line(mut self, line: impl Into<String>) -> Self {
            self.data.push(line.into());
            self
        }
    }

    /// Encodes events into one SSE document with the selected line endings.
    ///
    /// Events are separated by a blank line; comment-only groups are also
    /// terminated so parsers observe them as standalone dispatch units.
    #[must_use]
    pub fn encode_sse(events: &[SseEvent], ending: LineEnding) -> Vec<u8> {
        let eol = ending.as_str();
        let mut document = String::new();
        for event in events {
            for comment in &event.comments {
                document.push_str(": ");
                document.push_str(comment);
                document.push_str(eol);
            }
            if let Some(name) = &event.event {
                document.push_str("event: ");
                document.push_str(name);
                document.push_str(eol);
            }
            for line in &event.data {
                document.push_str("data: ");
                document.push_str(line);
                document.push_str(eol);
            }
            document.push_str(eol);
        }
        document.into_bytes()
    }

    /// A chunking pattern for feeding encoded documents to incremental parsers.
    #[derive(Clone, Debug, Eq, PartialEq)]
    pub enum ChunkPattern {
        /// One byte per chunk; splits multi-byte UTF-8 sequences across chunks.
        OneByte,
        /// Fixed-size chunks of the given non-zero size.
        Fixed(usize),
        /// Explicit chunk sizes in order. A size larger than the remaining
        /// input is clamped; leftover bytes after the listed sizes form one
        /// final chunk.
        Explicit(Vec<usize>),
    }

    /// Splits a document into chunks according to the pattern.
    ///
    /// Concatenating the returned chunks always restores the input exactly.
    ///
    /// # Panics
    ///
    /// Panics if a [`ChunkPattern::Fixed`] size is zero.
    #[must_use]
    pub fn chunk_bytes(bytes: &[u8], pattern: &ChunkPattern) -> Vec<Vec<u8>> {
        match pattern {
            ChunkPattern::OneByte => bytes.chunks(1).map(<[u8]>::to_vec).collect(),
            ChunkPattern::Fixed(size) => {
                assert!(*size > 0, "fixed chunk size must be non-zero");
                bytes.chunks(*size).map(<[u8]>::to_vec).collect()
            }
            ChunkPattern::Explicit(sizes) => {
                let mut chunks = Vec::new();
                let mut offset = 0_usize;
                for size in sizes {
                    let end = offset.saturating_add(*size).min(bytes.len());
                    if offset < end {
                        chunks.push(bytes[offset..end].to_vec());
                    }
                    offset = end;
                }
                if offset < bytes.len() {
                    chunks.push(bytes[offset..].to_vec());
                }
                chunks
            }
        }
    }
}

struct OpenContentBlock {
    slot: usize,
    text: String,
    metadata: oven_sdk::PartMetadata,
}

fn assembled_content(parts: &[StreamPart]) -> Result<Vec<AssistantPart>, ConformanceError> {
    let mut content = Vec::<Option<AssistantPart>>::new();
    let mut text = BTreeMap::<String, OpenContentBlock>::new();
    let mut reasoning = BTreeMap::<String, OpenContentBlock>::new();
    let mut tools = BTreeMap::<String, usize>::new();
    for part in parts {
        match part {
            StreamPart::TextStart { id, metadata } => {
                let slot = content.len();
                content.push(None);
                text.insert(
                    id.clone(),
                    OpenContentBlock {
                        slot,
                        text: String::new(),
                        metadata: metadata.clone(),
                    },
                );
            }
            StreamPart::TextDelta { id, delta, .. } => text
                .get_mut(id)
                .ok_or_else(|| {
                    ConformanceError::new(format!("text delta for unopened block `{id}`"))
                })?
                .text
                .push_str(delta),
            StreamPart::TextEnd { id, .. } => {
                let block = text.remove(id).ok_or_else(|| {
                    ConformanceError::new(format!("text end for unopened block `{id}`"))
                })?;
                content[block.slot] = Some(AssistantPart::Text(TextPart {
                    text: block.text,
                    metadata: block.metadata,
                }));
            }
            StreamPart::ReasoningStart { id, metadata } => {
                let slot = content.len();
                content.push(None);
                reasoning.insert(
                    id.clone(),
                    OpenContentBlock {
                        slot,
                        text: String::new(),
                        metadata: metadata.clone(),
                    },
                );
            }
            StreamPart::ReasoningDelta { id, delta, .. } => reasoning
                .get_mut(id)
                .ok_or_else(|| {
                    ConformanceError::new(format!("reasoning delta for unopened block `{id}`"))
                })?
                .text
                .push_str(delta),
            StreamPart::ReasoningEnd { id, .. } => {
                let block = reasoning.remove(id).ok_or_else(|| {
                    ConformanceError::new(format!("reasoning end for unopened block `{id}`"))
                })?;
                content[block.slot] = Some(AssistantPart::Reasoning(oven_sdk::ReasoningPart {
                    text: block.text,
                    metadata: block.metadata,
                }));
            }
            StreamPart::ToolCallStart { id, .. } => {
                let slot = content.len();
                content.push(None);
                tools.insert(id.clone(), slot);
            }
            StreamPart::ToolCall { tool_call } => {
                if let Some(slot) = tools.remove(&tool_call.id) {
                    content[slot] = Some(AssistantPart::ToolCall(tool_call.clone()));
                } else {
                    content.push(Some(AssistantPart::ToolCall(tool_call.clone())));
                }
            }
            StreamPart::ToolResult { tool_result } => {
                content.push(Some(AssistantPart::ToolResult(tool_result.clone())));
            }
            StreamPart::Source { source } => {
                content.push(Some(AssistantPart::Source(source.clone())));
            }
            StreamPart::File { file } => {
                content.push(Some(AssistantPart::File(file.clone())));
            }
            StreamPart::Custom { part } => {
                content.push(Some(AssistantPart::Custom(part.clone())));
            }
            StreamPart::ApprovalRequested { approval } => {
                content.push(Some(AssistantPart::ToolApproval(approval.clone())));
            }
            _ => {}
        }
    }
    if !text.is_empty() || !reasoning.is_empty() || !tools.is_empty() {
        return Err(ConformanceError::new(
            "stream ended with open content while assembling complete() expectation",
        ));
    }
    content
        .into_iter()
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| ConformanceError::new("stream left an unfilled ordered content slot"))
}

fn validate_lifecycle(parts: Vec<StreamPart>) -> Result<StreamReport, ConformanceError> {
    let Some(StreamPart::StreamStart { warnings }) = parts.first() else {
        return Err(ConformanceError::new(
            "stream must begin with exactly one StreamStart",
        ));
    };
    let warnings = warnings.clone();
    let mut text = BTreeSet::new();
    let mut reasoning = BTreeSet::new();
    let mut tool = BTreeSet::new();
    let mut ended_tool = BTreeSet::new();
    let mut finish = None;
    let mut in_band_error = false;
    for (index, part) in parts.iter().enumerate() {
        if index > 0 && matches!(part, StreamPart::StreamStart { .. }) {
            return Err(ConformanceError::new(
                "stream emitted multiple StreamStart parts",
            ));
        }
        if finish.is_some() {
            return Err(ConformanceError::new("stream emitted a part after Finish"));
        }
        if in_band_error && !matches!(part, StreamPart::Finish { .. }) {
            return Err(ConformanceError::new(
                "in-band Error was not followed immediately by Finish(Error)",
            ));
        }
        match part {
            StreamPart::TextStart { id, .. } => {
                if !text.insert(id.clone()) {
                    return Err(ConformanceError::new(format!(
                        "duplicate text start `{id}`"
                    )));
                }
            }
            StreamPart::TextDelta { id, .. } if !text.contains(id) => {
                return Err(ConformanceError::new(format!(
                    "text delta without start `{id}`"
                )));
            }
            StreamPart::TextEnd { id, .. } if !text.remove(id) => {
                return Err(ConformanceError::new(format!(
                    "text end without start `{id}`"
                )));
            }
            StreamPart::ReasoningStart { id, .. } => {
                if !reasoning.insert(id.clone()) {
                    return Err(ConformanceError::new(format!(
                        "duplicate reasoning start `{id}`"
                    )));
                }
            }
            StreamPart::ReasoningDelta { id, .. } if !reasoning.contains(id) => {
                return Err(ConformanceError::new(format!(
                    "reasoning delta without start `{id}`"
                )));
            }
            StreamPart::ReasoningEnd { id, .. } if !reasoning.remove(id) => {
                return Err(ConformanceError::new(format!(
                    "reasoning end without start `{id}`"
                )));
            }
            StreamPart::ToolCallStart { id, .. } => {
                if !tool.insert(id.clone()) {
                    return Err(ConformanceError::new(format!(
                        "duplicate tool-call start `{id}`"
                    )));
                }
            }
            StreamPart::ToolCallDelta { id, .. } if !tool.contains(id) => {
                return Err(ConformanceError::new(format!(
                    "tool-call delta without start `{id}`"
                )));
            }
            StreamPart::ToolCallEnd { id, .. } => {
                if !tool.remove(id) {
                    return Err(ConformanceError::new(format!(
                        "tool-call end without start `{id}`"
                    )));
                }
                ended_tool.insert(id.clone());
            }
            StreamPart::ToolCall { tool_call } => {
                if !ended_tool.remove(&tool_call.id) && tool.contains(&tool_call.id) {
                    return Err(ConformanceError::new(format!(
                        "tool call `{}` finalized before end",
                        tool_call.id
                    )));
                }
            }
            StreamPart::Error { .. } => in_band_error = true,
            StreamPart::Finish { finish: candidate } => {
                if !text.is_empty()
                    || !reasoning.is_empty()
                    || !tool.is_empty()
                    || !ended_tool.is_empty()
                {
                    return Err(ConformanceError::new(
                        "Finish arrived with unclosed or unfinalized blocks",
                    ));
                }
                if in_band_error && candidate.finish_reason != FinishReason::Error {
                    return Err(ConformanceError::new(
                        "in-band Error requires Finish(Error)",
                    ));
                }
                finish = Some(candidate.clone());
            }
            _ => {}
        }
    }
    let finish =
        finish.ok_or_else(|| ConformanceError::unexpected_eof("EOF before mandatory Finish"))?;
    Ok(StreamReport {
        parts,
        finish,
        warnings,
    })
}

/// Returns the normalized probe requests used by [`assert_capability_honesty`].
///
/// Adapter test suites can reuse these requests to build their own
/// capability matrices; each entry pairs the claimed capability with a
/// minimal request that exercises it.
#[must_use]
pub fn capability_probe_requests() -> Vec<(Capability, Request, &'static str)> {
    let schema = oven_sdk::JsonSchema::new(serde_json::json!({
        "type":"object",
        "properties":{},
        "required":[],
        "additionalProperties":false
    }))
    .expect("object schema");
    let mut temperature = InferenceOptions::new();
    temperature.temperature = Some(0.5);
    let mut top_p = InferenceOptions::new();
    top_p.top_p = Some(0.9);
    let mut output = InferenceOptions::new();
    output.max_output_tokens = Some(1);
    let mut reasoning = InferenceOptions::new();
    reasoning.reasoning_effort = Some("medium".into());
    vec![
        (
            Capability::TOOL_CALLING,
            Request::new(Vec::new()).with_tools(vec![ToolDefinition::new(
                "probe",
                "conformance tool",
                schema.clone(),
            )]),
            "TOOL_CALLING",
        ),
        (
            Capability::STRUCTURED_OUTPUT,
            Request::new(Vec::new()).with_response_format(ResponseFormat::structured(schema)),
            "STRUCTURED_OUTPUT",
        ),
        (
            Capability::TEMPERATURE,
            Request::new(Vec::new()).with_inference(temperature),
            "TEMPERATURE",
        ),
        (
            Capability::TOP_P,
            Request::new(Vec::new()).with_inference(top_p),
            "TOP_P",
        ),
        (
            Capability::MAX_OUTPUT_TOKENS,
            Request::new(Vec::new()).with_inference(output),
            "MAX_OUTPUT_TOKENS",
        ),
        (
            Capability::REASONING,
            Request::new(Vec::new()).with_inference(reasoning),
            "REASONING",
        ),
    ]
}

/// Maximum MIME declarations exercised by one bounded media fixture set.
pub const MAX_MEDIA_PROBE_PATTERNS: usize = 64;

/// Maximum generated media probes, including negative source-form cases.
pub const MAX_MEDIA_PROBES: usize = 512;

/// Source form exercised by a media conformance probe.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum MediaProbeSource {
    /// Inline bytes.
    InlineBytes,
    /// Inline text.
    InlineText,
    /// Caller-owned URL.
    Url,
    /// Provider-native reference.
    ProviderReference,
}

impl MediaProbeSource {
    const ALL: [Self; 4] = [
        Self::InlineBytes,
        Self::InlineText,
        Self::Url,
        Self::ProviderReference,
    ];

    fn capability(self) -> MediaSourceSupport {
        match self {
            Self::InlineBytes => MediaSourceSupport::INLINE_BYTES,
            Self::InlineText => MediaSourceSupport::INLINE_TEXT,
            Self::Url => MediaSourceSupport::URL,
            Self::ProviderReference => MediaSourceSupport::PROVIDER_REFERENCE,
        }
    }

    fn fixture(self) -> FileSource {
        match self {
            Self::InlineBytes => FileSource::Bytes(b"conformance".to_vec().into()),
            Self::InlineText => FileSource::Text("conformance".into()),
            Self::Url => FileSource::Url(
                "https://example.com/conformance"
                    .parse()
                    .expect("fixture URL"),
            ),
            Self::ProviderReference => FileSource::ProviderReference {
                provider: ProviderId::new("conformance.fixture"),
                id: "file-1".into(),
            },
        }
    }
}

/// One request exercising an explicit media MIME/source combination.
#[derive(Clone, Debug)]
pub struct MediaProbe {
    /// Modality whose declaration introduced this representative MIME value.
    pub modality: Modality,
    /// Exact declared MIME value or trailing-wildcard pattern under test.
    pub declared_media_type: String,
    /// Concrete MIME value used by the fixture.
    pub media_type: String,
    /// Source form used by the fixture.
    pub source: MediaProbeSource,
    /// Whether the complete declaration permits this MIME/source combination.
    pub expected_supported: bool,
    /// Representative request.
    pub request: Request,
}

/// Builds exhaustive bounded MIME-pattern/value × source-form probes.
pub fn media_probe_requests(
    capabilities: &ModelCapabilities,
) -> Result<Vec<MediaProbe>, ConformanceError> {
    let pattern_count = capabilities
        .media
        .input
        .values()
        .map(|support| support.media_types.len())
        .sum::<usize>();
    if pattern_count > MAX_MEDIA_PROBE_PATTERNS {
        return Err(ConformanceError::new(format!(
            "media declaration has {pattern_count} MIME patterns; maximum is {MAX_MEDIA_PROBE_PATTERNS}"
        )));
    }

    let mut fixtures = Vec::<(String, String, MediaProbeSource, Modality)>::new();
    for (modality, support) in &capabilities.media.input {
        for declared in &support.media_types {
            let media_type = declared
                .strip_suffix('*')
                .map_or_else(|| declared.clone(), |prefix| format!("{prefix}conformance"));
            for source in MediaProbeSource::ALL {
                fixtures.push((
                    declared.clone(),
                    media_type.clone(),
                    source,
                    modality.clone(),
                ));
            }
        }
    }

    let undeclared = (0..=MAX_MEDIA_PROBE_PATTERNS)
        .map(|index| format!("oven-conformance-{index}/unsupported"))
        .find(|candidate| !media_type_is_declared(capabilities, candidate))
        .ok_or_else(|| ConformanceError::new("could not construct undeclared MIME fixture"))?;
    for source in MediaProbeSource::ALL {
        fixtures.push((
            undeclared.clone(),
            undeclared.clone(),
            source,
            Modality::new("undeclared").map_err(model_error)?,
        ));
    }

    if fixtures.len() > MAX_MEDIA_PROBES {
        return Err(ConformanceError::new(format!(
            "media declaration generates {} probes; maximum is {MAX_MEDIA_PROBES}",
            fixtures.len()
        )));
    }

    Ok(fixtures
        .into_iter()
        .map(|(declared_media_type, media_type, source, modality)| {
            let expected_supported = capabilities.media.input.values().any(|support| {
                support.sources.contains(source.capability())
                    && support
                        .media_types
                        .iter()
                        .any(|declared| media_type_matches(declared, &media_type))
            });
            MediaProbe {
                modality,
                declared_media_type,
                media_type: media_type.clone(),
                source,
                expected_supported,
                request: media_request(FilePart::new(media_type, source.fixture())),
            }
        })
        .collect())
}

fn media_type_is_declared(capabilities: &ModelCapabilities, media_type: &str) -> bool {
    capabilities.media.input.values().any(|support| {
        support
            .media_types
            .iter()
            .any(|declared| media_type_matches(declared, media_type))
    })
}

fn media_type_matches(declared: &str, media_type: &str) -> bool {
    declared.strip_suffix('*').map_or_else(
        || declared.eq_ignore_ascii_case(media_type),
        |prefix| {
            media_type
                .get(..prefix.len())
                .is_some_and(|value| value.eq_ignore_ascii_case(prefix))
        },
    )
}

/// Verifies every positive and negative media probe against model validation.
pub fn assert_media_honesty(model: &dyn LanguageModel) -> Result<(), ConformanceError> {
    let capabilities = model.capabilities();
    for probe in media_probe_requests(capabilities)? {
        let validation = model.validate_request(&probe.request);
        if probe.expected_supported {
            validation.map_err(|error| {
                ConformanceError::new(format!(
                    "model rejects declared `{}` media `{}`/{:?}: {error}",
                    probe.modality.as_str(),
                    probe.media_type,
                    probe.source
                ))
            })?;
            if !model.supports_request(&probe.request) {
                return Err(ConformanceError::new(format!(
                    "supports_request rejects declared media `{}`/{:?}",
                    probe.media_type, probe.source
                )));
            }
        } else {
            if !validation
                .as_ref()
                .is_err_and(|error| error.is_kind(ModelErrorKind::Unsupported))
            {
                return Err(ConformanceError::new(format!(
                    "undeclared media `{}`/{:?} was not rejected as Unsupported",
                    probe.media_type, probe.source
                )));
            }
            if model.supports_request(&probe.request) {
                return Err(ConformanceError::new(format!(
                    "supports_request accepted undeclared media `{}`/{:?}",
                    probe.media_type, probe.source
                )));
            }
        }
    }
    Ok(())
}

fn media_request(file: FilePart) -> Request {
    Request::new(vec![HistoryTurn::user(UserMessage::new(vec![
        InputPart::File(file),
    ]))])
}

fn model_error(error: ModelError) -> ConformanceError {
    ConformanceError::model_error(format!("model returned {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use oven_sdk::ToolCallPart;
    use std::{cell::RefCell, marker::PhantomPinned};

    struct PinnedFixtureStream {
        items: RefCell<VecDeque<StreamItem>>,
        _pin: PhantomPinned,
    }

    impl Stream for PinnedFixtureStream {
        type Item = StreamItem;

        fn poll_next(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Option<Self::Item>> {
            Poll::Ready(self.items.borrow_mut().pop_front())
        }
    }

    struct DivergentCompleteModel {
        inner: MockLanguageModel,
        extra_warning: bool,
        alter_tool_input: bool,
        alter_source: bool,
        alter_response: bool,
        alter_request_metadata: bool,
        replacement_warnings: Option<Vec<String>>,
    }

    impl LanguageModel for DivergentCompleteModel {
        fn descriptor(&self) -> &LanguageModelDescriptor {
            self.inner.descriptor()
        }
        fn validate_request(&self, request: &Request) -> Result<(), ModelError> {
            self.inner.validate_request(request)
        }
        fn supports_request(&self, request: &Request) -> bool {
            self.inner.supports_request(request)
        }
        fn stream<'a>(
            &'a self,
            request: Request,
            abort: AbortSignal,
        ) -> BoxFuture<'a, Result<StreamResponse, ModelError>> {
            self.inner.stream(request, abort)
        }
        fn complete<'a>(
            &'a self,
            request: Request,
            abort: AbortSignal,
        ) -> BoxFuture<'a, Result<CompleteResult, ModelError>> {
            Box::pin(async move {
                let mut result = self.inner.complete(request, abort).await?;
                if self.extra_warning {
                    result.turn.warnings.push("extra".into());
                }
                if let Some(warnings) = &self.replacement_warnings {
                    result.turn.warnings = warnings.clone();
                }
                if self.alter_tool_input {
                    for part in &mut result.turn.message.content {
                        if let AssistantPart::ToolCall(call) = part {
                            call.input = serde_json::json!({"different": true});
                        }
                    }
                }
                if self.alter_source {
                    for part in &mut result.turn.message.content {
                        if let AssistantPart::Source(source) = part {
                            source.title = Some("different".into());
                        }
                    }
                }
                if self.alter_response {
                    result.response.request_id = Some("different".into());
                }
                if self.alter_request_metadata {
                    result
                        .request
                        .provider_metadata
                        .insert("different".into(), serde_json::json!(true));
                }
                Ok(result)
            })
        }
    }

    fn block_on<F: std::future::Future>(future: F) -> F::Output {
        let mut future = std::pin::pin!(future);
        let mut context = Context::from_waker(std::task::Waker::noop());
        match future.as_mut().poll(&mut context) {
            Poll::Ready(output) => output,
            Poll::Pending => panic!("fixture future unexpectedly pending"),
        }
    }

    fn mock_scope(model_id: &str, resource: &str) -> NativeContextScope {
        NativeContextScope::new(
            ProviderId::new("conformance.mock"),
            ModelId::new(model_id),
            ResourceId::new(resource).expect("resource"),
        )
        .expect("scope")
    }

    fn declared_capabilities() -> ModelCapabilities {
        let mut capabilities = ModelCapabilities::conservative();
        capabilities.features = Capability::TOOL_CALLING
            | Capability::STRUCTURED_OUTPUT
            | Capability::TEMPERATURE
            | Capability::TOP_P
            | Capability::MAX_OUTPUT_TOKENS
            | Capability::REASONING;
        capabilities.limits.output = Some(4_096);
        capabilities.modalities.input.extend([
            Modality::image(),
            Modality::audio(),
            Modality::video(),
            Modality::pdf(),
        ]);
        for (modality, media_type) in [
            (Modality::image(), "image/png"),
            (Modality::audio(), "audio/mpeg"),
            (Modality::video(), "video/mp4"),
            (Modality::pdf(), "application/pdf"),
        ] {
            capabilities.media.input.insert(
                modality,
                oven_sdk::MediaInputSupport::new(
                    [media_type.to_owned()],
                    MediaSourceSupport::INLINE_BYTES,
                )
                .expect("media support"),
            );
        }
        capabilities
    }

    fn optional_replay_capabilities() -> ModelCapabilities {
        let mut capabilities = ModelCapabilities::conservative();
        capabilities.replay = ReplayDeclaration {
            policy: ReplayPolicy::IfValid,
            capability: ReplayCapability::Optional,
            reasoning: false,
        };
        capabilities
    }

    fn required_replay_capabilities(policy: ReplayPolicy) -> ModelCapabilities {
        let mut capabilities = ModelCapabilities::conservative();
        capabilities.replay = ReplayDeclaration {
            policy,
            capability: ReplayCapability::Required,
            reasoning: false,
        };
        capabilities
    }

    fn native_compaction_capabilities() -> ModelCapabilities {
        let mut capabilities = ModelCapabilities::conservative();
        capabilities.compaction = CompactionCapability::Native;
        capabilities.cancellation = CancellationCapability::LocalOnly;
        capabilities
    }

    fn compaction_result(scope: &NativeContextScope, marker: &str) -> CompactionResult {
        CompactionResult::new(
            NativeContextWindow::new(
                AdapterId::new("conformance.mock"),
                scope.clone(),
                serde_json::json!({"marker": marker}),
            )
            .expect("native context"),
        )
    }

    fn descriptor_with_capabilities(capabilities: ModelCapabilities) -> LanguageModelDescriptor {
        LanguageModelDescriptor::new(
            ModelIdentity::new(
                ProviderId::new("conformance.mock"),
                ModelId::new("scripted"),
            )
            .expect("identity"),
            AdapterId::new("conformance.mock"),
            capabilities,
        )
        .expect("descriptor")
    }

    fn empty_turn(artifact: Option<NativeReplayArtifact>) -> CompletedTurn {
        let mut finish = Finish::new(Default::default(), FinishReason::Stop);
        finish.native_replay = artifact;
        CompletedTurn::new(oven_sdk::AssistantMessage::new(Vec::new()), finish)
    }

    #[test]
    fn mock_model_passes_lifecycle_drain_and_history_suites() {
        let model = MockLanguageModel::builder().build();
        let request = Request::new(Vec::new());
        block_on(assert_stream_lifecycle(&model, request.clone())).expect("lifecycle");
        let turn = block_on(assert_complete_drain(&model, request)).expect("drain");
        assert_history_round_trip(&model, turn.turn).expect("history");
    }

    #[test]
    fn drain_suite_accepts_interleaved_blocks_in_start_order() {
        let script = StreamFixture::from_parts(vec![
            StreamPart::StreamStart {
                warnings: Vec::new(),
            },
            StreamPart::TextStart {
                id: "text".into(),
                metadata: None,
            },
            StreamPart::ReasoningStart {
                id: "reasoning".into(),
                metadata: None,
            },
            StreamPart::ReasoningDelta {
                id: "reasoning".into(),
                delta: "thought".into(),
                metadata: None,
            },
            StreamPart::ReasoningEnd {
                id: "reasoning".into(),
                metadata: None,
            },
            StreamPart::TextDelta {
                id: "text".into(),
                delta: "answer".into(),
                metadata: None,
            },
            StreamPart::TextEnd {
                id: "text".into(),
                metadata: None,
            },
            StreamPart::Finish {
                finish: Finish::new(Default::default(), FinishReason::Stop),
            },
        ]);
        let model = MockLanguageModel::builder().script(script).build();
        block_on(assert_complete_drain(&model, Request::new(Vec::new())))
            .expect("interleaved drain");
    }

    #[test]
    fn complete_drain_compares_all_ordered_content_and_response_head() {
        let mut source = oven_sdk::SourcePart::new();
        source.title = Some("source".into());
        let file = FilePart::new("text/plain", FileSource::Text("file".into()));
        let custom = oven_sdk::CustomPart::new(
            "google.server_tool_call",
            serde_json::json!({"provider_tool":"google_search"}),
        );
        let tool_call = ToolCallPart::new("call", "lookup", serde_json::json!({"x":1}));
        let tool_result =
            oven_sdk::ToolResultPart::new("call", oven_sdk::ToolContent::Text("result".into()));
        let approval = oven_sdk::ToolApprovalPart::new("approval-call");
        let script = StreamFixture::from_parts(vec![
            StreamPart::StreamStart {
                warnings: Vec::new(),
            },
            StreamPart::TextStart {
                id: "text".into(),
                metadata: Some(BTreeMap::from([(
                    "part".into(),
                    serde_json::json!("metadata"),
                )])),
            },
            StreamPart::Source { source },
            StreamPart::File { file },
            StreamPart::Custom { part: custom },
            StreamPart::ToolCall { tool_call },
            StreamPart::ToolResult { tool_result },
            StreamPart::ApprovalRequested { approval },
            StreamPart::TextDelta {
                id: "text".into(),
                delta: "answer".into(),
                metadata: None,
            },
            StreamPart::TextEnd {
                id: "text".into(),
                metadata: None,
            },
            StreamPart::Finish {
                finish: Finish::new(Default::default(), FinishReason::ToolCalls),
            },
        ]);
        let response_head = ResponseHead {
            http_status: Some(200),
            request_id: Some("request".into()),
            response_metadata: BTreeMap::from([("head".into(), serde_json::json!(true))]),
        };
        let model = MockLanguageModel::builder()
            .script(script)
            .request_metadata(RequestMetadata {
                replay: ReplayOutcome::default(),
                provider_metadata: BTreeMap::from([(
                    "request.provider_tool".into(),
                    serde_json::json!("google_search"),
                )]),
            })
            .response_head(response_head.clone())
            .build();
        let completed = block_on(assert_complete_drain(&model, Request::new(Vec::new())))
            .expect("full ordered drain");
        assert_eq!(completed.response, response_head);
        assert_eq!(completed.turn.message.content.len(), 7);
        assert_eq!(
            completed.request.provider_metadata["request.provider_tool"],
            "google_search"
        );
        assert!(matches!(
            completed.turn.message.content[0],
            AssistantPart::Text(_)
        ));
        assert!(matches!(
            completed.turn.message.content[6],
            AssistantPart::ToolApproval(_)
        ));
    }

    #[test]
    fn mock_compaction_queue_capture_cancellation_and_round_trip_are_conformant() {
        let scope = mock_scope("scripted", "conformance.mock.resource");
        let request = CompactionRequest::new(Request::new(Vec::new()));
        let model = MockLanguageModel::builder()
            .capabilities(native_compaction_capabilities())
            .native_context_scope(scope.clone())
            .compaction_result(Ok(compaction_result(&scope, "first")))
            .compaction_result(Ok(compaction_result(&scope, "second")))
            .build();

        assert_declaration_honesty(&model).expect("declaration");
        let first = block_on(assert_native_compaction(&model, &scope, request.clone()))
            .expect("native compaction");
        assert_eq!(first.native_context.payload()["marker"], "first");
        let (second, _) = block_on(assert_compaction_round_trip(
            &model,
            &scope,
            request.clone(),
            Request::new(Vec::new()),
        ))
        .expect("compaction round trip");
        assert_eq!(second.native_context.payload()["marker"], "second");
        assert_eq!(model.captured_compactions(), vec![request.clone(), request]);

        let cancelled = MockLanguageModel::builder()
            .capabilities(native_compaction_capabilities())
            .native_context_scope(scope)
            .build();
        block_on(assert_compaction_cancellation(
            &cancelled,
            CompactionRequest::new(Request::new(Vec::new())),
        ))
        .expect("cancelled compaction");
        assert!(cancelled.captured_compactions().is_empty());
    }

    #[test]
    fn unsupported_compaction_fails_before_mock_io_capture() {
        let model = MockLanguageModel::builder().build();
        block_on(assert_compaction_unsupported_before_io(
            &model,
            CompactionRequest::new(Request::new(Vec::new())),
        ))
        .expect("unsupported before I/O");
        assert!(model.captured_compactions().is_empty());
    }

    #[test]
    fn complete_drain_rejects_source_or_response_head_divergence() {
        let source_script = StreamFixture::from_parts(vec![
            StreamPart::StreamStart {
                warnings: Vec::new(),
            },
            StreamPart::Source {
                source: oven_sdk::SourcePart::new(),
            },
            StreamPart::Finish {
                finish: Finish::new(Default::default(), FinishReason::Stop),
            },
        ]);
        let source_model = DivergentCompleteModel {
            inner: MockLanguageModel::builder().script(source_script).build(),
            extra_warning: false,
            alter_tool_input: false,
            alter_source: true,
            alter_response: false,
            alter_request_metadata: false,
            replacement_warnings: None,
        };
        block_on(assert_complete_drain(
            &source_model,
            Request::new(Vec::new()),
        ))
        .expect_err("source divergence must fail");

        let response_model = DivergentCompleteModel {
            inner: MockLanguageModel::builder()
                .response_head(ResponseHead {
                    http_status: Some(200),
                    request_id: Some("request".into()),
                    response_metadata: BTreeMap::new(),
                })
                .build(),
            extra_warning: false,
            alter_tool_input: false,
            alter_source: false,
            alter_response: true,
            alter_request_metadata: false,
            replacement_warnings: None,
        };
        block_on(assert_complete_drain(
            &response_model,
            Request::new(Vec::new()),
        ))
        .expect_err("response-head divergence must fail");

        let request_metadata_model = DivergentCompleteModel {
            inner: MockLanguageModel::builder()
                .request_metadata(RequestMetadata {
                    replay: ReplayOutcome::default(),
                    provider_metadata: BTreeMap::from([("expected".into(), serde_json::json!(1))]),
                })
                .build(),
            extra_warning: false,
            alter_tool_input: false,
            alter_source: false,
            alter_response: false,
            alter_request_metadata: true,
            replacement_warnings: None,
        };
        block_on(assert_complete_drain(
            &request_metadata_model,
            Request::new(Vec::new()),
        ))
        .expect_err("request provider metadata divergence must fail");
    }

    #[test]
    fn complete_drain_rejects_extra_warnings() {
        let model = DivergentCompleteModel {
            inner: MockLanguageModel::builder().build(),
            extra_warning: true,
            alter_tool_input: false,
            alter_source: false,
            alter_response: false,
            alter_request_metadata: false,
            replacement_warnings: None,
        };
        block_on(assert_complete_drain(&model, Request::new(Vec::new())))
            .expect_err("extra complete warning must fail");
    }

    #[test]
    fn complete_drain_rejects_different_tool_input() {
        let script = StreamFixture::from_parts(vec![
            StreamPart::StreamStart {
                warnings: Vec::new(),
            },
            StreamPart::ToolCall {
                tool_call: ToolCallPart::new("call", "tool", serde_json::json!({"value": 1})),
            },
            StreamPart::Finish {
                finish: Finish::new(Default::default(), FinishReason::ToolCalls),
            },
        ]);
        let model = DivergentCompleteModel {
            inner: MockLanguageModel::builder().script(script).build(),
            extra_warning: false,
            alter_tool_input: true,
            alter_source: false,
            alter_response: false,
            alter_request_metadata: false,
            replacement_warnings: None,
        };
        block_on(assert_complete_drain(&model, Request::new(Vec::new())))
            .expect_err("different complete tool input must fail");
    }

    #[test]
    fn complete_drain_rejects_collapsed_or_reordered_warnings() {
        for warnings in [vec!["a".into()], vec!["b".into(), "a".into()]] {
            let script = StreamFixture::from_parts(vec![
                StreamPart::StreamStart {
                    warnings: vec!["a".into(), "b".into()],
                },
                StreamPart::Finish {
                    finish: Finish::new(Default::default(), FinishReason::Stop),
                },
            ]);
            let model = DivergentCompleteModel {
                inner: MockLanguageModel::builder().script(script).build(),
                extra_warning: false,
                alter_tool_input: false,
                alter_source: false,
                alter_response: false,
                alter_request_metadata: false,
                replacement_warnings: Some(warnings),
            };
            block_on(assert_complete_drain(&model, Request::new(Vec::new())))
                .expect_err("warning multiplicity and order must match");
        }
    }

    #[test]
    fn mock_model_honors_capability_probes() {
        let model = MockLanguageModel::builder()
            .capabilities(declared_capabilities())
            .build();
        assert_capability_honesty(&model).expect("capability probes");
        assert_media_honesty(&model).expect("media probes");
    }

    #[test]
    fn media_probes_cover_all_patterns_sources_negatives_and_bounds() {
        let modality = Modality::new("chemical").expect("modality");
        let mut capabilities = ModelCapabilities::conservative();
        capabilities.modalities.input.insert(modality.clone());
        capabilities.media.input.insert(
            modality,
            oven_sdk::MediaInputSupport::new(
                ["chemical/conformance".to_owned(), "chemical/*".to_owned()],
                MediaSourceSupport::INLINE_BYTES | MediaSourceSupport::URL,
            )
            .expect("support"),
        );
        let probes = media_probe_requests(&capabilities).expect("probes");
        assert_eq!(probes.len(), 12);
        assert_eq!(
            probes
                .iter()
                .filter(|probe| probe.expected_supported)
                .count(),
            4
        );
        assert_eq!(
            probes
                .iter()
                .filter(|probe| !probe.expected_supported)
                .count(),
            8
        );

        let modality = Modality::new("bounded").expect("modality");
        let mut excessive = ModelCapabilities::conservative();
        excessive.modalities.input.insert(modality.clone());
        excessive.media.input.insert(
            modality,
            oven_sdk::MediaInputSupport::new(
                (0..=MAX_MEDIA_PROBE_PATTERNS).map(|index| format!("bounded/type-{index}")),
                MediaSourceSupport::INLINE_BYTES,
            )
            .expect("support"),
        );
        assert!(media_probe_requests(&excessive).is_err());
    }

    #[test]
    fn mock_model_rejects_unclaimed_capability_probes() {
        assert_capability_honesty(&MockLanguageModel::builder().build())
            .expect("unclaimed capability probes");
    }

    #[test]
    fn lifecycle_suite_rejects_eof_before_finish() {
        let model = MockLanguageModel::builder()
            .script(StreamFixture::eof_before_finish())
            .build();
        let error = block_on(assert_stream_lifecycle(&model, Request::new(Vec::new())))
            .expect_err("EOF must fail lifecycle conformance");
        assert_eq!(error.kind(), ConformanceErrorKind::UnexpectedEof);
    }

    #[test]
    fn stream_contract_accepts_parser_level_fixture_stream() {
        let fixture = StreamFixture::valid_text("parser output");
        let stream = FixtureStream(VecDeque::from(fixture.items().to_vec()));
        let report = block_on(assert_stream_contract(stream)).expect("parser stream contract");
        assert_eq!(report.parts.len(), fixture.items().len());
    }

    #[test]
    fn stream_contract_rejects_parser_level_eof_before_finish() {
        let fixture = StreamFixture::eof_before_finish();
        let stream = FixtureStream(VecDeque::from(fixture.items().to_vec()));
        let error = block_on(assert_stream_contract(stream))
            .expect_err("EOF must fail parser stream conformance");
        assert_eq!(error.kind(), ConformanceErrorKind::UnexpectedEof);
    }

    #[test]
    fn stream_contract_accepts_non_unpin_parser_stream() {
        let fixture = StreamFixture::valid_text("pinned parser output");
        let stream = PinnedFixtureStream {
            items: RefCell::new(VecDeque::from(fixture.items().to_vec())),
            _pin: PhantomPinned,
        };
        let report = block_on(assert_stream_contract(stream)).expect("pinned parser stream");
        assert_eq!(report.parts.len(), fixture.items().len());
    }

    #[test]
    fn replay_artifact_assertions_enforce_the_complete_matrix() {
        let scope = mock_scope("scripted", "conformance.mock.resource");
        let exact_artifact = || {
            NativeReplayArtifact::new(
                AdapterId::new("conformance.mock"),
                scope.clone(),
                serde_json::json!({"ok": true}),
            )
            .expect("artifact")
        };

        let conservative = descriptor_with_capabilities(ModelCapabilities::conservative());
        assert_replay_artifact(&conservative, &scope, &empty_turn(None))
            .expect("Never/Unsupported without artifact");
        assert_replay_artifact(&conservative, &scope, &empty_turn(Some(exact_artifact())))
            .expect_err("Never/Unsupported must reject capture");

        let optional = descriptor_with_capabilities(optional_replay_capabilities());
        assert_replay_artifact(&optional, &scope, &empty_turn(None))
            .expect("Optional artifact may be absent");
        assert_replay_artifact(&optional, &scope, &empty_turn(Some(exact_artifact())))
            .expect("Optional exact artifact");
        let foreign_scope = NativeReplayArtifact::new(
            AdapterId::new("conformance.mock"),
            mock_scope("scripted", "other-resource"),
            serde_json::json!({"ok": true}),
        )
        .expect("artifact");
        assert_replay_artifact(&optional, &scope, &empty_turn(Some(foreign_scope)))
            .expect_err("Optional artifact must have exact scope");
        let foreign_adapter = NativeReplayArtifact::new(
            AdapterId::new("foreign.adapter"),
            scope.clone(),
            serde_json::json!({"ok": true}),
        )
        .expect("artifact");
        assert_replay_artifact(&optional, &scope, &empty_turn(Some(foreign_adapter)))
            .expect_err("Optional artifact must have exact adapter");

        for policy in [ReplayPolicy::IfValid, ReplayPolicy::Always] {
            let required = descriptor_with_capabilities(required_replay_capabilities(policy));
            assert_replay_artifact(&required, &scope, &empty_turn(None))
                .expect_err("Required replay must capture");
            assert_replay_artifact(&required, &scope, &empty_turn(Some(exact_artifact())))
                .expect("Required exact artifact");
        }

        let mut invalid = optional_replay_capabilities();
        invalid.replay.policy = ReplayPolicy::Never;
        let mut descriptor = descriptor_with_capabilities(optional_replay_capabilities());
        descriptor.capabilities = invalid;
        assert_replay_artifact(&descriptor, &scope, &empty_turn(None))
            .expect_err("invalid matrix must fail conformance");
    }

    #[test]
    fn never_replay_reconstructs_without_inspecting_attached_artifacts() {
        let scope = mock_scope("scripted", "conformance.mock.resource");
        let artifact = NativeReplayArtifact::new(
            AdapterId::new("foreign.adapter"),
            mock_scope("foreign-model", "foreign-resource"),
            serde_json::json!("garbage"),
        )
        .expect("artifact");
        let request = Request::new(vec![HistoryTurn::assistant(empty_turn(Some(artifact)))]);
        let model = MockLanguageModel::builder()
            .native_context_scope(scope.clone())
            .build();
        let report = block_on(assert_stream_lifecycle(&model, request)).expect("stream");
        assert_eq!(
            report.request.replay.decisions,
            vec![ReplayDecision {
                history_index: 0,
                disposition: ReplayDisposition::ReconstructedNormalized,
            }]
        );
        validate_replay_log(
            &[HistoryTurn::assistant(empty_turn(None))],
            &report.request.replay.decisions,
            &model.descriptor().adapter_id,
            &scope,
            &model.descriptor().capabilities.replay,
        )
        .expect("Never disposition");
    }

    #[test]
    fn replay_suite_checks_artifact_and_garbage_reconstruction() {
        let adapter = AdapterId::new("conformance.mock");
        let scope = mock_scope("scripted", "conformance.mock.resource");
        let artifact = NativeReplayArtifact::new(
            adapter.clone(),
            scope.clone(),
            serde_json::json!({"ok": true}),
        )
        .expect("small artifact");
        let mut replay_capabilities = ModelCapabilities::conservative();
        replay_capabilities.replay = oven_sdk::ReplayDeclaration {
            policy: oven_sdk::ReplayPolicy::IfValid,
            capability: oven_sdk::ReplayCapability::Optional,
            reasoning: false,
        };
        let model = MockLanguageModel::builder()
            .capabilities(replay_capabilities)
            .native_context_scope(scope.clone())
            .native_replay(artifact)
            .build();
        let turn = block_on(assert_complete_drain(&model, Request::new(Vec::new())))
            .expect("turn")
            .turn;
        assert_replay_artifact(model.descriptor(), &scope, &turn).expect("artifact");
        let garbage =
            NativeReplayArtifact::new(adapter, scope.clone(), serde_json::json!("garbage"))
                .expect("artifact");
        let mut garbage_turn = turn;
        garbage_turn.finish.native_replay = Some(garbage);
        block_on(assert_invalid_replay_reconstructs(
            &model,
            &scope,
            Request::new(vec![HistoryTurn::assistant(garbage_turn)]),
        ))
        .expect("garbage reconstruction");
    }

    #[test]
    fn replay_round_trip_accepts_same_adapter_and_rejects_foreign_artifacts() {
        let adapter = AdapterId::new("conformance.mock");
        let scope = mock_scope("scripted", "conformance.mock.resource");
        let artifact =
            NativeReplayArtifact::new(adapter, scope.clone(), serde_json::json!({"ok": true}))
                .expect("small artifact");
        let model = MockLanguageModel::builder()
            .capabilities(optional_replay_capabilities())
            .native_context_scope(scope.clone())
            .native_replay(artifact)
            .build();
        let turn = block_on(assert_complete_drain(&model, Request::new(Vec::new())))
            .expect("drained replay turn")
            .turn;

        block_on(assert_replay_round_trip(
            &model,
            &scope,
            Request::new(vec![HistoryTurn::assistant(turn.clone())]),
        ))
        .expect("same-adapter replay");

        let foreign = NativeReplayArtifact::new(
            AdapterId::new("conformance.foreign"),
            scope.clone(),
            serde_json::json!({"ok": true}),
        )
        .expect("small foreign artifact");
        let mut foreign_turn = turn;
        foreign_turn.finish.native_replay = Some(foreign);
        block_on(assert_foreign_replay_is_reported(
            &model,
            &scope,
            Request::new(vec![HistoryTurn::assistant(foreign_turn)]),
        ))
        .expect("foreign artifact reconstruction");
    }

    #[test]
    fn replay_scope_mismatch_is_reported_before_reconstruction() {
        let expected_scope = mock_scope("scripted", "conformance.mock.resource");
        let artifact = NativeReplayArtifact::new(
            AdapterId::new("conformance.mock"),
            expected_scope.clone(),
            serde_json::json!({"ok": true}),
        )
        .expect("artifact");
        let model = MockLanguageModel::builder()
            .capabilities(optional_replay_capabilities())
            .native_context_scope(expected_scope.clone())
            .native_replay(artifact)
            .build();
        let mut turn = block_on(assert_complete_drain(&model, Request::new(Vec::new())))
            .expect("turn")
            .turn;
        turn.finish.native_replay = Some(
            NativeReplayArtifact::new(
                AdapterId::new("conformance.mock"),
                mock_scope("scripted", "other-resource"),
                serde_json::json!({"ok": true}),
            )
            .expect("foreign-scope artifact"),
        );
        block_on(assert_foreign_replay_scope_is_reported(
            &model,
            &expected_scope,
            Request::new(vec![HistoryTurn::assistant(turn)]),
        ))
        .expect("foreign scope reconstruction");
    }

    #[test]
    fn declarations_are_complete_and_model_names_do_not_change_behavior() {
        let capabilities = declared_capabilities();
        let first = MockLanguageModel::builder()
            .descriptor(
                LanguageModelDescriptor::new(
                    ModelIdentity::new(
                        ProviderId::new("conformance.mock"),
                        ModelId::new("known-looking-model"),
                    )
                    .expect("identity"),
                    AdapterId::new("conformance.mock"),
                    capabilities.clone(),
                )
                .expect("descriptor"),
            )
            .build();
        let second = MockLanguageModel::builder()
            .descriptor(
                LanguageModelDescriptor::new(
                    ModelIdentity::new(
                        ProviderId::new("conformance.mock"),
                        ModelId::new("future-unknown-model"),
                    )
                    .expect("identity"),
                    AdapterId::new("conformance.mock"),
                    capabilities,
                )
                .expect("descriptor"),
            )
            .build();
        assert_declaration_honesty(&first).expect("first declaration");
        assert_declaration_honesty(&second).expect("second declaration");
        block_on(assert_model_id_independence(&first, &second)).expect("model ID independence");

        let mut limit = InferenceOptions::new();
        limit.max_output_tokens = Some(4_096);
        block_on(assert_model_id_independence_with(
            &first,
            &second,
            [ModelIdIndependenceProbe::new(
                "caller:max-output-limit",
                Request::new(Vec::new()).with_inference(limit),
            )],
        ))
        .expect("caller probes");
    }

    #[test]
    fn model_id_independence_compares_normalized_stream_and_complete_output() {
        let capabilities = ModelCapabilities::conservative();
        let first = MockLanguageModel::builder()
            .descriptor(
                LanguageModelDescriptor::new(
                    ModelIdentity::new(
                        ProviderId::new("conformance.mock"),
                        ModelId::new("model-a"),
                    )
                    .expect("identity"),
                    AdapterId::new("conformance.mock"),
                    capabilities.clone(),
                )
                .expect("descriptor"),
            )
            .script(StreamFixture::valid_text("first"))
            .build();
        let second = MockLanguageModel::builder()
            .descriptor(
                LanguageModelDescriptor::new(
                    ModelIdentity::new(
                        ProviderId::new("conformance.mock"),
                        ModelId::new("model-b"),
                    )
                    .expect("identity"),
                    AdapterId::new("conformance.mock"),
                    capabilities,
                )
                .expect("descriptor"),
            )
            .script(StreamFixture::valid_text("second"))
            .build();
        block_on(assert_model_id_independence(&first, &second))
            .expect_err("different normalized output must fail");
    }

    #[test]
    fn replay_helpers_reject_missing_or_reordered_decisions() {
        let artifact = NativeReplayArtifact::new(
            AdapterId::new("conformance.mock"),
            mock_scope("scripted", "conformance.mock.resource"),
            serde_json::json!({"ok": true}),
        )
        .expect("artifact");
        let mut finish = Finish::new(Default::default(), FinishReason::Stop);
        finish.native_replay = Some(artifact);
        let request = Request::new(vec![HistoryTurn::assistant(CompletedTurn::new(
            oven_sdk::AssistantMessage::new(Vec::new()),
            finish,
        ))]);
        let missing = MockLanguageModel::builder()
            .capabilities(optional_replay_capabilities())
            .replay_outcome(ReplayOutcome::default())
            .build();
        let scope = mock_scope("scripted", "conformance.mock.resource");
        block_on(assert_replay_round_trip(&missing, &scope, request.clone()))
            .expect_err("missing replay decision must fail");

        let reordered = MockLanguageModel::builder()
            .capabilities(optional_replay_capabilities())
            .replay_outcome(ReplayOutcome {
                decisions: vec![
                    ReplayDecision {
                        history_index: 1,
                        disposition: ReplayDisposition::Replayed,
                    },
                    ReplayDecision {
                        history_index: 0,
                        disposition: ReplayDisposition::Replayed,
                    },
                ],
            })
            .build();
        let request = Request::new(vec![request.history[0].clone(), request.history[0].clone()]);
        block_on(assert_replay_round_trip(&reordered, &scope, request))
            .expect_err("reordered replay decisions must fail");
    }

    #[test]
    fn sse_encoder_supports_line_endings_comments_named_and_multiline_events() {
        let events = [
            sse::SseEvent::data("first")
                .with_comment("before data")
                .with_data_line("second"),
            sse::SseEvent::named("update", "named").with_comment("before named"),
            sse::SseEvent::comment("heartbeat"),
        ];
        let lf = ": before data\ndata: first\ndata: second\n\n: before named\nevent: update\ndata: named\n\n: heartbeat\n\n";
        assert_eq!(sse::encode_sse(&events, sse::LineEnding::Lf), lf.as_bytes());

        let crlf = ": before data\r\ndata: first\r\ndata: second\r\n\r\n: before named\r\nevent: update\r\ndata: named\r\n\r\n: heartbeat\r\n\r\n";
        assert_eq!(
            sse::encode_sse(&events, sse::LineEnding::Crlf),
            crlf.as_bytes()
        );
    }

    #[test]
    fn sse_encoder_handles_empty_and_usage_only_events() {
        assert!(sse::encode_sse(&[], sse::LineEnding::Lf).is_empty());
        assert!(sse::encode_sse(&[], sse::LineEnding::Crlf).is_empty());

        let usage_only = [sse::SseEvent::data(r#"{"usage":{"output_tokens":1}}"#)];
        let document = sse::encode_sse(&usage_only, sse::LineEnding::Lf);
        assert_eq!(document, b"data: {\"usage\":{\"output_tokens\":1}}\n\n");
        let chunks = sse::chunk_bytes(&document, &sse::ChunkPattern::Fixed(5));
        assert_eq!(chunks.concat(), document);
    }

    #[test]
    fn sse_chunking_patterns_preserve_original_bytes() {
        let document = "data: héllo\n\n".as_bytes();
        let one_byte = sse::chunk_bytes(document, &sse::ChunkPattern::OneByte);
        assert_eq!(one_byte.len(), document.len());
        assert!(one_byte.iter().all(|chunk| chunk.len() == 1));
        assert_eq!(one_byte[7], [0xc3]);
        assert_eq!(one_byte[8], [0xa9]);
        assert_eq!(one_byte.concat(), document);

        let fixed = sse::chunk_bytes(document, &sse::ChunkPattern::Fixed(3));
        assert_eq!(
            fixed,
            vec![
                b"dat".to_vec(),
                b"a: ".to_vec(),
                "hé".as_bytes().to_vec(),
                b"llo".to_vec(),
                b"\n\n".to_vec(),
            ]
        );
        assert_eq!(fixed.concat(), document);

        let explicit_leftover =
            sse::chunk_bytes(document, &sse::ChunkPattern::Explicit(vec![2, 3]));
        assert_eq!(
            explicit_leftover,
            vec![
                b"da".to_vec(),
                b"ta:".to_vec(),
                " héllo\n\n".as_bytes().to_vec()
            ]
        );
        assert_eq!(explicit_leftover.concat(), document);

        let explicit_clamped =
            sse::chunk_bytes(document, &sse::ChunkPattern::Explicit(vec![2, 99]));
        assert_eq!(
            explicit_clamped,
            vec![b"da".to_vec(), "ta: héllo\n\n".as_bytes().to_vec()]
        );
        assert_eq!(explicit_clamped.concat(), document);
    }

    #[test]
    fn sse_chunking_handles_empty_documents_and_explicit_zero_sizes() {
        for pattern in [
            sse::ChunkPattern::OneByte,
            sse::ChunkPattern::Fixed(3),
            sse::ChunkPattern::Explicit(vec![0, 2, 0]),
        ] {
            assert!(sse::chunk_bytes(&[], &pattern).is_empty());
        }

        let document = b"abcdef";
        let chunks = sse::chunk_bytes(document, &sse::ChunkPattern::Explicit(vec![0, 2, 0, 1, 0]));
        assert_eq!(chunks, vec![b"ab".to_vec(), b"c".to_vec(), b"def".to_vec()]);
        assert_eq!(chunks.concat(), document);
    }

    #[test]
    fn capability_probe_requests_cover_expected_capabilities() {
        let capabilities = capability_probe_requests()
            .into_iter()
            .map(|(capability, _, _)| capability)
            .collect::<Vec<_>>();
        assert_eq!(
            capabilities,
            vec![
                Capability::TOOL_CALLING,
                Capability::STRUCTURED_OUTPUT,
                Capability::TEMPERATURE,
                Capability::TOP_P,
                Capability::MAX_OUTPUT_TOKENS,
                Capability::REASONING,
            ]
        );
    }

    #[test]
    #[allow(clippy::result_large_err)] // ModelError deliberately owns typed diagnostics.
    fn taxonomy_and_malformed_payload_helpers_are_self_tested() {
        assert_error_taxonomy(&[ModelError::timeout("timed out")]).expect("taxonomy");
        assert_malformed_payload_returns_error(|| Err(ModelError::invalid_response("bad JSON")))
            .expect("malformed payload");
    }

    #[test]
    fn validate_for_consistency_helper_is_self_tested() {
        let model = MockLanguageModel::builder().build();
        assert_validate_for_consistency(&model, &Request::new(Vec::new()))
            .expect("validation consistency");
    }
}
