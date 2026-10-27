use oven_sdk::{
    AssistantMessage, AssistantPart, CancellationCapability, Capability, CompactionCapability,
    CompletedTurn, DeferredTools, DeferredToolsCapability, Finish, FinishReason, HistoryTurn,
    MediaCapabilities, Modalities, Modality, ModelCapabilities, ModelErrorKind, ModelLimits,
    ReplayCapability, ReplayDeclaration, ReplayPolicy, Request, ToolCallPart, ToolChoice,
    ToolContent, ToolDefinition, ToolLoading, ToolMessage, ToolResultPart, UserMessage,
};

fn capabilities(deferred_tools: DeferredToolsCapability) -> ModelCapabilities {
    ModelCapabilities {
        features: Capability::TOOL_CALLING,
        limits: ModelLimits::new(Some(16_384), Some(12_000), Some(4_096)),
        modalities: Modalities::new([Modality::text()], [Modality::text()]),
        media: MediaCapabilities::default(),
        cancellation: CancellationCapability::LocalOnly,
        compaction: CompactionCapability::Unsupported,
        deferred_tools,
        replay: ReplayDeclaration {
            policy: ReplayPolicy::Never,
            capability: ReplayCapability::Unsupported,
            reasoning: false,
        },
    }
}

fn tool(name: &str) -> ToolDefinition {
    ToolDefinition::new(
        name,
        format!("The {name} tool."),
        oven_sdk::JsonSchema::new(serde_json::json!({"type": "object"})).expect("schema"),
    )
}

/// One assistant turn calling `id` and the tool message answering it.
fn call_and_result(id: &str, content: ToolContent) -> [HistoryTurn; 2] {
    [
        HistoryTurn::assistant(CompletedTurn::new(
            AssistantMessage::new(vec![AssistantPart::ToolCall(ToolCallPart::new(
                id,
                "tool_search",
                serde_json::json!({}),
            ))]),
            Finish::new(Default::default(), FinishReason::ToolCalls),
        )),
        HistoryTurn::tool(ToolMessage::new(vec![ToolResultPart::new(id, content)])),
    ]
}

fn load(names: &[&str]) -> ToolContent {
    ToolContent::LoadTools {
        names: names.iter().map(|name| (*name).to_owned()).collect(),
        text: format!("Loaded tools: {}", names.join(", ")),
    }
}

fn request(tools: Vec<ToolDefinition>, loads: &[(&str, &[&str])]) -> Request {
    let mut history = vec![HistoryTurn::user(UserMessage::new(vec![
        oven_sdk::InputPart::Text(oven_sdk::TextPart::new("go")),
    ]))];
    for (id, names) in loads {
        history.extend(call_and_result(id, load(names)));
    }
    Request::new(history).with_tools(tools)
}

#[test]
fn eager_tools_serialize_as_before_and_deferred_ones_carry_loading() {
    let eager = serde_json::to_value(tool("read")).expect("serialize");
    assert!(eager.get("loading").is_none());
    let deferred = serde_json::to_value(tool("mcp_search").deferred()).expect("serialize");
    assert_eq!(deferred["loading"], "deferred");
    let back: ToolDefinition = serde_json::from_value(deferred).expect("deserialize");
    assert_eq!(back.loading, ToolLoading::Deferred);
    assert!(back.is_deferred());
}

#[test]
fn load_tools_content_round_trips() {
    let content = load(&["a", "b"]);
    let value = serde_json::to_value(&content).expect("serialize");
    assert_eq!(value["type"], "load_tools");
    assert_eq!(value["value"]["names"], serde_json::json!(["a", "b"]));
    let back: ToolContent = serde_json::from_value(value).expect("deserialize");
    assert_eq!(back, content);
}

#[test]
fn declarations_written_before_deferred_tools_existed_stay_unsupported() {
    let mut value =
        serde_json::to_value(capabilities(DeferredToolsCapability::Native)).expect("serialize");
    value
        .as_object_mut()
        .expect("object")
        .remove("deferred_tools");
    let back: ModelCapabilities = serde_json::from_value(value).expect("deserialize");
    assert_eq!(back.deferred_tools, DeferredToolsCapability::Unsupported);
}

#[test]
fn native_deferral_requires_tool_calling() {
    let mut declared = capabilities(DeferredToolsCapability::Native);
    declared.features = Capability::empty();
    assert!(declared.validate().is_err());
    declared.deferred_tools = DeferredToolsCapability::Unsupported;
    assert!(declared.validate().is_ok());
}

#[test]
fn a_request_needs_an_eager_tool_beside_deferred_ones() {
    let declared = capabilities(DeferredToolsCapability::Native);
    let only_deferred = request(vec![tool("a").deferred(), tool("b").deferred()], &[]);
    assert!(matches!(
        only_deferred.validate_for(&declared),
        Err(error) if error.kind() == ModelErrorKind::InvalidRequest
    ));
    let with_loader = request(vec![tool("tool_search"), tool("a").deferred()], &[]);
    assert!(with_loader.validate_for(&declared).is_ok());
}

#[test]
fn a_named_tool_choice_may_name_a_deferred_tool_only_once_loaded() {
    let declared = capabilities(DeferredToolsCapability::Native);
    let tools = || vec![tool("tool_search"), tool("a").deferred()];
    let unloaded = request(tools(), &[]).with_tool_choice(ToolChoice::Tool("a".into()));
    assert!(matches!(
        unloaded.validate_for(&declared),
        Err(error) if error.kind() == ModelErrorKind::InvalidRequest
    ));
    let loaded =
        request(tools(), &[("call_1", &["a"])]).with_tool_choice(ToolChoice::Tool("a".into()));
    assert!(loaded.validate_for(&declared).is_ok());
}

#[test]
fn the_first_result_naming_a_deferred_tool_loads_it() {
    let request = request(
        vec![
            tool("tool_search"),
            tool("a").deferred(),
            tool("b").deferred(),
        ],
        &[("call_1", &["a", "a"]), ("call_2", &["a", "b"])],
    );
    let plan = DeferredTools::of(&request);
    assert!(plan.is_active());
    assert_eq!(plan.newly_loaded("call_1"), ["a"]);
    assert_eq!(plan.newly_loaded("call_2"), ["b"]);
    assert!(plan.is_loaded("a") && plan.is_loaded("b"));
    assert!(plan.newly_loaded("call_unknown").is_empty());
    assert!(plan.warnings().is_empty());
}

#[test]
fn undeclared_names_warn_and_eager_names_are_dropped_quietly() {
    let request = request(
        vec![tool("tool_search"), tool("read"), tool("a").deferred()],
        &[("call_1", &["gone", "read", "a"])],
    );
    let plan = DeferredTools::of(&request);
    assert_eq!(plan.newly_loaded("call_1"), ["a"]);
    assert!(!plan.is_loaded("read"));
    assert_eq!(plan.warnings().len(), 1);
    assert!(plan.warnings()[0].contains("`gone`"));
}

#[test]
fn only_a_request_with_deferred_tools_warns_about_sending_them_up_front() {
    let eager_only = request(vec![tool("read")], &[]);
    assert_eq!(DeferredTools::of(&eager_only).upfront_warning("x"), None);
    let deferred = request(vec![tool("read"), tool("a").deferred()], &[]);
    assert!(
        DeferredTools::of(&deferred)
            .upfront_warning("google-gemini")
            .is_some_and(|warning| warning.contains("google-gemini"))
    );
}
