//! Azure/OpenAI strict JSON Schema validation.

use std::collections::BTreeSet;

use oven_sdk::{ErrorStage, JsonValue, ModelError, Request, ToolChoice};

const MAX_OBJECT_DEPTH: usize = 5;
const MAX_PROPERTIES: usize = 100;

pub(crate) fn validate_request_schemas(
    request: &Request,
    parallel_tool_calls: Option<bool>,
) -> Result<(), ModelError> {
    if let oven_sdk::ResponseFormat::Json {
        schema: Some(schema),
    } = &request.response_format
    {
        validate_strict_schema(schema.as_value(), "structured output")?;
    }

    let mut has_strict_tool = false;
    if !matches!(request.tool_choice, ToolChoice::None) {
        for tool in &request.tools {
            let strict = strict_tool(tool)?;
            if strict {
                has_strict_tool = true;
                validate_strict_schema(tool.input_schema.as_value(), "strict tool")?;
            }
        }
    }
    if has_strict_tool && parallel_tool_calls != Some(false) {
        return Err(schema_error(
            "strict tools require parallel_tool_calls to be explicitly false",
        ));
    }
    Ok(())
}

pub(crate) fn strict_tool(tool: &oven_sdk::ToolDefinition) -> Result<bool, ModelError> {
    let Some(options) = tool.provider_options.get("azure_openai") else {
        return Ok(false);
    };
    let Some(object) = options.as_object() else {
        return Err(schema_error("Azure tool options must be an object"));
    };
    match object.get("strict") {
        None => Ok(false),
        Some(JsonValue::Bool(value)) => Ok(*value),
        Some(_) => Err(schema_error("Azure tool strict must be a boolean")),
    }
}

fn validate_strict_schema(schema: &JsonValue, context: &str) -> Result<(), ModelError> {
    let root = schema
        .as_object()
        .ok_or_else(|| schema_error(&format!("Azure {context} schema root must be an object")))?;
    if root.get("type").and_then(JsonValue::as_str) != Some("object")
        || root.contains_key("anyOf")
        || root.contains_key("$ref")
    {
        return Err(schema_error(&format!(
            "Azure {context} schema root must use type object without a union or reference"
        )));
    }
    let mut property_count = 0;
    validate_node(schema, schema, 0, &mut property_count, context)?;
    validate_reference_paths(schema, schema, 0, &mut Vec::new(), context)?;
    if let Some(definitions) = root.get("$defs").and_then(JsonValue::as_object) {
        for definition in definitions.values() {
            validate_reference_paths(schema, definition, 0, &mut Vec::new(), context)?;
        }
    }
    Ok(())
}

fn validate_node(
    root: &JsonValue,
    schema: &JsonValue,
    object_depth: usize,
    property_count: &mut usize,
    context: &str,
) -> Result<(), ModelError> {
    let object = schema.as_object().ok_or_else(|| {
        schema_error(&format!(
            "Azure {context} schema contains a non-object node"
        ))
    })?;
    const ALLOWED: &[&str] = &[
        "$defs",
        "$ref",
        "type",
        "properties",
        "required",
        "additionalProperties",
        "items",
        "enum",
        "anyOf",
        "description",
        "title",
    ];
    if object.keys().any(|key| !ALLOWED.contains(&key.as_str())) {
        return Err(schema_error(&format!(
            "Azure {context} schema uses an unsupported keyword"
        )));
    }
    for key in ["description", "title"] {
        if object.get(key).is_some_and(|value| !value.is_string()) {
            return Err(schema_error(&format!(
                "Azure {context} schema requires {key} to be a string"
            )));
        }
    }
    if let Some(reference) = object.get("$ref") {
        let reference = reference.as_str().ok_or_else(|| {
            schema_error(&format!(
                "Azure {context} schema requires $ref to be a string"
            ))
        })?;
        if object
            .keys()
            .any(|key| key != "$ref" && key != "description" && key != "title")
        {
            return Err(schema_error(&format!(
                "Azure {context} schema contains an invalid local reference"
            )));
        }
        resolve_reference(root, reference, context)?;
        return Ok(());
    }
    if let Some(value) = object.get("type") {
        validate_type(value, context)?;
    }
    if let Some(value) = object.get("enum")
        && value.as_array().is_none_or(|values| values.is_empty())
    {
        return Err(schema_error(&format!(
            "Azure {context} schema requires enum to be a non-empty array"
        )));
    }
    if let Some(definitions) = object.get("$defs") {
        let definitions = definitions.as_object().ok_or_else(|| {
            schema_error(&format!(
                "Azure {context} schema requires $defs to be an object"
            ))
        })?;
        for definition in definitions.values() {
            validate_node(root, definition, object_depth, property_count, context)?;
        }
    }
    if let Some(branches) = object.get("anyOf") {
        let branches = branches
            .as_array()
            .filter(|values| !values.is_empty())
            .ok_or_else(|| {
                schema_error(&format!("Azure {context} schema requires non-empty anyOf"))
            })?;
        for branch in branches {
            validate_node(root, branch, object_depth, property_count, context)?;
        }
    }
    let is_object =
        type_contains(object.get("type"), "object") || object.contains_key("properties");
    let next_depth = if is_object {
        object_depth + 1
    } else {
        object_depth
    };
    if next_depth > MAX_OBJECT_DEPTH {
        return Err(schema_error(&format!(
            "Azure {context} schema exceeds {MAX_OBJECT_DEPTH} object levels"
        )));
    }
    if is_object {
        let properties = object
            .get("properties")
            .and_then(JsonValue::as_object)
            .ok_or_else(|| {
                schema_error(&format!(
                    "Azure {context} object schema requires properties"
                ))
            })?;
        *property_count = property_count.saturating_add(properties.len());
        if *property_count > MAX_PROPERTIES {
            return Err(schema_error(&format!(
                "Azure {context} schema exceeds {MAX_PROPERTIES} properties"
            )));
        }
        if object.get("additionalProperties") != Some(&JsonValue::Bool(false)) {
            return Err(schema_error(&format!(
                "Azure {context} object schema requires additionalProperties:false"
            )));
        }
        let required = unique_strings(object.get("required")).ok_or_else(|| {
            schema_error(&format!(
                "Azure {context} object schema requires a unique-string required array"
            ))
        })?;
        let property_names = properties
            .keys()
            .map(String::as_str)
            .collect::<BTreeSet<_>>();
        if required != property_names {
            return Err(schema_error(&format!(
                "Azure {context} object schema must require every and only declared property"
            )));
        }
        for property in properties.values() {
            validate_node(root, property, next_depth, property_count, context)?;
        }
    } else if object.contains_key("required") || object.contains_key("additionalProperties") {
        return Err(schema_error(&format!(
            "Azure {context} schema uses object keywords without object type"
        )));
    }
    if type_contains(object.get("type"), "array") {
        let items = object
            .get("items")
            .ok_or_else(|| schema_error(&format!("Azure {context} array schema requires items")))?;
        validate_node(root, items, next_depth, property_count, context)?;
    } else if object.contains_key("items") {
        return Err(schema_error(&format!(
            "Azure {context} schema uses items without array type"
        )));
    }
    Ok(())
}

fn validate_reference_paths(
    root: &JsonValue,
    schema: &JsonValue,
    object_depth: usize,
    stack: &mut Vec<String>,
    context: &str,
) -> Result<(), ModelError> {
    let object = schema.as_object().ok_or_else(|| {
        schema_error(&format!(
            "Azure {context} schema contains a non-object node"
        ))
    })?;
    if let Some(reference) = object.get("$ref").and_then(JsonValue::as_str) {
        if stack.iter().any(|active| active == reference) {
            return Err(schema_error(&format!(
                "Azure {context} schema contains a reference cycle"
            )));
        }
        let target = resolve_reference(root, reference, context)?;
        stack.push(reference.to_owned());
        let result = validate_reference_paths(root, target, object_depth, stack, context);
        stack.pop();
        return result;
    }
    let is_object =
        type_contains(object.get("type"), "object") || object.contains_key("properties");
    let next_depth = if is_object {
        object_depth + 1
    } else {
        object_depth
    };
    if next_depth > MAX_OBJECT_DEPTH {
        return Err(schema_error(&format!(
            "Azure {context} schema exceeds {MAX_OBJECT_DEPTH} object levels through references"
        )));
    }
    if let Some(properties) = object.get("properties").and_then(JsonValue::as_object) {
        for property in properties.values() {
            validate_reference_paths(root, property, next_depth, stack, context)?;
        }
    }
    if let Some(items) = object.get("items") {
        validate_reference_paths(root, items, next_depth, stack, context)?;
    }
    if let Some(branches) = object.get("anyOf").and_then(JsonValue::as_array) {
        for branch in branches {
            validate_reference_paths(root, branch, next_depth, stack, context)?;
        }
    }
    Ok(())
}

fn resolve_reference<'a>(
    root: &'a JsonValue,
    reference: &str,
    context: &str,
) -> Result<&'a JsonValue, ModelError> {
    let name = reference
        .strip_prefix("#/$defs/")
        .filter(|name| !name.is_empty() && !name.contains('/') && !name.contains('~'));
    let target = name
        .and_then(|name| root.get("$defs").and_then(JsonValue::as_object)?.get(name))
        .filter(|target| target.is_object());
    target.ok_or_else(|| {
        schema_error(&format!(
            "Azure {context} schema contains an unresolved or escaping local reference"
        ))
    })
}

fn validate_type(value: &JsonValue, context: &str) -> Result<(), ModelError> {
    let valid = |value: &str| {
        matches!(
            value,
            "null" | "boolean" | "object" | "array" | "number" | "integer" | "string"
        )
    };
    let accepted = value.as_str().is_some_and(valid)
        || value.as_array().is_some_and(|values| {
            let types = values
                .iter()
                .filter_map(JsonValue::as_str)
                .collect::<BTreeSet<_>>();
            !types.is_empty()
                && types.len() == values.len()
                && types.iter().all(|value| valid(value))
        });
    if accepted {
        Ok(())
    } else {
        Err(schema_error(&format!(
            "Azure {context} schema has an invalid type"
        )))
    }
}

fn type_contains(value: Option<&JsonValue>, expected: &str) -> bool {
    value.is_some_and(|value| {
        value.as_str() == Some(expected)
            || value
                .as_array()
                .is_some_and(|values| values.iter().any(|value| value.as_str() == Some(expected)))
    })
}

fn unique_strings(value: Option<&JsonValue>) -> Option<BTreeSet<&str>> {
    let values = value?.as_array()?;
    let strings = values
        .iter()
        .filter_map(JsonValue::as_str)
        .collect::<BTreeSet<_>>();
    (strings.len() == values.len()).then_some(strings)
}

fn schema_error(message: &str) -> ModelError {
    ModelError::invalid_request(message).with_stage(ErrorStage::RequestEncoding)
}
