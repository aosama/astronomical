use std::collections::BTreeMap;

use astronomical_ipc_protocol::ChatToolDefinition;
use serde_json::{Map, Number, Value};

use super::output_parser_error::Qwen3_5OutputParserError;

const MAX_TOOL_SCHEMA_DEPTH: usize = 8;

#[derive(Debug)]
pub(super) struct DeclaredTool {
    parameter_schemas: BTreeMap<String, DeclaredParameterSchema>,
}

#[derive(Debug)]
struct DeclaredParameterSchema {
    parameter_type: Option<String>,
    is_nullable: bool,
}

impl DeclaredTool {
    pub(super) fn from_definition(
        tool_definition: &ChatToolDefinition,
    ) -> Result<Self, Qwen3_5OutputParserError> {
        let schema =
            serde_json::from_str::<Value>(&tool_definition.parameters_json).map_err(|source| {
                Qwen3_5OutputParserError::InvalidDeclaredToolSchema {
                    function_name: tool_definition.name.clone(),
                    source,
                }
            })?;
        let Value::Object(schema_fields) = schema else {
            return Err(Qwen3_5OutputParserError::DeclaredToolSchemaMustBeObject {
                function_name: tool_definition.name.clone(),
            });
        };
        let parameter_schemas = parse_parameter_schemas(&tool_definition.name, &schema_fields, 0)?;
        Ok(Self { parameter_schemas })
    }
}

pub(super) fn parse_tool_parameters(
    parameter_content: &str,
    declared_tool: &DeclaredTool,
) -> Result<Map<String, Value>, Qwen3_5OutputParserError> {
    let mut remaining_parameters = parameter_content.trim();
    let mut parsed_arguments = Map::new();
    while !remaining_parameters.is_empty() {
        let parameter_with_name = remaining_parameters
            .strip_prefix("<parameter=")
            .ok_or(Qwen3_5OutputParserError::MalformedToolParameter)?;
        let parameter_name_end = parameter_with_name
            .find('>')
            .ok_or(Qwen3_5OutputParserError::ToolParameterMissingNameEnd)?;
        let parameter_name = &parameter_with_name[..parameter_name_end];
        let parameter_with_value = &parameter_with_name[parameter_name_end + 1..];
        let parameter_end = parameter_with_value
            .find("</parameter>")
            .ok_or(Qwen3_5OutputParserError::ToolParameterMissingEnd)?;
        let parameter_value = trim_one_boundary_newline(&parameter_with_value[..parameter_end]);
        let parsed_parameter_value = match declared_tool.parameter_schemas.get(parameter_name) {
            Some(parameter_schema) => parse_parameter_value(parameter_value, parameter_schema),
            None => parse_untyped_parameter_value(parameter_value),
        };
        parsed_arguments.insert(parameter_name.to_owned(), parsed_parameter_value);
        remaining_parameters = parameter_with_value[parameter_end + "</parameter>".len()..].trim();
    }
    Ok(parsed_arguments)
}

fn parse_parameter_schemas(
    function_name: &str,
    schema_fields: &Map<String, Value>,
    parent_schema_depth: usize,
) -> Result<BTreeMap<String, DeclaredParameterSchema>, Qwen3_5OutputParserError> {
    let Some(Value::Object(property_schemas)) = schema_fields.get("properties") else {
        return Ok(BTreeMap::new());
    };
    let mut parameter_schemas = BTreeMap::new();
    for (parameter_name, property_schema) in property_schemas {
        let parameter_schema_depth = parent_schema_depth.checked_add(1).ok_or_else(|| {
            Qwen3_5OutputParserError::ToolParameterSchemaTooDeep {
                function_name: function_name.to_owned(),
                parameter_name: parameter_name.clone(),
                maximum_schema_depth: MAX_TOOL_SCHEMA_DEPTH,
            }
        })?;
        if parameter_schema_depth > MAX_TOOL_SCHEMA_DEPTH {
            return Err(Qwen3_5OutputParserError::ToolParameterSchemaTooDeep {
                function_name: function_name.to_owned(),
                parameter_name: parameter_name.clone(),
                maximum_schema_depth: MAX_TOOL_SCHEMA_DEPTH,
            });
        }
        let Value::Object(property_schema_fields) = property_schema else {
            return Err(Qwen3_5OutputParserError::InvalidToolPropertySchema {
                function_name: function_name.to_owned(),
                parameter_name: parameter_name.clone(),
            });
        };
        parameter_schemas.insert(
            parameter_name.clone(),
            parse_declared_parameter_schema(
                function_name,
                parameter_name,
                property_schema_fields,
                parameter_schema_depth,
            )?,
        );
    }
    Ok(parameter_schemas)
}

fn parse_declared_parameter_schema(
    function_name: &str,
    parameter_name: &str,
    property_schema_fields: &Map<String, Value>,
    schema_depth: usize,
) -> Result<DeclaredParameterSchema, Qwen3_5OutputParserError> {
    if schema_depth > MAX_TOOL_SCHEMA_DEPTH {
        return Err(Qwen3_5OutputParserError::ToolParameterSchemaTooDeep {
            function_name: function_name.to_owned(),
            parameter_name: parameter_name.to_owned(),
            maximum_schema_depth: MAX_TOOL_SCHEMA_DEPTH,
        });
    }
    let (parameter_type, is_nullable) = match property_schema_fields.get("type") {
        Some(Value::String(parameter_type)) => (Some(parameter_type.clone()), false),
        Some(Value::Array(parameter_types)) => {
            let mut non_null_parameter_types =
                parameter_types
                    .iter()
                    .filter_map(|parameter_type| match parameter_type {
                        Value::String(parameter_type) if parameter_type != "null" => {
                            Some(parameter_type.as_str())
                        }
                        _ => None,
                    });
            let Some(non_null_parameter_type) = non_null_parameter_types.next() else {
                return Err(
                    Qwen3_5OutputParserError::InvalidToolParameterTypeDeclaration {
                        function_name: function_name.to_owned(),
                        parameter_name: parameter_name.to_owned(),
                    },
                );
            };
            let has_exact_nullable_pair = non_null_parameter_types.next().is_none()
                && parameter_types.len() == 2
                && parameter_types
                    .iter()
                    .any(|parameter_type| parameter_type == "null");
            if !has_exact_nullable_pair {
                return Err(
                    Qwen3_5OutputParserError::InvalidToolParameterTypeDeclaration {
                        function_name: function_name.to_owned(),
                        parameter_name: parameter_name.to_owned(),
                    },
                );
            }
            (Some(non_null_parameter_type.to_owned()), true)
        }
        None => match parse_nullable_any_of_type(property_schema_fields) {
            Some((parameter_type, is_nullable)) => (Some(parameter_type), is_nullable),
            None if has_supported_flexible_any_of(property_schema_fields) => (None, false),
            None if !property_schema_fields.contains_key("anyOf") => (None, false),
            None => {
                return Err(Qwen3_5OutputParserError::MissingToolParameterType {
                    function_name: function_name.to_owned(),
                    parameter_name: parameter_name.to_owned(),
                });
            }
        },
        Some(_) => {
            return Err(
                Qwen3_5OutputParserError::InvalidToolParameterTypeDeclaration {
                    function_name: function_name.to_owned(),
                    parameter_name: parameter_name.to_owned(),
                },
            );
        }
    };
    Ok(DeclaredParameterSchema {
        parameter_type,
        is_nullable,
    })
}

fn has_supported_flexible_any_of(property_schema_fields: &Map<String, Value>) -> bool {
    let Some(Value::Array(any_of_schemas)) = property_schema_fields.get("anyOf") else {
        return false;
    };
    if any_of_schemas.len() != 2 {
        return false;
    }
    any_of_schemas.iter().all(|any_of_schema| {
        let Value::Object(schema_fields) = any_of_schema else {
            return false;
        };
        matches!(
            schema_fields.get("type"),
            Some(Value::String(parameter_type))
                if matches!(
                    parameter_type.as_str(),
                    "string" | "boolean" | "integer" | "number" | "array" | "object"
                )
        )
    })
}

fn parse_nullable_any_of_type(
    property_schema_fields: &Map<String, Value>,
) -> Option<(String, bool)> {
    let Value::Array(any_of_schemas) = property_schema_fields.get("anyOf")? else {
        return None;
    };
    if any_of_schemas.len() != 2 {
        return None;
    }
    let mut non_null_type = None;
    let mut has_null_type = false;
    for any_of_schema in any_of_schemas {
        let Value::Object(schema_fields) = any_of_schema else {
            return None;
        };
        match schema_fields.get("type") {
            Some(Value::String(parameter_type)) if parameter_type == "null" => {
                if schema_fields.len() != 1 {
                    return None;
                }
                has_null_type = true;
            }
            Some(Value::String(parameter_type)) if non_null_type.is_none() => {
                if schema_fields.keys().any(|schema_keyword| {
                    !matches!(
                        schema_keyword.as_str(),
                        "type"
                            | "description"
                            | "title"
                            | "default"
                            | "minimum"
                            | "maximum"
                            | "exclusiveMinimum"
                            | "exclusiveMaximum"
                            | "minLength"
                            | "maxLength"
                            | "pattern"
                            | "format"
                            | "minItems"
                            | "maxItems"
                            | "minProperties"
                            | "maxProperties"
                            | "uniqueItems"
                    )
                }) {
                    return None;
                }
                non_null_type = Some(parameter_type.clone());
            }
            _ => return None,
        }
    }
    has_null_type.then_some((non_null_type?, true))
}

fn parse_parameter_value(
    parameter_value: &str,
    parameter_schema: &DeclaredParameterSchema,
) -> Value {
    if parameter_schema.is_nullable && parameter_value == "null" {
        return Value::Null;
    }
    let Some(parameter_type) = parameter_schema.parameter_type.as_deref() else {
        return parse_untyped_parameter_value(parameter_value);
    };
    match parameter_type {
        "string" => Value::String(parameter_value.to_owned()),
        "boolean" => match parameter_value {
            "true" => Value::Bool(true),
            "false" => Value::Bool(false),
            _ => Value::String(parameter_value.to_owned()),
        },
        "integer" => parameter_value
            .parse::<i64>()
            .map(|integer_value| Value::Number(Number::from(integer_value)))
            .unwrap_or_else(|_| Value::String(parameter_value.to_owned())),
        "number" => parameter_value
            .parse::<f64>()
            .ok()
            .and_then(Number::from_f64)
            .map(Value::Number)
            .unwrap_or_else(|| Value::String(parameter_value.to_owned())),
        "array" | "object" => match serde_json::from_str::<Value>(parameter_value) {
            Ok(json_value)
                if matches!(
                    (&json_value, parameter_type),
                    (Value::Array(_), "array") | (Value::Object(_), "object")
                ) =>
            {
                json_value
            }
            _ => Value::String(parameter_value.to_owned()),
        },
        _ => Value::String(parameter_value.to_owned()),
    }
}

fn parse_untyped_parameter_value(parameter_value: &str) -> Value {
    match serde_json::from_str::<Value>(parameter_value) {
        Ok(json_value) if !json_value.is_string() => json_value,
        _ => Value::String(parameter_value.to_owned()),
    }
}

fn trim_one_boundary_newline(parameter_value: &str) -> &str {
    let parameter_value = parameter_value
        .strip_prefix('\n')
        .unwrap_or(parameter_value);
    parameter_value
        .strip_suffix('\n')
        .unwrap_or(parameter_value)
}
