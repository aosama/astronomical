use std::collections::{BTreeMap, BTreeSet};

use astronomical_ipc_protocol::ChatToolDefinition;
use serde_json::{Map, Value};

use crate::strict_json::{DUPLICATE_JSON_FIELD_MARKER, DuplicateAwareJsonValue};

use super::LagunaOutputParserError;

const MAXIMUM_DECLARED_SCHEMA_BYTES: usize = 256 * 1024;
const MAXIMUM_ERROR_TEXT_CHARACTERS: usize = 64;

/// Flat scalar schema retained for one request-declared Poolside function.
#[derive(Debug)]
pub(super) struct LagunaDeclaredTool {
    parameter_types: BTreeMap<String, LagunaScalarType>,
    required_parameters: BTreeSet<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LagunaScalarType {
    String,
    Boolean,
    Integer,
    Number,
    Null,
    Array,
    Object,
    Untyped,
    Flexible,
    NullableString,
    NullableBoolean,
    NullableInteger,
    NullableNumber,
    NullableArray,
    NullableObject,
}

impl LagunaDeclaredTool {
    pub(super) fn from_definition(
        tool_definition: &ChatToolDefinition,
    ) -> Result<Self, LagunaOutputParserError> {
        let function_name = bounded_text(&tool_definition.name);
        if tool_definition.parameters_json.len() > MAXIMUM_DECLARED_SCHEMA_BYTES {
            return Err(LagunaOutputParserError::DeclaredToolSchemaTooLarge { function_name });
        }
        let schema = serde_json::from_slice::<DuplicateAwareJsonValue>(
            tool_definition.parameters_json.as_bytes(),
        )
        .map(|strict_value| strict_value.0)
        .map_err(|source| {
            if source.to_string().contains(DUPLICATE_JSON_FIELD_MARKER) {
                LagunaOutputParserError::DuplicateDeclaredToolSchemaField {
                    function_name: function_name.clone(),
                }
            } else {
                LagunaOutputParserError::InvalidDeclaredToolSchema {
                    function_name: function_name.clone(),
                    source,
                }
            }
        })?;
        let Value::Object(schema_fields) = schema else {
            return Err(LagunaOutputParserError::DeclaredToolSchemaMustBeObject { function_name });
        };
        let parameter_types = parse_parameter_types(&function_name, &schema_fields)?;
        let required_parameters =
            parse_required_parameters(&function_name, &schema_fields, &parameter_types)?;
        Ok(Self {
            parameter_types,
            required_parameters,
        })
    }

    pub(super) fn parse_arguments(
        &self,
        function_name: &str,
        raw_arguments: Vec<(String, String)>,
    ) -> Result<Map<String, Value>, LagunaOutputParserError> {
        let mut parsed_arguments = Map::new();
        for (argument_name, raw_argument_value) in raw_arguments {
            if parsed_arguments.contains_key(&argument_name) {
                return Err(LagunaOutputParserError::DuplicateToolArgument {
                    argument_name: bounded_text(&argument_name),
                });
            }
            // The tool client owns complete JSON Schema validation; retaining bounded extra
            // metadata avoids aborting useful calls while preventing undeclared fields from
            // gaining any server-side behavior.
            let parameter_type = self
                .parameter_types
                .get(&argument_name)
                .copied()
                .unwrap_or(LagunaScalarType::Untyped);
            let parsed_argument = parse_scalar_argument(&raw_argument_value, parameter_type)?;
            parsed_arguments.insert(argument_name, parsed_argument);
        }
        for required_parameter in &self.required_parameters {
            if !parsed_arguments.contains_key(required_parameter) {
                return Err(LagunaOutputParserError::MissingRequiredToolArgument {
                    function_name: bounded_text(function_name),
                    argument_name: bounded_text(required_parameter),
                });
            }
        }
        Ok(parsed_arguments)
    }
}

fn parse_parameter_types(
    function_name: &str,
    schema_fields: &Map<String, Value>,
) -> Result<BTreeMap<String, LagunaScalarType>, LagunaOutputParserError> {
    let Some(properties) = schema_fields.get("properties") else {
        return Ok(BTreeMap::new());
    };
    let Value::Object(property_schemas) = properties else {
        return Err(LagunaOutputParserError::InvalidDeclaredToolProperty {
            function_name: function_name.to_owned(),
            argument_name: "properties".to_owned(),
        });
    };
    let mut parameter_types = BTreeMap::new();
    for (argument_name, property_schema) in property_schemas {
        let Value::Object(property_fields) = property_schema else {
            return Err(LagunaOutputParserError::InvalidDeclaredToolProperty {
                function_name: function_name.to_owned(),
                argument_name: bounded_text(argument_name),
            });
        };
        let parameter_type = parse_scalar_type(property_fields).ok_or_else(|| {
            LagunaOutputParserError::InvalidDeclaredToolProperty {
                function_name: function_name.to_owned(),
                argument_name: bounded_text(argument_name),
            }
        })?;
        parameter_types.insert(argument_name.clone(), parameter_type);
    }
    Ok(parameter_types)
}

fn parse_scalar_type(property_fields: &Map<String, Value>) -> Option<LagunaScalarType> {
    match property_fields.get("type") {
        None if property_fields.contains_key("anyOf") => parse_any_of_type(property_fields),
        None => Some(LagunaScalarType::Untyped),
        Some(Value::String(type_name)) => scalar_type(type_name),
        Some(Value::Array(type_names)) if type_names.len() == 2 => {
            let non_null_type = type_names
                .iter()
                .filter_map(Value::as_str)
                .find(|type_name| *type_name != "null")?;
            if !type_names.iter().any(|type_name| type_name == "null") {
                return None;
            }
            nullable_scalar_type(non_null_type)
        }
        _ => None,
    }
}

fn parse_any_of_type(property_fields: &Map<String, Value>) -> Option<LagunaScalarType> {
    let Value::Array(any_of_schemas) = property_fields.get("anyOf")? else {
        return None;
    };
    if any_of_schemas.len() != 2 {
        return None;
    }
    let mut declared_types = Vec::with_capacity(2);
    let mut has_null_type = false;
    for schema in any_of_schemas {
        let Value::Object(schema_fields) = schema else {
            return None;
        };
        match schema_fields.get("type").and_then(Value::as_str) {
            Some("null") => has_null_type = true,
            Some(type_name) if scalar_type(type_name).is_some() => declared_types.push(type_name),
            _ => return None,
        }
    }
    if has_null_type && declared_types.len() == 1 {
        return nullable_scalar_type(declared_types[0]);
    }
    // OpenCode and Copilot use two-branch unions such as string-or-array.
    // Preserve them as dynamic strict JSON instead of rejecting the complete
    // tool catalog before the model has generated any output.
    (!has_null_type && declared_types.len() == 2).then_some(LagunaScalarType::Flexible)
}

fn scalar_type(type_name: &str) -> Option<LagunaScalarType> {
    match type_name {
        "string" => Some(LagunaScalarType::String),
        "boolean" => Some(LagunaScalarType::Boolean),
        "integer" => Some(LagunaScalarType::Integer),
        "number" => Some(LagunaScalarType::Number),
        "null" => Some(LagunaScalarType::Null),
        "array" => Some(LagunaScalarType::Array),
        "object" => Some(LagunaScalarType::Object),
        _ => None,
    }
}

fn nullable_scalar_type(type_name: &str) -> Option<LagunaScalarType> {
    match type_name {
        "string" => Some(LagunaScalarType::NullableString),
        "boolean" => Some(LagunaScalarType::NullableBoolean),
        "integer" => Some(LagunaScalarType::NullableInteger),
        "number" => Some(LagunaScalarType::NullableNumber),
        "array" => Some(LagunaScalarType::NullableArray),
        "object" => Some(LagunaScalarType::NullableObject),
        _ => None,
    }
}

fn parse_required_parameters(
    function_name: &str,
    schema_fields: &Map<String, Value>,
    parameter_types: &BTreeMap<String, LagunaScalarType>,
) -> Result<BTreeSet<String>, LagunaOutputParserError> {
    let Some(required_values) = schema_fields.get("required") else {
        return Ok(BTreeSet::new());
    };
    let Value::Array(required_values) = required_values else {
        return Err(LagunaOutputParserError::InvalidRequiredToolArguments {
            function_name: function_name.to_owned(),
        });
    };
    let mut required_parameters = BTreeSet::new();
    for required_value in required_values {
        let Some(required_name) = required_value.as_str() else {
            return Err(LagunaOutputParserError::InvalidRequiredToolArguments {
                function_name: function_name.to_owned(),
            });
        };
        if !parameter_types.contains_key(required_name)
            || !required_parameters.insert(required_name.to_owned())
        {
            return Err(LagunaOutputParserError::InvalidRequiredToolArguments {
                function_name: function_name.to_owned(),
            });
        }
    }
    Ok(required_parameters)
}

fn parse_scalar_argument(
    raw_argument_value: &str,
    parameter_type: LagunaScalarType,
) -> Result<Value, LagunaOutputParserError> {
    if matches!(
        parameter_type,
        LagunaScalarType::NullableString
            | LagunaScalarType::NullableBoolean
            | LagunaScalarType::NullableInteger
            | LagunaScalarType::NullableNumber
            | LagunaScalarType::NullableArray
            | LagunaScalarType::NullableObject
    ) && raw_argument_value == "null"
    {
        return Ok(Value::Null);
    }
    if matches!(
        parameter_type,
        LagunaScalarType::String | LagunaScalarType::NullableString
    ) {
        return Ok(Value::String(raw_argument_value.to_owned()));
    }
    if matches!(
        parameter_type,
        LagunaScalarType::Untyped | LagunaScalarType::Flexible
    ) {
        let parsed_value = serde_json::from_str::<DuplicateAwareJsonValue>(raw_argument_value)
            .map(|strict_value| strict_value.0);
        return match parsed_value {
            Ok(Value::Bool(boolean_value)) => Ok(Value::Bool(boolean_value)),
            Ok(Value::Number(number_value)) => Ok(Value::Number(number_value)),
            Ok(Value::Null) => Ok(Value::Null),
            Ok(structured_value @ (Value::Array(_) | Value::Object(_))) => Ok(structured_value),
            Ok(Value::String(string_value)) => Ok(Value::String(string_value)),
            Err(_) => Ok(Value::String(raw_argument_value.to_owned())),
        };
    }
    let parsed_value = serde_json::from_str::<DuplicateAwareJsonValue>(raw_argument_value)
        .map(|strict_value| strict_value.0)
        .map_err(|_| LagunaOutputParserError::InvalidToolArgumentValue)?;
    match (parameter_type, parsed_value) {
        (LagunaScalarType::Boolean | LagunaScalarType::NullableBoolean, Value::Bool(value)) => {
            Ok(Value::Bool(value))
        }
        (LagunaScalarType::Integer | LagunaScalarType::NullableInteger, Value::Number(value))
            if value.is_i64() || value.is_u64() =>
        {
            Ok(Value::Number(value))
        }
        (LagunaScalarType::Number | LagunaScalarType::NullableNumber, Value::Number(value)) => {
            Ok(Value::Number(value))
        }
        (LagunaScalarType::Array | LagunaScalarType::NullableArray, value @ Value::Array(_)) => {
            Ok(value)
        }
        (LagunaScalarType::Object | LagunaScalarType::NullableObject, value @ Value::Object(_)) => {
            Ok(value)
        }
        (LagunaScalarType::Null, Value::Null) => Ok(Value::Null),
        (_, Value::Array(_) | Value::Object(_)) => {
            Err(LagunaOutputParserError::StructuredToolArgumentUnsupported)
        }
        _ => Err(LagunaOutputParserError::InvalidToolArgumentValue),
    }
}

pub(super) fn bounded_text(unbounded_text: &str) -> String {
    unbounded_text
        .chars()
        .take(MAXIMUM_ERROR_TEXT_CHARACTERS)
        .collect()
}
