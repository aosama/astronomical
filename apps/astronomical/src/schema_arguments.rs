//! Arguments for `astronomical schema object`, mirroring the fm-style
//! modifier grammar: kind flags open a property and trailing modifiers attach
//! to the most recently opened one.

use std::ffi::OsString;

use crate::errors::UsageError;

/// JSON property kinds the schema verb can express.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SchemaPropertyKind {
    String,
    Integer,
    Double,
    Boolean,
}

impl SchemaPropertyKind {
    pub const fn kind_flag(self) -> &'static str {
        match self {
            Self::String => "--string",
            Self::Integer => "--int",
            Self::Double => "--double",
            Self::Boolean => "--boolean",
        }
    }

    pub const fn json_type_name(self) -> &'static str {
        match self {
            Self::String => "string",
            Self::Integer => "integer",
            Self::Double => "number",
            Self::Boolean => "boolean",
        }
    }
}

/// One property as written on the command line, before nesting is resolved.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SchemaPropertyInput {
    pub dotted_path: String,
    pub kind: SchemaPropertyKind,
    pub is_array: bool,
    pub is_optional: bool,
    pub description: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SchemaArguments {
    pub object_name: String,
    pub properties: Vec<SchemaPropertyInput>,
}

pub fn parse_schema_arguments(
    remaining_arguments: &[OsString],
) -> Result<SchemaArguments, UsageError> {
    let mut object_name: Option<String> = None;
    let mut properties: Vec<SchemaPropertyInput> = Vec::new();

    let mut argument_index = 0;
    while argument_index < remaining_arguments.len() {
        let argument_text = remaining_arguments[argument_index]
            .to_string_lossy()
            .into_owned();
        match argument_text.as_str() {
            "--name" => {
                if object_name.is_some() {
                    return Err(UsageError::RepeatedArgument("--name"));
                }
                let raw_name = required_flag_value(remaining_arguments, argument_index, "--name")?;
                if raw_name.is_empty() || raw_name.starts_with('-') {
                    return Err(UsageError::MissingValue("--name"));
                }
                object_name = Some(raw_name);
                argument_index += 2;
            }
            "--string" | "--int" | "--double" | "--boolean" => {
                let kind_flag: &'static str = match argument_text.as_str() {
                    "--string" => "--string",
                    "--int" => "--int",
                    "--double" => "--double",
                    _ => "--boolean",
                };
                let raw_dotted_path =
                    required_flag_value(remaining_arguments, argument_index, kind_flag)?;
                if raw_dotted_path.is_empty() || raw_dotted_path.starts_with('-') {
                    return Err(UsageError::MissingValue(kind_flag));
                }
                validate_dotted_property_path(&raw_dotted_path)?;
                reject_conflicting_property_path(&raw_dotted_path, &properties)?;
                properties.push(SchemaPropertyInput {
                    dotted_path: raw_dotted_path,
                    kind: property_kind_for_flag(kind_flag),
                    is_array: false,
                    is_optional: false,
                    description: None,
                });
                argument_index += 2;
            }
            "--array" | "--optional" => {
                let modified_property = properties.last_mut().ok_or_else(|| {
                    UsageError::SchemaModifierWithoutProperty(argument_text.clone())
                })?;
                if argument_text == "--array" {
                    modified_property.is_array = true;
                } else {
                    modified_property.is_optional = true;
                }
                argument_index += 1;
            }
            "--description" => {
                let description_text =
                    required_flag_value(remaining_arguments, argument_index, "--description")?;
                if description_text.is_empty() || description_text.starts_with('-') {
                    return Err(UsageError::MissingValue("--description"));
                }
                let modified_property =
                    properties
                        .last_mut()
                        .ok_or(UsageError::SchemaModifierWithoutProperty(
                            "--description".to_owned(),
                        ))?;
                modified_property.description = Some(description_text);
                argument_index += 2;
            }
            _ => return Err(UsageError::UnknownArgument(argument_text)),
        }
    }

    let object_name = object_name.ok_or(UsageError::SchemaNameRequired)?;
    if properties.is_empty() {
        return Err(UsageError::SchemaPropertyRequired);
    }
    Ok(SchemaArguments {
        object_name,
        properties,
    })
}

fn property_kind_for_flag(kind_flag: &str) -> SchemaPropertyKind {
    match kind_flag {
        "--string" => SchemaPropertyKind::String,
        "--int" => SchemaPropertyKind::Integer,
        "--double" => SchemaPropertyKind::Double,
        _ => SchemaPropertyKind::Boolean,
    }
}

fn required_flag_value(
    remaining_arguments: &[OsString],
    flag_index: usize,
    flag: &'static str,
) -> Result<String, UsageError> {
    remaining_arguments
        .get(flag_index + 1)
        .and_then(|value| value.to_str())
        .map(str::to_owned)
        .ok_or(UsageError::MissingValue(flag))
}

fn validate_dotted_property_path(dotted_path: &str) -> Result<(), UsageError> {
    for path_segment in dotted_path.split('.') {
        let is_usable_segment = !path_segment.is_empty()
            && path_segment.chars().all(|character| {
                character.is_ascii_alphanumeric() || character == '_' || character == '-'
            });
        if !is_usable_segment {
            return Err(UsageError::SchemaInvalidPropertyPath(
                dotted_path.to_owned(),
            ));
        }
    }
    Ok(())
}

fn reject_conflicting_property_path(
    candidate_path: &str,
    existing_properties: &[SchemaPropertyInput],
) -> Result<(), UsageError> {
    for existing_property in existing_properties {
        let existing_path = existing_property.dotted_path.as_str();
        let nested_pair = existing_path.starts_with(&format!("{candidate_path}."))
            || candidate_path.starts_with(&format!("{existing_path}."));
        if nested_pair || existing_path == candidate_path {
            return Err(UsageError::SchemaDuplicateProperty(
                candidate_path.to_owned(),
            ));
        }
    }
    Ok(())
}
