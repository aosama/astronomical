//! Builds strict JSON object schemas in-process for `astronomical schema
//! object`. The document shape follows the fm convention: `additionalProperties:
//! false` and a `required` list at every object level so downstream tools can
//! never silently accept undeclared keys.

use std::io;

use serde_json::{Value, json};

pub fn build_schema_document(schema_arguments: &SchemaArguments) -> Value {
    let mut root_node = SchemaObjectNode::default();
    for schema_property in &schema_arguments.properties {
        insert_property(&mut root_node, schema_property, 0);
    }
    let mut root_document = root_node.into_property_value();
    root_document["title"] = json!(schema_arguments.object_name);
    root_document
}

pub fn run_schema(
    schema_arguments: &SchemaArguments,
    rendered_output: &mut dyn io::Write,
) -> io::Result<()> {
    let schema_document = build_schema_document(schema_arguments);
    let rendered_text =
        serde_json::to_string_pretty(&schema_document).expect("schema documents serialize");
    writeln!(rendered_output, "{rendered_text}")
}

#[derive(Default)]
struct SchemaObjectNode {
    child_nodes: Vec<(String, SchemaNode)>,
}

enum SchemaNode {
    Leaf(SchemaPropertyInput),
    Branch(SchemaObjectNode),
}

impl SchemaObjectNode {
    fn child_entry_mut(&mut self, segment_name: &str) -> Option<&mut SchemaNode> {
        self.child_nodes
            .iter_mut()
            .find(|(existing_name, _)| existing_name == segment_name)
            .map(|(_, child_node)| child_node)
    }

    fn into_property_value(self) -> Value {
        let mut properties_document = serde_json::Map::new();
        let mut required_names = Vec::new();
        for (segment_name, child_node) in self.child_nodes {
            if !child_node.is_optional() {
                required_names.push(json!(segment_name));
            }
            properties_document.insert(segment_name, child_node.into_value());
        }
        json!({
            "type": "object",
            "properties": properties_document,
            "required": required_names,
            "additionalProperties": false,
        })
    }
}

impl SchemaNode {
    fn is_optional(&self) -> bool {
        match self {
            Self::Leaf(property) => property.is_optional,
            Self::Branch(_) => false,
        }
    }

    fn into_value(self) -> Value {
        match self {
            Self::Leaf(property) => leaf_property_value(&property),
            Self::Branch(branch) => branch.into_property_value(),
        }
    }
}

fn insert_property(
    object_node: &mut SchemaObjectNode,
    schema_property: &SchemaPropertyInput,
    segment_index: usize,
) {
    let path_segments = schema_property.dotted_path.split('.').collect::<Vec<_>>();
    let segment_name = path_segments[segment_index];
    let is_final_segment = segment_index == path_segments.len() - 1;
    if is_final_segment {
        if let Some(existing_child_node) = object_node.child_entry_mut(segment_name) {
            // Command-line parsing rejects duplicate and conflicting paths, so
            // this only matters for direct library callers: a later leaf wins
            // its position, and a leaf-over-branch clash keeps the branch.
            if matches!(existing_child_node, SchemaNode::Leaf(_)) {
                *existing_child_node = SchemaNode::Leaf(schema_property.clone());
            }
            return;
        }
        object_node.child_nodes.push((
            segment_name.to_owned(),
            SchemaNode::Leaf(schema_property.clone()),
        ));
        return;
    }
    if !object_node
        .child_nodes
        .iter()
        .any(|(name, _)| name == segment_name)
    {
        object_node.child_nodes.push((
            segment_name.to_owned(),
            SchemaNode::Branch(SchemaObjectNode::default()),
        ));
    }
    let SchemaNode::Branch(child_object_node) = object_node
        .child_entry_mut(segment_name)
        .expect("branch was just inserted")
    else {
        return;
    };
    insert_property(child_object_node, schema_property, segment_index + 1);
}

fn leaf_property_value(schema_property: &SchemaPropertyInput) -> Value {
    let type_document = json!({ "type": schema_property.kind.json_type_name() });
    if schema_property.is_array {
        return json!({ "type": "array", "items": type_document });
    }
    match &schema_property.description {
        Some(description_text) => json!({
            "type": schema_property.kind.json_type_name(),
            "description": description_text,
        }),
        None => type_document,
    }
}

// Arguments for `astronomical schema object`, mirroring the fm-style
// modifier grammar: kind flags open a property and trailing modifiers attach
// to the most recently opened one.

use std::ffi::OsString;

use crate::arguments::UsageError;

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

#[cfg(test)]
mod schema_tests;

#[cfg(test)]
mod schema_arguments_tests;
