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
    /// The kind a property flag declares, or `None` for any other argument.
    fn from_kind_flag(argument_text: &str) -> Option<Self> {
        match argument_text {
            "--string" => Some(Self::String),
            "--int" => Some(Self::Integer),
            "--double" => Some(Self::Double),
            "--boolean" => Some(Self::Boolean),
            _ => None,
        }
    }

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

/// Parses `schema object` arguments. Modifiers (`--array`, `--optional`,
/// `--description`) apply to the property declared just before them, which
/// is why the arguments are walked in order instead of collected by flag.
pub fn parse_schema_arguments(
    remaining_arguments: &[OsString],
) -> Result<SchemaArguments, UsageError> {
    let mut schema_accumulator = SchemaArgumentAccumulator::default();
    let mut argument_index = 0;
    while argument_index < remaining_arguments.len() {
        let argument_text = remaining_arguments[argument_index]
            .to_string_lossy()
            .into_owned();
        argument_index += schema_accumulator.apply_argument(
            &argument_text,
            remaining_arguments,
            argument_index,
        )?;
    }
    schema_accumulator.finish()
}

/// What the `schema object` arguments declared so far.
#[derive(Default)]
struct SchemaArgumentAccumulator {
    object_name: Option<String>,
    properties: Vec<SchemaPropertyInput>,
}

impl SchemaArgumentAccumulator {
    /// Applies the argument at `argument_index` and answers how many
    /// arguments it consumed: two for a flag with a value, one for a switch.
    fn apply_argument(
        &mut self,
        argument_text: &str,
        remaining_arguments: &[OsString],
        argument_index: usize,
    ) -> Result<usize, UsageError> {
        if let Some(property_kind) = SchemaPropertyKind::from_kind_flag(argument_text) {
            let dotted_path = required_flag_text(
                remaining_arguments,
                argument_index,
                property_kind.kind_flag(),
            )?;
            self.declare_property(property_kind, dotted_path)?;
            return Ok(2);
        }
        match argument_text {
            "--name" => {
                if self.object_name.is_some() {
                    return Err(UsageError::RepeatedArgument("--name"));
                }
                self.object_name = Some(required_flag_text(
                    remaining_arguments,
                    argument_index,
                    "--name",
                )?);
                Ok(2)
            }
            "--array" => {
                self.last_property(argument_text)?.is_array = true;
                Ok(1)
            }
            "--optional" => {
                self.last_property(argument_text)?.is_optional = true;
                Ok(1)
            }
            "--description" => {
                let description_text =
                    required_flag_text(remaining_arguments, argument_index, "--description")?;
                self.last_property(argument_text)?.description = Some(description_text);
                Ok(2)
            }
            _ => Err(UsageError::UnknownArgument(argument_text.to_owned())),
        }
    }

    fn declare_property(
        &mut self,
        property_kind: SchemaPropertyKind,
        dotted_path: String,
    ) -> Result<(), UsageError> {
        validate_dotted_property_path(&dotted_path)?;
        reject_conflicting_property_path(&dotted_path, &self.properties)?;
        self.properties.push(SchemaPropertyInput {
            dotted_path,
            kind: property_kind,
            is_array: false,
            is_optional: false,
            description: None,
        });
        Ok(())
    }

    fn last_property(
        &mut self,
        modifier_flag: &str,
    ) -> Result<&mut SchemaPropertyInput, UsageError> {
        self.properties
            .last_mut()
            .ok_or_else(|| UsageError::SchemaModifierWithoutProperty(modifier_flag.to_owned()))
    }

    fn finish(self) -> Result<SchemaArguments, UsageError> {
        let object_name = self.object_name.ok_or(UsageError::SchemaNameRequired)?;
        if self.properties.is_empty() {
            return Err(UsageError::SchemaPropertyRequired);
        }
        Ok(SchemaArguments {
            object_name,
            properties: self.properties,
        })
    }
}

/// A flag's value, refused when it is missing, empty or looks like another
/// flag — a forgotten value must not swallow the next flag.
fn required_flag_text(
    remaining_arguments: &[OsString],
    flag_index: usize,
    flag: &'static str,
) -> Result<String, UsageError> {
    let flag_text = required_flag_value(remaining_arguments, flag_index, flag)?;
    if flag_text.is_empty() || flag_text.starts_with('-') {
        return Err(UsageError::MissingValue(flag));
    }
    Ok(flag_text)
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
