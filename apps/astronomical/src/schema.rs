//! Builds strict JSON object schemas in-process for `astronomical schema
//! object`. The document shape follows the fm convention: `additionalProperties:
//! false` and a `required` list at every object level so downstream tools can
//! never silently accept undeclared keys.

use std::io;

use serde_json::{Value, json};

use crate::schema_arguments::{SchemaArguments, SchemaPropertyInput};

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

#[cfg(test)]
mod tests {
    use crate::schema_arguments::SchemaPropertyKind;

    #[test]
    fn should_map_double_kind_to_json_number() {
        assert_eq!(SchemaPropertyKind::Double.json_type_name(), "number");
    }
}
