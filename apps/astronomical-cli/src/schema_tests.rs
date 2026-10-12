//! Unit tests for `crate::schema`: strict JSON object schema construction
//! from parsed `schema object` arguments and its stdout rendering.

use crate::schema::{build_schema_document, run_schema};
use crate::schema_arguments::{SchemaArguments, SchemaPropertyInput, SchemaPropertyKind};

fn leaf_property(dotted_path: &str, kind: SchemaPropertyKind) -> SchemaPropertyInput {
    SchemaPropertyInput {
        dotted_path: dotted_path.to_owned(),
        kind,
        is_array: false,
        is_optional: false,
        description: None,
    }
}

#[test]
fn should_build_strict_object_schema_document() {
    let schema_arguments = SchemaArguments {
        object_name: "Color".to_owned(),
        properties: vec![
            leaf_property("name", SchemaPropertyKind::String),
            leaf_property("position", SchemaPropertyKind::Integer),
        ],
    };
    let schema_document = build_schema_document(&schema_arguments);
    assert_eq!(schema_document["title"], "Color");
    assert_eq!(schema_document["type"], "object");
    assert_eq!(schema_document["additionalProperties"], false);
    assert_eq!(schema_document["properties"]["name"]["type"], "string");
    assert_eq!(schema_document["properties"]["position"]["type"], "integer");
    let required = schema_document["required"]
        .as_array()
        .expect("required list");
    let required_names = required
        .iter()
        .map(|value| value.as_str().expect("string entries"))
        .collect::<Vec<_>>();
    assert_eq!(required_names, vec!["name", "position"]);
}

#[test]
fn should_nest_dotted_property_paths() {
    let schema_arguments = SchemaArguments {
        object_name: "Person".to_owned(),
        properties: vec![
            leaf_property("address.street", SchemaPropertyKind::String),
            leaf_property("address.city", SchemaPropertyKind::String),
            leaf_property("age", SchemaPropertyKind::Integer),
        ],
    };
    let schema_document = build_schema_document(&schema_arguments);
    let address_property = &schema_document["properties"]["address"];
    assert_eq!(address_property["type"], "object");
    assert_eq!(address_property["additionalProperties"], false);
    assert_eq!(address_property["properties"]["street"]["type"], "string");
    assert_eq!(address_property["properties"]["city"]["type"], "string");
    let nested_required = address_property["required"]
        .as_array()
        .expect("nested required list")
        .iter()
        .map(|value| value.as_str().expect("string entries"))
        .collect::<Vec<_>>();
    assert_eq!(nested_required, vec!["street", "city"]);
    assert_eq!(schema_document["properties"]["age"]["type"], "integer");
    let root_required = schema_document["required"]
        .as_array()
        .expect("root required list")
        .iter()
        .map(|value| value.as_str().expect("string entries"))
        .collect::<Vec<_>>();
    assert_eq!(root_required, vec!["address", "age"]);
}

#[test]
fn should_exclude_optional_properties_from_required() {
    let schema_arguments = SchemaArguments {
        object_name: "Draft".to_owned(),
        properties: vec![
            SchemaPropertyInput {
                dotted_path: "title".to_owned(),
                kind: SchemaPropertyKind::String,
                is_array: false,
                is_optional: true,
                description: None,
            },
            leaf_property("body", SchemaPropertyKind::String),
        ],
    };
    let schema_document = build_schema_document(&schema_arguments);
    assert_eq!(schema_document["properties"]["title"]["type"], "string");
    let root_required = schema_document["required"]
        .as_array()
        .expect("root required list")
        .iter()
        .map(|value| value.as_str().expect("string entries"))
        .collect::<Vec<_>>();
    assert_eq!(root_required, vec!["body"]);
}

#[test]
fn should_wrap_array_properties_in_items() {
    let schema_arguments = SchemaArguments {
        object_name: "Library".to_owned(),
        properties: vec![SchemaPropertyInput {
            dotted_path: "tags".to_owned(),
            kind: SchemaPropertyKind::String,
            is_array: true,
            is_optional: false,
            description: None,
        }],
    };
    let schema_document = build_schema_document(&schema_arguments);
    let tags_property = &schema_document["properties"]["tags"];
    assert_eq!(tags_property["type"], "array");
    assert_eq!(tags_property["items"]["type"], "string");
}

#[test]
fn should_attach_property_descriptions() {
    let schema_arguments = SchemaArguments {
        object_name: "Described".to_owned(),
        properties: vec![SchemaPropertyInput {
            dotted_path: "name".to_owned(),
            kind: SchemaPropertyKind::String,
            is_array: false,
            is_optional: false,
            description: Some("The display name".to_owned()),
        }],
    };
    let schema_document = build_schema_document(&schema_arguments);
    assert_eq!(
        schema_document["properties"]["name"]["description"],
        "The display name"
    );
}

#[test]
fn should_write_pretty_json_schema_to_stdout() {
    let schema_arguments = SchemaArguments {
        object_name: "Color".to_owned(),
        properties: vec![leaf_property("name", SchemaPropertyKind::String)],
    };
    let mut rendered_output = Vec::new();
    run_schema(&schema_arguments, &mut rendered_output).expect("schema renders");
    let rendered_text = String::from_utf8(rendered_output).expect("utf-8 output");
    assert!(rendered_text.ends_with('\n'));
    let parsed_document: serde_json::Value =
        serde_json::from_str(rendered_text.trim_end()).expect("pretty json output");
    assert_eq!(parsed_document["title"], "Color");
}
