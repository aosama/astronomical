//! Hermetic acceptance tests for `astronomical schema object` (issue #821
//! slice 1). The verb must produce strict JSON schemas in-process with no
//! daemon, no GPU, and no network.

use std::ffi::OsString;

use astronomical_cli::{
    CliCommand, SchemaArguments, SchemaPropertyInput, SchemaPropertyKind, UsageError,
    build_schema_document, parse_command, run_schema,
};

fn parse(arguments: &[&str]) -> Result<CliCommand, UsageError> {
    parse_command(
        std::iter::once(OsString::from("astronomical")).chain(arguments.iter().map(OsString::from)),
    )
}

fn parsed_schema(arguments: &[&str]) -> SchemaArguments {
    let mut full_arguments = vec!["schema"];
    full_arguments.extend_from_slice(arguments);
    match parse(&full_arguments) {
        Ok(CliCommand::Schema(schema_arguments)) => schema_arguments,
        other => panic!("expected schema command, got {other:?}"),
    }
}

#[test]
fn should_parse_schema_object_with_basic_properties() {
    let schema_arguments = parsed_schema(&[
        "object", "--name", "Color", "--string", "name", "--int", "position",
    ]);
    assert_eq!(schema_arguments.object_name, "Color");
    assert_eq!(schema_arguments.properties.len(), 2);
    assert_eq!(schema_arguments.properties[0].dotted_path, "name");
    assert_eq!(
        schema_arguments.properties[0].kind,
        SchemaPropertyKind::String
    );
    assert_eq!(schema_arguments.properties[1].dotted_path, "position");
    assert_eq!(
        schema_arguments.properties[1].kind,
        SchemaPropertyKind::Integer
    );
}

#[test]
fn should_parse_every_property_kind() {
    let schema_arguments = parsed_schema(&[
        "object",
        "--name",
        "Everything",
        "--string",
        "text",
        "--int",
        "count",
        "--double",
        "ratio",
        "--boolean",
        "enabled",
    ]);
    let kinds = schema_arguments
        .properties
        .iter()
        .map(|property| property.kind)
        .collect::<Vec<_>>();
    assert_eq!(
        kinds,
        vec![
            SchemaPropertyKind::String,
            SchemaPropertyKind::Integer,
            SchemaPropertyKind::Double,
            SchemaPropertyKind::Boolean,
        ]
    );
}

#[test]
fn should_apply_modifiers_to_the_preceding_property() {
    let schema_arguments = parsed_schema(&[
        "object",
        "--name",
        "Tags",
        "--string",
        "label",
        "--array",
        "--optional",
        "--description",
        "Optional tag names",
        "--int",
        "weight",
    ]);
    assert_eq!(schema_arguments.properties.len(), 2);
    let label_property = &schema_arguments.properties[0];
    assert!(label_property.is_array);
    assert!(label_property.is_optional);
    assert_eq!(
        label_property.description.as_deref(),
        Some("Optional tag names")
    );
    let weight_property = &schema_arguments.properties[1];
    assert!(!weight_property.is_array);
    assert!(!weight_property.is_optional);
    assert!(weight_property.description.is_none());
}

#[test]
fn should_reject_schema_object_without_name() {
    let usage_error =
        parse(&["schema", "object", "--string", "name"]).expect_err("name is required");
    assert!(usage_error.to_string().contains("--name"));
}

#[test]
fn should_reject_schema_object_without_properties() {
    let usage_error = parse(&["schema", "object", "--name", "Empty"])
        .expect_err("at least one property is required");
    assert!(usage_error.to_string().contains("property"));
}

#[test]
fn should_reject_modifier_before_any_property() {
    let usage_error = parse(&["schema", "object", "--name", "Broken", "--array"])
        .expect_err("no property to modify");
    assert!(usage_error.to_string().contains("--array"));
}

#[test]
fn should_reject_duplicate_property_path() {
    let usage_error = parse(&[
        "schema", "object", "--name", "Dupes", "--string", "name", "--int", "name",
    ])
    .expect_err("duplicate property");
    assert!(usage_error.to_string().contains("name"));
}

#[test]
fn should_reject_unknown_schema_flag() {
    let usage_error = parse(&[
        "schema", "object", "--name", "Weird", "--string", "name", "--long",
    ])
    .expect_err("unknown flag");
    assert!(usage_error.to_string().contains("--long"));
}

#[test]
fn should_build_strict_object_schema_document() {
    let schema_arguments = SchemaArguments {
        object_name: "Color".to_owned(),
        properties: vec![
            SchemaPropertyInput {
                dotted_path: "name".to_owned(),
                kind: SchemaPropertyKind::String,
                is_array: false,
                is_optional: false,
                description: None,
            },
            SchemaPropertyInput {
                dotted_path: "position".to_owned(),
                kind: SchemaPropertyKind::Integer,
                is_array: false,
                is_optional: false,
                description: None,
            },
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
fn should_reject_schema_without_object_noun() {
    let usage_error = parse(&["schema", "property"]).expect_err("only object is supported");
    assert!(usage_error.to_string().contains("object"));
}

#[test]
fn should_reject_leaf_property_conflicting_with_nested_path() {
    let usage_error = parse(&[
        "schema",
        "object",
        "--name",
        "Clash",
        "--string",
        "address",
        "--string",
        "address.street",
    ])
    .expect_err("leaf conflicts with nested object");
    assert!(usage_error.to_string().contains("address"));
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

fn leaf_property(dotted_path: &str, kind: SchemaPropertyKind) -> SchemaPropertyInput {
    SchemaPropertyInput {
        dotted_path: dotted_path.to_owned(),
        kind,
        is_array: false,
        is_optional: false,
        description: None,
    }
}
