//! Unit tests for `crate::verbs::schema` argument parsing: the `schema object` grammar —
//! property kinds, modifiers, dotted paths, and the rejection rules.

use std::ffi::OsString;

use crate::arguments::UsageError;
use crate::arguments::parse_command;
use crate::verbs::schema::{SchemaArguments, SchemaPropertyKind};

fn parse(arguments: &[&str]) -> Result<crate::arguments::CliCommand, UsageError> {
    let process_arguments =
        std::iter::once(OsString::from("astronomical")).chain(arguments.iter().map(OsString::from));
    parse_command(process_arguments)
}

fn parsed_schema(arguments: &[&str]) -> SchemaArguments {
    let mut full_arguments = vec!["schema"];
    full_arguments.extend_from_slice(arguments);
    match parse(&full_arguments) {
        Ok(crate::arguments::CliCommand::Schema(schema_arguments)) => schema_arguments,
        other => panic!("expected schema command, got {other:?}"),
    }
}

#[test]
fn should_map_double_kind_to_json_number() {
    assert_eq!(SchemaPropertyKind::Double.json_type_name(), "number");
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

fn os_arguments(argument_texts: &[&str]) -> Vec<OsString> {
    argument_texts.iter().map(OsString::from).collect()
}

#[test]
fn should_recognize_exactly_the_four_kind_flags() {
    for property_kind in [
        SchemaPropertyKind::String,
        SchemaPropertyKind::Integer,
        SchemaPropertyKind::Double,
        SchemaPropertyKind::Boolean,
    ] {
        assert_eq!(
            SchemaPropertyKind::from_kind_flag(property_kind.kind_flag()),
            Some(property_kind)
        );
    }
    assert_eq!(SchemaPropertyKind::from_kind_flag("--number"), None);
    assert_eq!(SchemaPropertyKind::from_kind_flag("string"), None);
}

#[test]
fn should_refuse_a_kind_flag_whose_path_is_another_flag() {
    assert_eq!(
        super::parse_schema_arguments(&os_arguments(&["--name", "Play", "--string", "--array"])),
        Err(UsageError::MissingValue("--string"))
    );
}

#[test]
fn should_refuse_a_description_without_text_at_the_end_of_the_arguments() {
    assert_eq!(
        super::parse_schema_arguments(&os_arguments(&[
            "--name",
            "Play",
            "--string",
            "title",
            "--description",
        ])),
        Err(UsageError::MissingValue("--description"))
    );
}

#[test]
fn should_refuse_a_repeated_object_name() {
    assert_eq!(
        super::parse_schema_arguments(&os_arguments(&["--name", "Play", "--name", "Sonnet"])),
        Err(UsageError::RepeatedArgument("--name"))
    );
}

#[test]
fn should_attach_a_description_to_the_most_recent_property_only() {
    let schema_arguments = super::parse_schema_arguments(&os_arguments(&[
        "--name",
        "Play",
        "--string",
        "title",
        "--int",
        "act_count",
        "--description",
        "Acts in the play",
    ]))
    .expect("a described property parses");
    assert_eq!(schema_arguments.properties[0].description, None);
    assert_eq!(
        schema_arguments.properties[1].description.as_deref(),
        Some("Acts in the play")
    );
}
