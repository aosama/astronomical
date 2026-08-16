use serde_json::{Map, Value};

use super::document::bounded_value_label;
use super::error::LagunaNormalizationError;

pub(super) fn validate_common_null_defaults(
    fields: &Map<String, Value>,
    location: &str,
) -> Result<(), LagunaNormalizationError> {
    validate_optional_null(fields, "actorder", location)?;
    validate_optional_empty_object(fields, "observer_kwargs", location)?;
    validate_optional_null(fields, "zp_dtype", location)
}

pub(super) fn validate_optional_block_128(
    fields: &Map<String, Value>,
    location: &str,
) -> Result<(), LagunaNormalizationError> {
    let Some(block_structure) = fields.get("block_structure") else {
        return Ok(());
    };
    if block_structure == &serde_json::json!([128, 128]) {
        return Ok(());
    }
    Err(unsupported(
        &format!("{location}.block_structure"),
        "[128,128]",
        block_structure,
    ))
}

pub(super) fn reject_unknown_fields(
    fields: &Map<String, Value>,
    supported_fields: &[&str],
    location: &str,
) -> Result<(), LagunaNormalizationError> {
    if let Some((field_name, field_value)) = fields
        .iter()
        .find(|(field_name, _)| !supported_fields.contains(&field_name.as_str()))
    {
        return Err(unsupported(
            &format!("{location}.{field_name}"),
            "supported field",
            field_value,
        ));
    }
    Ok(())
}

pub(super) fn required_object<'a>(
    fields: &'a Map<String, Value>,
    field_name: &str,
    location: &str,
) -> Result<&'a Map<String, Value>, LagunaNormalizationError> {
    fields
        .get(field_name)
        .and_then(Value::as_object)
        .ok_or_else(|| {
            unsupported(
                &format!("{location}.{field_name}"),
                "object",
                fields.get(field_name).unwrap_or(&Value::Null),
            )
        })
}

pub(super) fn optional_string<'a>(
    fields: &'a Map<String, Value>,
    field_name: &str,
    location: &str,
) -> Result<Option<&'a str>, LagunaNormalizationError> {
    fields
        .get(field_name)
        .map(|value| {
            value
                .as_str()
                .ok_or_else(|| unsupported(&format!("{location}.{field_name}"), "string", value))
        })
        .transpose()
}

pub(super) fn validate_exact_u32(
    fields: &Map<String, Value>,
    field_name: &str,
    expected_value: u32,
    location: &str,
) -> Result<(), LagunaNormalizationError> {
    if fields.get(field_name).and_then(Value::as_u64) == Some(u64::from(expected_value)) {
        return Ok(());
    }
    Err(field_error(
        fields,
        field_name,
        location,
        "exact evidenced integer",
    ))
}

pub(super) fn validate_exact_string(
    fields: &Map<String, Value>,
    field_name: &str,
    expected_value: &str,
    location: &str,
) -> Result<(), LagunaNormalizationError> {
    if fields.get(field_name).and_then(Value::as_str) == Some(expected_value) {
        return Ok(());
    }
    Err(field_error(
        fields,
        field_name,
        location,
        "exact evidenced string",
    ))
}

pub(super) fn validate_optional_string(
    fields: &Map<String, Value>,
    field_name: &str,
    expected_value: &str,
    location: &str,
) -> Result<(), LagunaNormalizationError> {
    match fields.get(field_name) {
        None => Ok(()),
        Some(value) if value.as_str() == Some(expected_value) => Ok(()),
        Some(_) => Err(field_error(
            fields,
            field_name,
            location,
            "exact evidenced string",
        )),
    }
}

pub(super) fn validate_optional_nonempty_string(
    fields: &Map<String, Value>,
    field_name: &str,
    location: &str,
) -> Result<(), LagunaNormalizationError> {
    match fields.get(field_name) {
        None => Ok(()),
        Some(Value::String(value)) if !value.is_empty() => Ok(()),
        Some(_) => Err(field_error(
            fields,
            field_name,
            location,
            "non-empty string",
        )),
    }
}

pub(super) fn validate_exact_true(
    fields: &Map<String, Value>,
    field_name: &str,
    location: &str,
) -> Result<(), LagunaNormalizationError> {
    validate_exact_bool(fields, field_name, true, location)
}

pub(super) fn validate_exact_false(
    fields: &Map<String, Value>,
    field_name: &str,
    location: &str,
) -> Result<(), LagunaNormalizationError> {
    validate_exact_bool(fields, field_name, false, location)
}

fn validate_exact_bool(
    fields: &Map<String, Value>,
    field_name: &str,
    expected_value: bool,
    location: &str,
) -> Result<(), LagunaNormalizationError> {
    if fields.get(field_name).and_then(Value::as_bool) == Some(expected_value) {
        return Ok(());
    }
    Err(field_error(
        fields,
        field_name,
        location,
        "exact evidenced boolean",
    ))
}

pub(super) fn validate_optional_true(
    fields: &Map<String, Value>,
    field_name: &str,
    location: &str,
) -> Result<(), LagunaNormalizationError> {
    validate_optional_bool(fields, field_name, true, location)
}

pub(super) fn validate_optional_false(
    fields: &Map<String, Value>,
    field_name: &str,
    location: &str,
) -> Result<(), LagunaNormalizationError> {
    validate_optional_bool(fields, field_name, false, location)
}

fn validate_optional_bool(
    fields: &Map<String, Value>,
    field_name: &str,
    expected_value: bool,
    location: &str,
) -> Result<(), LagunaNormalizationError> {
    match fields.get(field_name) {
        None => Ok(()),
        Some(value) if value.as_bool() == Some(expected_value) => Ok(()),
        Some(_) => Err(field_error(
            fields,
            field_name,
            location,
            "exact evidenced boolean",
        )),
    }
}

pub(super) fn validate_optional_null(
    fields: &Map<String, Value>,
    field_name: &str,
    location: &str,
) -> Result<(), LagunaNormalizationError> {
    match fields.get(field_name) {
        None | Some(Value::Null) => Ok(()),
        Some(_) => Err(field_error(fields, field_name, location, "null")),
    }
}

pub(super) fn validate_optional_empty_object(
    fields: &Map<String, Value>,
    field_name: &str,
    location: &str,
) -> Result<(), LagunaNormalizationError> {
    match fields.get(field_name) {
        None => Ok(()),
        Some(Value::Object(object)) if object.is_empty() => Ok(()),
        Some(_) => Err(field_error(fields, field_name, location, "empty object")),
    }
}

fn field_error(
    fields: &Map<String, Value>,
    field_name: &str,
    location: &str,
    description: &'static str,
) -> LagunaNormalizationError {
    unsupported(
        &format!("{location}.{field_name}"),
        description,
        fields.get(field_name).unwrap_or(&Value::Null),
    )
}

pub(super) fn unsupported(
    location: &str,
    description: &'static str,
    actual_value: &Value,
) -> LagunaNormalizationError {
    LagunaNormalizationError::UnsupportedQuantizationValue {
        location: location.to_owned(),
        description,
        actual_value: bounded_value_label(actual_value),
    }
}
