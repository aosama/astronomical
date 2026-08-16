use serde_json::{Map, Value};

use super::{
    document::{LagunaConfigurationDocument, MAX_POSITION_COUNT, bounded_value_label},
    error::LagunaNormalizationError,
    layer_descriptor::LagunaAttentionKind,
    rope_descriptor::{
        LagunaDefaultRopeDescriptor, LagunaRopeDescriptor, LagunaYarnRopeDescriptor,
    },
};

pub(super) fn normalize_rope_for_attention_kind(
    document: &LagunaConfigurationDocument,
    attention_kind: LagunaAttentionKind,
    head_dimension: u32,
) -> Result<LagunaRopeDescriptor, LagunaNormalizationError> {
    let selected_parameters = select_rope_parameters(document, attention_kind)?;
    let top_level_theta = optional_document_float(document, "rope_theta")?;
    let top_level_partial = optional_document_float(document, "partial_rotary_factor")?;
    let rope_theta = optional_object_float(selected_parameters, "rope_theta")?
        .or(top_level_theta)
        .unwrap_or(10_000.0);
    validate_positive_finite("rope_theta", rope_theta)?;
    let partial_rotary_factor =
        optional_object_float(selected_parameters, "partial_rotary_factor")?
            .or(top_level_partial)
            .unwrap_or(1.0);
    let rotary_dimension = normalize_rotary_dimension(head_dimension, partial_rotary_factor)?;
    let rope_type =
        object_string_alias(selected_parameters, "rope_type", "type")?.unwrap_or("default");
    match rope_type {
        "default" | "none" => Ok(LagunaRopeDescriptor::Default(
            LagunaDefaultRopeDescriptor::new(rope_theta, rotary_dimension),
        )),
        "yarn" => normalize_yarn(selected_parameters, rope_theta, rotary_dimension),
        actual_value => Err(LagunaNormalizationError::UnsupportedValue {
            field_name: "rope_type".to_owned(),
            actual_value: actual_value.to_owned(),
        }),
    }
}

fn select_rope_parameters<'a>(
    document: &'a LagunaConfigurationDocument,
    attention_kind: LagunaAttentionKind,
) -> Result<&'a Map<String, Value>, LagunaNormalizationError> {
    let rope_parameters = optional_object(document.field("rope_parameters"), "rope_parameters")?;
    let nested_field_name = match attention_kind {
        LagunaAttentionKind::Full => "full_attention",
        LagunaAttentionKind::Sliding => "sliding_attention",
    };
    if let Some(nested_parameters) =
        rope_parameters.and_then(|fields| fields.get(nested_field_name))
    {
        return required_object(
            nested_parameters,
            &format!("rope_parameters.{nested_field_name}"),
        );
    }
    if attention_kind == LagunaAttentionKind::Sliding
        && let Some(sliding_override) = document.field("swa_rope_parameters")
    {
        return required_object(sliding_override, "swa_rope_parameters");
    }
    if let Some(flat_parameters) = rope_parameters
        && !contains_per_kind_parameters(flat_parameters)
    {
        return Ok(flat_parameters);
    }
    if let Some(legacy_parameters) = document.field("rope_scaling") {
        return required_object(legacy_parameters, "rope_scaling");
    }
    // Empty parameters permit the documented default policy and top-level fallbacks.
    static EMPTY_PARAMETERS: std::sync::LazyLock<Map<String, Value>> =
        std::sync::LazyLock::new(Map::new);
    Ok(&EMPTY_PARAMETERS)
}

fn normalize_yarn(
    parameters: &Map<String, Value>,
    rope_theta: f64,
    rotary_dimension: u32,
) -> Result<LagunaRopeDescriptor, LagunaNormalizationError> {
    let factor = required_object_float(parameters, "factor")?;
    let original_maximum_position_count = required_object_u32(
        parameters,
        "original_max_position_embeddings",
        MAX_POSITION_COUNT,
    )?;
    let beta_slow = required_object_float(parameters, "beta_slow")?;
    let beta_fast = required_object_float(parameters, "beta_fast")?;
    let attention_factor = required_object_float(parameters, "attention_factor")?;
    for (field_name, numeric_value) in [
        ("factor", factor),
        ("beta_slow", beta_slow),
        ("beta_fast", beta_fast),
        ("attention_factor", attention_factor),
    ] {
        validate_positive_finite(field_name, numeric_value)?;
    }
    if beta_fast < beta_slow {
        return Err(invalid_rope(
            "beta_fast",
            "must be greater than or equal to beta_slow",
        ));
    }
    Ok(LagunaRopeDescriptor::Yarn(LagunaYarnRopeDescriptor::new(
        rope_theta,
        factor,
        original_maximum_position_count,
        beta_slow,
        beta_fast,
        attention_factor,
        rotary_dimension,
    )))
}

fn normalize_rotary_dimension(
    head_dimension: u32,
    partial_rotary_factor: f64,
) -> Result<u32, LagunaNormalizationError> {
    if !partial_rotary_factor.is_finite()
        || partial_rotary_factor <= 0.0
        || partial_rotary_factor > 1.0
    {
        return Err(invalid_rope(
            "partial_rotary_factor",
            "must be finite, positive, and at most one",
        ));
    }
    let rotary_dimension_f64 = f64::from(head_dimension) * partial_rotary_factor;
    if rotary_dimension_f64.fract() != 0.0
        || rotary_dimension_f64 <= 0.0
        || rotary_dimension_f64 > f64::from(u32::MAX)
    {
        return Err(invalid_rope(
            "partial_rotary_factor",
            "must produce an integral positive rotary dimension",
        ));
    }
    let rotary_dimension = rotary_dimension_f64 as u32;
    if !rotary_dimension.is_multiple_of(2) {
        return Err(invalid_rope(
            "partial_rotary_factor",
            "must produce an even rotary dimension",
        ));
    }
    Ok(rotary_dimension)
}

fn optional_document_float(
    document: &LagunaConfigurationDocument,
    field_name: &str,
) -> Result<Option<f64>, LagunaNormalizationError> {
    let Some(field_value) = document.field(field_name) else {
        return Ok(None);
    };
    parse_float(field_value, field_name).map(Some)
}

fn optional_object_float(
    fields: &Map<String, Value>,
    field_name: &str,
) -> Result<Option<f64>, LagunaNormalizationError> {
    fields
        .get(field_name)
        .map(|field_value| parse_float(field_value, field_name))
        .transpose()
}

fn required_object_float(
    fields: &Map<String, Value>,
    field_name: &str,
) -> Result<f64, LagunaNormalizationError> {
    optional_object_float(fields, field_name)?
        .ok_or_else(|| invalid_rope(field_name, "is required for YaRN"))
}

fn required_object_u32(
    fields: &Map<String, Value>,
    field_name: &str,
    maximum_value: u32,
) -> Result<u32, LagunaNormalizationError> {
    let field_value = fields
        .get(field_name)
        .ok_or_else(|| invalid_rope(field_name, "is required for YaRN"))?;
    let unsigned_value = field_value
        .as_u64()
        .ok_or_else(|| invalid_rope(field_name, "must be a positive integer"))?;
    let bounded_value = u32::try_from(unsigned_value)
        .map_err(|_| invalid_rope(field_name, "exceeds the supported integer range"))?;
    if bounded_value == 0 || bounded_value > maximum_value {
        return Err(invalid_rope(
            field_name,
            "must be positive and within the supported safety bound",
        ));
    }
    Ok(bounded_value)
}

fn object_string_alias<'a>(
    fields: &'a Map<String, Value>,
    primary_name: &str,
    alias_name: &str,
) -> Result<Option<&'a str>, LagunaNormalizationError> {
    let primary_value = fields.get(primary_name);
    let alias_value = fields.get(alias_name);
    if let (Some(primary_value), Some(alias_value)) = (primary_value, alias_value)
        && primary_value != alias_value
    {
        return Err(LagunaNormalizationError::ConflictingEnvelopeField {
            field_name: format!("{primary_name}/{alias_name}"),
        });
    }
    let Some(selected_value) = primary_value.or(alias_value) else {
        return Ok(None);
    };
    selected_value
        .as_str()
        .map(Some)
        .ok_or_else(|| LagunaNormalizationError::UnsupportedValue {
            field_name: primary_name.to_owned(),
            actual_value: bounded_value_label(selected_value),
        })
}

fn parse_float(field_value: &Value, field_name: &str) -> Result<f64, LagunaNormalizationError> {
    let numeric_value = field_value
        .as_f64()
        .ok_or_else(|| invalid_rope(field_name, "must be a finite JSON number"))?;
    if !numeric_value.is_finite() {
        return Err(invalid_rope(field_name, "must be finite"));
    }
    Ok(numeric_value)
}

fn validate_positive_finite(
    field_name: &str,
    numeric_value: f64,
) -> Result<(), LagunaNormalizationError> {
    if numeric_value.is_finite() && numeric_value > 0.0 {
        return Ok(());
    }
    Err(invalid_rope(field_name, "must be a positive finite number"))
}

fn optional_object<'a>(
    field_value: Option<&'a Value>,
    field_name: &str,
) -> Result<Option<&'a Map<String, Value>>, LagunaNormalizationError> {
    field_value
        .map(|field_value| required_object(field_value, field_name))
        .transpose()
}

fn required_object<'a>(
    field_value: &'a Value,
    field_name: &str,
) -> Result<&'a Map<String, Value>, LagunaNormalizationError> {
    field_value
        .as_object()
        .ok_or_else(|| LagunaNormalizationError::ExpectedObject {
            field_name: field_name.to_owned(),
        })
}

fn contains_per_kind_parameters(fields: &Map<String, Value>) -> bool {
    fields.contains_key("full_attention") || fields.contains_key("sliding_attention")
}

fn invalid_rope(field_name: &str, description: &'static str) -> LagunaNormalizationError {
    LagunaNormalizationError::InvalidRopeValue {
        field_name: field_name.to_owned(),
        description,
    }
}
