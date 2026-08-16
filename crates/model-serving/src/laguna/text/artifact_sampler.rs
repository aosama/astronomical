use serde_json::{Map, Value};

use super::artifact_documents::required_u32;
use super::{LagunaSamplerConfig, LagunaTextArtifactError};

/// Preserves artifact-specific sampling differences without a family-wide policy.
pub(super) fn normalize_sampler_config(
    generation_fields: &Map<String, Value>,
) -> Result<LagunaSamplerConfig, LagunaTextArtifactError> {
    let uses_sampling = optional_bool(generation_fields, "do_sample")?.unwrap_or(false);
    let temperature_thousandths =
        optional_decimal_thousandths(generation_fields, "temperature", 0, 2_000)?.unwrap_or(1_000);
    let top_p_thousandths =
        optional_decimal_thousandths(generation_fields, "top_p", 0, 1_000)?.unwrap_or(1_000);
    let min_p_thousandths =
        optional_decimal_thousandths(generation_fields, "min_p", 0, 1_000)?.unwrap_or(0);
    let repetition_penalty_thousandths =
        optional_decimal_thousandths(generation_fields, "repetition_penalty", 1, u16::MAX)?
            .unwrap_or(1_000);
    let top_k = generation_fields
        .get("top_k")
        .map(|_| required_u32(generation_fields, "top_k", false))
        .transpose()?
        .map(|top_k| {
            u16::try_from(top_k).map_err(|_| LagunaTextArtifactError::InvalidNumericField {
                field_name: "top_k".to_owned(),
            })
        })
        .transpose()?;
    let maximum_new_tokens = generation_fields
        .get("max_new_tokens")
        .map(|_| required_u32(generation_fields, "max_new_tokens", false))
        .transpose()?;
    Ok(LagunaSamplerConfig::new(
        uses_sampling,
        temperature_thousandths,
        top_p_thousandths,
        min_p_thousandths,
        top_k,
        repetition_penalty_thousandths,
        maximum_new_tokens,
        None,
    ))
}

fn optional_bool(
    fields: &Map<String, Value>,
    field_name: &str,
) -> Result<Option<bool>, LagunaTextArtifactError> {
    fields
        .get(field_name)
        .map(|field_value| {
            field_value
                .as_bool()
                .ok_or_else(|| LagunaTextArtifactError::InvalidField {
                    field_name: field_name.to_owned(),
                })
        })
        .transpose()
}

fn optional_decimal_thousandths(
    fields: &Map<String, Value>,
    field_name: &str,
    minimum_thousandths: u16,
    maximum_thousandths: u16,
) -> Result<Option<u16>, LagunaTextArtifactError> {
    let Some(decimal_value) = fields.get(field_name) else {
        return Ok(None);
    };
    let decimal_value =
        decimal_value
            .as_f64()
            .ok_or_else(|| LagunaTextArtifactError::InvalidNumericField {
                field_name: field_name.to_owned(),
            })?;
    let exact_scaled_value = decimal_value * 1_000.0;
    let integral_scaled_value = exact_scaled_value.round();
    let floating_point_tolerance = f64::EPSILON * exact_scaled_value.abs().max(1.0) * 8.0;
    if !exact_scaled_value.is_finite()
        || (exact_scaled_value - integral_scaled_value).abs() > floating_point_tolerance
        || integral_scaled_value < f64::from(minimum_thousandths)
        || integral_scaled_value > f64::from(maximum_thousandths)
    {
        return Err(LagunaTextArtifactError::InvalidNumericField {
            field_name: field_name.to_owned(),
        });
    }
    Ok(Some(integral_scaled_value as u16))
}
