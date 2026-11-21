//! Reading single fields of a drafter's `config.json`: each reader names the
//! field it could not read, so a refused checkpoint says exactly why.

use serde_json::Value;

use super::drafter_configuration::{
    DFlashDrafterConfigurationError, DrafterGeneration, DrafterLayerAttention,
};

const DFLASH_ARCHITECTURE: &str = "DFlashDraftModel";
const DFLASH2_ARCHITECTURE: &str = "DFlash2DraftModel";

/// The generation `architectures` declares. DFlash2 wins when both appear,
/// as in the reference loader, which checks for DFlash2 alone.
pub(super) fn read_generation(
    document: &Value,
) -> Result<DrafterGeneration, DFlashDrafterConfigurationError> {
    let mut declares_dflash = false;
    let architectures = document.get("architectures").and_then(Value::as_array);
    for architecture in architectures.into_iter().flatten() {
        match architecture.as_str() {
            Some(DFLASH2_ARCHITECTURE) => return Ok(DrafterGeneration::DFlash2),
            Some(DFLASH_ARCHITECTURE) => declares_dflash = true,
            _ => {}
        }
    }
    if declares_dflash {
        Ok(DrafterGeneration::DFlash)
    } else {
        Err(DFlashDrafterConfigurationError::Architecture)
    }
}

/// The reference model supports these, but the published drafters leave them
/// at their identity values; a checkpoint that changes one must not be served
/// with the wrong arithmetic.
pub(super) fn refuse_unimplemented_fields(
    document: &Value,
) -> Result<(), DFlashDrafterConfigurationError> {
    let dflash = document.get("dflash_config");
    if is_set_beyond_identity(Some(document), "final_logit_softcapping", 0.0)
        || is_set_beyond_identity(dflash, "final_logit_softcapping", 0.0)
    {
        return Err(DFlashDrafterConfigurationError::Unsupported {
            field: "final_logit_softcapping",
        });
    }
    if is_set_beyond_identity(dflash, "input_embedding_scale", 1.0) {
        return Err(DFlashDrafterConfigurationError::Unsupported {
            field: "input_embedding_scale",
        });
    }
    if is_set_beyond_identity(dflash, "output_multiplier", 1.0) {
        return Err(DFlashDrafterConfigurationError::Unsupported {
            field: "output_multiplier",
        });
    }
    let rope_type = document
        .get("rope_parameters")
        .or_else(|| document.get("rope_scaling"))
        .and_then(|rope| rope.get("rope_type"))
        .and_then(Value::as_str);
    if rope_type.is_some_and(|rope_type| rope_type != "default") {
        return Err(DFlashDrafterConfigurationError::Unsupported {
            field: "rope_parameters.rope_type",
        });
    }
    Ok(())
}

/// True when `field` is present with a value other than its identity; an
/// absent field or JSON null means the identity.
fn is_set_beyond_identity(container: Option<&Value>, field: &str, identity: f64) -> bool {
    match container
        .and_then(|object| object.get(field))
        .and_then(Value::as_f64)
    {
        Some(value) => value != identity,
        None => false,
    }
}

pub(super) fn rope_theta(document: &Value) -> Result<f32, DFlashDrafterConfigurationError> {
    document
        .get("rope_parameters")
        .and_then(|rope| rope.get("rope_theta"))
        .or_else(|| document.get("rope_theta"))
        .and_then(Value::as_f64)
        .map(|theta| theta as f32)
        .ok_or(DFlashDrafterConfigurationError::Field {
            field: "rope_parameters.rope_theta",
        })
}

pub(super) fn layer_attention(
    document: &Value,
) -> Result<Vec<DrafterLayerAttention>, DFlashDrafterConfigurationError> {
    let field = DFlashDrafterConfigurationError::Field {
        field: "layer_types",
    };
    let layer_types = document
        .get("layer_types")
        .and_then(Value::as_array)
        .ok_or(field.clone())?;
    let mut attention_kinds = Vec::with_capacity(layer_types.len());
    for layer_type in layer_types {
        attention_kinds.push(match layer_type.as_str() {
            Some("sliding_attention") => DrafterLayerAttention::SlidingWindow,
            Some("full_attention") => DrafterLayerAttention::Full,
            _ => return Err(field),
        });
    }
    Ok(attention_kinds)
}

pub(super) fn integer(
    container: &Value,
    field: &'static str,
) -> Result<i32, DFlashDrafterConfigurationError> {
    container
        .get(field)
        .and_then(Value::as_i64)
        .and_then(|value| i32::try_from(value).ok())
        .filter(|value| *value >= 0)
        .ok_or(DFlashDrafterConfigurationError::Field { field })
}

pub(super) fn number(
    container: &Value,
    field: &'static str,
) -> Result<f64, DFlashDrafterConfigurationError> {
    container
        .get(field)
        .and_then(Value::as_f64)
        .ok_or(DFlashDrafterConfigurationError::Field { field })
}

pub(super) fn integer_list(
    container: &Value,
    field: &'static str,
) -> Result<Vec<i32>, DFlashDrafterConfigurationError> {
    let malformed = DFlashDrafterConfigurationError::Field { field };
    let values = container
        .get(field)
        .and_then(Value::as_array)
        .ok_or(malformed.clone())?;
    let mut integers = Vec::with_capacity(values.len());
    for value in values {
        integers.push(
            value
                .as_i64()
                .and_then(|integer| i32::try_from(integer).ok())
                .ok_or(malformed.clone())?,
        );
    }
    Ok(integers)
}
