use std::collections::BTreeSet;

use serde_json::Value;

use super::{
    document::{
        LagunaConfigurationDocument, MAX_LAYER_COUNT, MAX_MODEL_DIMENSION, bounded_value_label,
    },
    error::LagunaNormalizationError,
    layer_descriptor::{
        LagunaAttentionKind, LagunaDenseFeedForwardDescriptor, LagunaFeedForwardDescriptor,
        LagunaGatingKind, LagunaMoeDescriptor,
    },
};

pub(super) struct LagunaLayerSchedules {
    pub(super) attention_kinds: Vec<LagunaAttentionKind>,
    pub(super) query_head_counts: Vec<u32>,
    pub(super) gating_kinds: Vec<LagunaGatingKind>,
    pub(super) feed_forward_descriptors: Vec<LagunaFeedForwardDescriptor>,
    pub(super) sliding_window: Option<u32>,
}

pub(super) fn normalize_layer_schedules(
    document: &LagunaConfigurationDocument,
    layer_count: usize,
    dense_intermediate_size: u32,
    key_value_head_count: u32,
    legacy_boolean_gating_kind: Option<LagunaGatingKind>,
) -> Result<LagunaLayerSchedules, LagunaNormalizationError> {
    let attention_kinds = normalize_attention_kinds(document, layer_count)?;
    let sliding_window = if attention_kinds.contains(&LagunaAttentionKind::Sliding) {
        Some(document.required_u32("sliding_window", MAX_MODEL_DIMENSION)?)
    } else {
        None
    };
    let query_head_counts =
        normalize_query_head_counts(document, layer_count, key_value_head_count)?;
    let gating_kinds = normalize_gating_kinds(document, layer_count, legacy_boolean_gating_kind)?;
    let feed_forward_descriptors =
        normalize_feed_forward_descriptors(document, layer_count, dense_intermediate_size)?;
    Ok(LagunaLayerSchedules {
        attention_kinds,
        query_head_counts,
        gating_kinds,
        feed_forward_descriptors,
        sliding_window,
    })
}

fn normalize_attention_kinds(
    document: &LagunaConfigurationDocument,
    layer_count: usize,
) -> Result<Vec<LagunaAttentionKind>, LagunaNormalizationError> {
    let Some(layer_types_value) = document.field("layer_types") else {
        return Ok(vec![LagunaAttentionKind::Full; layer_count]);
    };
    parse_layer_array(
        layer_types_value,
        "layer_types",
        layer_count,
        |layer_value| match layer_value {
            Value::String(kind) if matches!(kind.as_str(), "full" | "full_attention") => {
                Ok(LagunaAttentionKind::Full)
            }
            Value::String(kind) if matches!(kind.as_str(), "sliding" | "sliding_attention") => {
                Ok(LagunaAttentionKind::Sliding)
            }
            actual_value => unsupported("layer_types", actual_value),
        },
    )
}

fn normalize_query_head_counts(
    document: &LagunaConfigurationDocument,
    layer_count: usize,
    key_value_head_count: u32,
) -> Result<Vec<u32>, LagunaNormalizationError> {
    let query_head_counts =
        if let Some(per_layer_heads) = document.field("num_attention_heads_per_layer") {
            parse_layer_array(
                per_layer_heads,
                "num_attention_heads_per_layer",
                layer_count,
                |head_count_value| {
                    parse_positive_bounded_u32(
                        head_count_value,
                        "num_attention_heads_per_layer",
                        MAX_MODEL_DIMENSION,
                    )
                },
            )?
        } else {
            let global_head_count =
                document.required_u32("num_attention_heads", MAX_MODEL_DIMENSION)?;
            vec![global_head_count; layer_count]
        };
    for (layer_index, query_head_count) in query_head_counts.iter().copied().enumerate() {
        if !query_head_count.is_multiple_of(key_value_head_count) {
            return Err(LagunaNormalizationError::InvalidHeadDivisibility {
                layer_index,
                query_head_count,
                key_value_head_count,
            });
        }
    }
    Ok(query_head_counts)
}

fn normalize_gating_kinds(
    document: &LagunaConfigurationDocument,
    layer_count: usize,
    legacy_boolean_gating_kind: Option<LagunaGatingKind>,
) -> Result<Vec<LagunaGatingKind>, LagunaNormalizationError> {
    if let Some(gating_types) = document.field("gating_types") {
        return parse_layer_array(gating_types, "gating_types", layer_count, |gating_value| {
            parse_gating_kind(gating_value, "gating_types", legacy_boolean_gating_kind)
        });
    }
    let global_gating = match document.field("gating") {
        Some(gating_value) => {
            parse_gating_kind(gating_value, "gating", legacy_boolean_gating_kind)?
        }
        None => LagunaGatingKind::None,
    };
    Ok(vec![global_gating; layer_count])
}

fn parse_gating_kind(
    gating_value: &Value,
    field_name: &str,
    legacy_boolean_gating_kind: Option<LagunaGatingKind>,
) -> Result<LagunaGatingKind, LagunaNormalizationError> {
    match gating_value {
        Value::Bool(false) => Ok(LagunaGatingKind::None),
        Value::Bool(true) => {
            legacy_boolean_gating_kind.ok_or(LagunaNormalizationError::AmbiguousGatingBoolean)
        }
        Value::String(kind) if matches!(kind.as_str(), "none" | "false") => {
            Ok(LagunaGatingKind::None)
        }
        Value::String(kind) if matches!(kind.as_str(), "per_head" | "per-head") => {
            Ok(LagunaGatingKind::PerHead)
        }
        Value::String(kind) if matches!(kind.as_str(), "per_element" | "per-element") => {
            Ok(LagunaGatingKind::PerElement)
        }
        actual_value => unsupported(field_name, actual_value),
    }
}

fn normalize_feed_forward_descriptors(
    document: &LagunaConfigurationDocument,
    layer_count: usize,
    dense_intermediate_size: u32,
) -> Result<Vec<LagunaFeedForwardDescriptor>, LagunaNormalizationError> {
    let dense_descriptor = LagunaFeedForwardDescriptor::Dense(
        LagunaDenseFeedForwardDescriptor::new(dense_intermediate_size),
    );
    let explicit_kinds = document
        .field("mlp_layer_types")
        .map(|layer_types| {
            parse_layer_array(layer_types, "mlp_layer_types", layer_count, |layer_value| {
                match layer_value {
                    Value::String(kind) if kind == "dense" => Ok(false),
                    Value::String(kind) if matches!(kind.as_str(), "sparse" | "moe") => Ok(true),
                    actual_value => unsupported("mlp_layer_types", actual_value),
                }
            })
        })
        .transpose()?;

    if explicit_kinds
        .as_ref()
        .is_some_and(|kinds| !kinds.contains(&true))
    {
        return Ok(vec![dense_descriptor; layer_count]);
    }

    let expert_count = document
        .optional_non_negative_u32("num_experts", MAX_MODEL_DIMENSION)?
        .unwrap_or(0);
    if expert_count == 0 {
        if explicit_kinds
            .as_ref()
            .is_some_and(|kinds| kinds.contains(&true))
        {
            return Err(LagunaNormalizationError::MissingRequiredField {
                field_name: "num_experts".to_owned(),
            });
        }
        return Ok(vec![dense_descriptor; layer_count]);
    }
    let sparse_descriptor =
        LagunaFeedForwardDescriptor::Moe(normalize_moe_descriptor(document, expert_count)?);
    if let Some(explicit_kinds) = explicit_kinds {
        return Ok(explicit_kinds
            .into_iter()
            .map(|is_sparse| {
                if is_sparse {
                    sparse_descriptor
                } else {
                    dense_descriptor
                }
            })
            .collect());
    }

    let dense_layer_indexes = parse_dense_layer_indexes(document, layer_count)?;
    let sparse_step = document
        .optional_u32("decoder_sparse_step", MAX_LAYER_COUNT)?
        .unwrap_or(1);
    let sparse_step = usize::try_from(sparse_step).map_err(|_| {
        super::document::invalid_numeric(
            "decoder_sparse_step",
            "cannot be represented by this platform",
        )
    })?;
    Ok((0..layer_count)
        .map(|layer_index| {
            let cadence_selects_sparse = (layer_index + 1).is_multiple_of(sparse_step);
            if cadence_selects_sparse && !dense_layer_indexes.contains(&layer_index) {
                sparse_descriptor
            } else {
                dense_descriptor
            }
        })
        .collect())
}

fn normalize_moe_descriptor(
    document: &LagunaConfigurationDocument,
    expert_count: u32,
) -> Result<LagunaMoeDescriptor, LagunaNormalizationError> {
    let experts_per_token = document.required_u32("num_experts_per_tok", MAX_MODEL_DIMENSION)?;
    if experts_per_token > expert_count {
        return Err(LagunaNormalizationError::TopKExceedsExpertCount {
            experts_per_token,
            expert_count,
        });
    }
    if let Some(scoring_function) = document.field("scoring_func")
        && scoring_function.as_str() != Some("sigmoid")
    {
        return unsupported("scoring_func", scoring_function);
    }
    let routed_scaling_factor = document
        .optional_f64("moe_routed_scaling_factor")?
        .unwrap_or(1.0);
    if routed_scaling_factor <= 0.0 {
        return Err(super::document::invalid_numeric(
            "moe_routed_scaling_factor",
            "must be a positive finite number",
        ));
    }
    Ok(LagunaMoeDescriptor::new(
        expert_count,
        experts_per_token,
        document.required_u32("moe_intermediate_size", MAX_MODEL_DIMENSION)?,
        document
            .optional_non_negative_u32("shared_expert_intermediate_size", MAX_MODEL_DIMENSION)?
            .unwrap_or(0),
        document.optional_bool("norm_topk_prob", true)?,
        routed_scaling_factor,
        document.optional_bool("moe_apply_router_weight_on_input", false)?,
    ))
}

fn parse_dense_layer_indexes(
    document: &LagunaConfigurationDocument,
    layer_count: usize,
) -> Result<BTreeSet<usize>, LagunaNormalizationError> {
    let Some(layer_indexes_value) = document.field("mlp_only_layers") else {
        return Ok(BTreeSet::new());
    };
    let layer_indexes = layer_indexes_value.as_array().ok_or_else(|| {
        LagunaNormalizationError::UnsupportedValue {
            field_name: "mlp_only_layers".to_owned(),
            actual_value: bounded_value_label(layer_indexes_value),
        }
    })?;
    let mut dense_layer_indexes = BTreeSet::new();
    for layer_index_value in layer_indexes {
        let layer_index_u64 = layer_index_value.as_u64().ok_or_else(|| {
            super::document::invalid_numeric("mlp_only_layers", "must contain integer indexes")
        })?;
        let layer_index = usize::try_from(layer_index_u64).map_err(|_| {
            super::document::invalid_numeric("mlp_only_layers", "contains an overflowing index")
        })?;
        if layer_index >= layer_count || !dense_layer_indexes.insert(layer_index) {
            return Err(super::document::invalid_numeric(
                "mlp_only_layers",
                "must contain unique in-range indexes",
            ));
        }
    }
    Ok(dense_layer_indexes)
}

fn parse_layer_array<T, ParseEntry>(
    array_value: &Value,
    field_name: &str,
    layer_count: usize,
    parse_entry: ParseEntry,
) -> Result<Vec<T>, LagunaNormalizationError>
where
    ParseEntry: Fn(&Value) -> Result<T, LagunaNormalizationError>,
{
    let layer_values =
        array_value
            .as_array()
            .ok_or_else(|| LagunaNormalizationError::UnsupportedValue {
                field_name: field_name.to_owned(),
                actual_value: bounded_value_label(array_value),
            })?;
    if layer_values.len() != layer_count {
        return Err(LagunaNormalizationError::LayerArrayLengthMismatch {
            field_name: field_name.to_owned(),
            actual_count: layer_values.len(),
            expected_count: layer_count,
        });
    }
    layer_values.iter().map(parse_entry).collect()
}

fn parse_positive_bounded_u32(
    field_value: &Value,
    field_name: &str,
    maximum_value: u32,
) -> Result<u32, LagunaNormalizationError> {
    let unsigned_value = field_value.as_u64().ok_or_else(|| {
        super::document::invalid_numeric(field_name, "must contain positive integers")
    })?;
    let bounded_value = u32::try_from(unsigned_value).map_err(|_| {
        super::document::invalid_numeric(field_name, "contains an overflowing integer")
    })?;
    if bounded_value == 0 || bounded_value > maximum_value {
        return Err(super::document::invalid_numeric(
            field_name,
            "contains an integer outside the supported safety bound",
        ));
    }
    Ok(bounded_value)
}

fn unsupported<T>(field_name: &str, actual_value: &Value) -> Result<T, LagunaNormalizationError> {
    Err(LagunaNormalizationError::UnsupportedValue {
        field_name: field_name.to_owned(),
        actual_value: bounded_value_label(actual_value),
    })
}
