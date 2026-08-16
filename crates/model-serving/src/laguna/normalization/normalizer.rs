use serde_json::Value;

use super::{
    document::{
        LagunaConfigurationDocument, MAX_LAYER_COUNT, MAX_MODEL_DIMENSION, MAX_POSITION_COUNT,
        bounded_text_label, bounded_value_label,
    },
    error::LagunaNormalizationError,
    layer_descriptor::{
        LagunaAttentionDescriptor, LagunaAttentionKind, LagunaCacheDescriptor, LagunaGatingKind,
        LagunaLayerDescriptor,
    },
    model_descriptor::{LagunaExecutionDtype, LagunaModelDescriptor},
    rope::normalize_rope_for_attention_kind,
    schedule::{LagunaLayerSchedules, normalize_layer_schedules},
    storage::normalize_storage,
    target_contract::LagunaTargetContract,
};

/// Converts one bounded raw Laguna configuration into its canonical construction contract.
pub struct LagunaTargetNormalizer;

impl LagunaTargetNormalizer {
    /// Parses, normalizes, and validates configuration before any model object is constructed.
    pub fn normalize(
        config_json_bytes: &[u8],
    ) -> Result<LagunaTargetContract, LagunaNormalizationError> {
        Self::normalize_with_legacy_boolean_gating(config_json_bytes, None)
    }

    /// Applies the published legacy `gating=true` meaning only inside artifact validation.
    pub(crate) fn normalize_with_per_head_boolean_gating(
        config_json_bytes: &[u8],
    ) -> Result<LagunaTargetContract, LagunaNormalizationError> {
        Self::normalize_with_legacy_boolean_gating(
            config_json_bytes,
            Some(LagunaGatingKind::PerHead),
        )
    }

    fn normalize_with_legacy_boolean_gating(
        config_json_bytes: &[u8],
        legacy_boolean_gating_kind: Option<LagunaGatingKind>,
    ) -> Result<LagunaTargetContract, LagunaNormalizationError> {
        let document = LagunaConfigurationDocument::from_json_bytes(config_json_bytes)?;
        validate_family_identity(&document)?;
        let vocabulary_size = document.required_u32("vocab_size", MAX_MODEL_DIMENSION)?;
        let hidden_size = document.required_u32("hidden_size", MAX_MODEL_DIMENSION)?;
        let dense_intermediate_size =
            document.required_u32("intermediate_size", MAX_MODEL_DIMENSION)?;
        let layer_count_u32 = document.required_u32("num_hidden_layers", MAX_LAYER_COUNT)?;
        let layer_count = usize::try_from(layer_count_u32).map_err(|_| {
            super::document::invalid_numeric(
                "num_hidden_layers",
                "cannot be represented by this platform",
            )
        })?;
        let key_value_head_count =
            document.required_u32("num_key_value_heads", MAX_MODEL_DIMENSION)?;
        let head_dimension = document.required_u32("head_dim", MAX_MODEL_DIMENSION)?;
        let maximum_position_count =
            document.required_u32("max_position_embeddings", MAX_POSITION_COUNT)?;
        validate_execution_flags(&document)?;
        let rms_norm_epsilon = document.required_f64("rms_norm_eps")?;
        if rms_norm_epsilon <= 0.0 {
            return Err(super::document::invalid_numeric(
                "rms_norm_eps",
                "must be a positive finite number",
            ));
        }
        let execution_dtype = normalize_execution_dtype(&document)?;
        let declared_router_logit_softcap = document
            .optional_f64("moe_router_logit_softcapping")?
            .unwrap_or(0.0);
        if declared_router_logit_softcap < 0.0 {
            return Err(super::document::invalid_numeric(
                "moe_router_logit_softcapping",
                "must be zero or a positive finite number",
            ));
        }
        let router_logit_softcap = if declared_router_logit_softcap > 0.0 {
            declared_router_logit_softcap
        } else {
            0.0
        };

        let schedules = normalize_layer_schedules(
            &document,
            layer_count,
            dense_intermediate_size,
            key_value_head_count,
            legacy_boolean_gating_kind,
        )?;
        if schedules
            .sliding_window
            .is_some_and(|sliding_window| sliding_window > maximum_position_count)
        {
            return Err(super::document::invalid_numeric(
                "sliding_window",
                "must not exceed max_position_embeddings",
            ));
        }
        let full_rope = if schedules
            .attention_kinds
            .contains(&LagunaAttentionKind::Full)
        {
            Some(normalize_rope_for_attention_kind(
                &document,
                LagunaAttentionKind::Full,
                head_dimension,
            )?)
        } else {
            None
        };
        let sliding_rope = if schedules
            .attention_kinds
            .contains(&LagunaAttentionKind::Sliding)
        {
            Some(normalize_rope_for_attention_kind(
                &document,
                LagunaAttentionKind::Sliding,
                head_dimension,
            )?)
        } else {
            None
        };
        let LagunaLayerSchedules {
            attention_kinds,
            query_head_counts,
            gating_kinds,
            feed_forward_descriptors,
            sliding_window,
        } = schedules;
        let layers = attention_kinds
            .into_iter()
            .zip(query_head_counts)
            .zip(gating_kinds)
            .zip(feed_forward_descriptors)
            .enumerate()
            .map(
                |(
                    layer_index,
                    (((attention_kind, query_head_count), gating_kind), feed_forward),
                )| {
                    let (rope, cache) = match attention_kind {
                        LagunaAttentionKind::Full => (
                            full_rope.ok_or_else(|| missing_derived("full-attention RoPE"))?,
                            LagunaCacheDescriptor::AppendOnly,
                        ),
                        LagunaAttentionKind::Sliding => (
                            sliding_rope
                                .ok_or_else(|| missing_derived("sliding-attention RoPE"))?,
                            LagunaCacheDescriptor::Rotating {
                                window_size: sliding_window.ok_or_else(|| {
                                    missing_derived("sliding-attention cache window")
                                })?,
                            },
                        ),
                    };
                    let attention = LagunaAttentionDescriptor::new(
                        attention_kind,
                        query_head_count,
                        key_value_head_count,
                        head_dimension,
                        gating_kind,
                        rope,
                        cache,
                    );
                    Ok(LagunaLayerDescriptor::new(
                        layer_index,
                        attention,
                        feed_forward,
                    ))
                },
            )
            .collect::<Result<Vec<_>, LagunaNormalizationError>>()?;
        let model = LagunaModelDescriptor::new(
            vocabulary_size,
            hidden_size,
            dense_intermediate_size,
            layer_count,
            maximum_position_count,
            rms_norm_epsilon,
            execution_dtype,
            document.optional_bool("tie_word_embeddings", false)?,
            router_logit_softcap,
        );
        let storage = normalize_storage(document.quantization_documents())?;
        Ok(LagunaTargetContract::new(model, layers, storage))
    }
}

fn validate_family_identity(
    document: &LagunaConfigurationDocument,
) -> Result<(), LagunaNormalizationError> {
    let model_type = document.required_string("model_type")?;
    if model_type != "laguna" {
        return Err(LagunaNormalizationError::UnsupportedValue {
            field_name: "model_type".to_owned(),
            actual_value: bounded_text_label(model_type),
        });
    }
    let Some(architectures) = document.field("architectures") else {
        return Ok(());
    };
    let architecture_names =
        architectures
            .as_array()
            .ok_or_else(|| LagunaNormalizationError::UnsupportedValue {
                field_name: "architectures".to_owned(),
                actual_value: bounded_value_label(architectures),
            })?;
    if architecture_names.len() != 1
        || architecture_names.first().and_then(Value::as_str) != Some("LagunaForCausalLM")
    {
        return Err(LagunaNormalizationError::UnsupportedValue {
            field_name: "architectures".to_owned(),
            actual_value: bounded_value_label(architectures),
        });
    }
    Ok(())
}

fn validate_execution_flags(
    document: &LagunaConfigurationDocument,
) -> Result<(), LagunaNormalizationError> {
    for (field_name, actual_value, supported_value) in [
        (
            "use_cache",
            document.optional_bool("use_cache", true)?,
            true,
        ),
        (
            "attention_bias",
            document.optional_bool("attention_bias", false)?,
            false,
        ),
        (
            "qkv_bias",
            document.optional_bool("qkv_bias", false)?,
            false,
        ),
        (
            "swa_attention_sink_enabled",
            document.optional_bool("swa_attention_sink_enabled", false)?,
            false,
        ),
        (
            "moe_router_use_sigmoid",
            document.optional_bool("moe_router_use_sigmoid", true)?,
            true,
        ),
        (
            "use_bidirectional_attention",
            document.optional_bool("use_bidirectional_attention", false)?,
            false,
        ),
    ] {
        if actual_value != supported_value {
            return Err(LagunaNormalizationError::UnsupportedValue {
                field_name: field_name.to_owned(),
                actual_value: actual_value.to_string(),
            });
        }
    }
    let attention_dropout = document.optional_f64("attention_dropout")?.unwrap_or(0.0);
    if attention_dropout != 0.0 {
        return Err(LagunaNormalizationError::UnsupportedValue {
            field_name: "attention_dropout".to_owned(),
            actual_value: attention_dropout.to_string(),
        });
    }
    if let Some(rope_style_value) = document.field("rope_style")
        && rope_style_value.as_str() != Some("rotate-half")
    {
        return Err(LagunaNormalizationError::UnsupportedValue {
            field_name: "rope_style".to_owned(),
            actual_value: bounded_value_label(rope_style_value),
        });
    }
    if let Some(hidden_activation_value) = document.field("hidden_act")
        && hidden_activation_value.as_str() != Some("silu")
    {
        return Err(LagunaNormalizationError::UnsupportedValue {
            field_name: "hidden_act".to_owned(),
            actual_value: bounded_value_label(hidden_activation_value),
        });
    }
    Ok(())
}

fn normalize_execution_dtype(
    document: &LagunaConfigurationDocument,
) -> Result<LagunaExecutionDtype, LagunaNormalizationError> {
    match document.required_string("torch_dtype")? {
        "float16" | "float16_t" | "fp16" => Ok(LagunaExecutionDtype::Float16),
        "bfloat16" | "bf16" => Ok(LagunaExecutionDtype::Bfloat16),
        "float32" | "fp32" => Ok(LagunaExecutionDtype::Float32),
        unsupported_dtype => Err(LagunaNormalizationError::UnsupportedValue {
            field_name: "torch_dtype".to_owned(),
            actual_value: bounded_text_label(unsupported_dtype),
        }),
    }
}

fn missing_derived(field_name: &str) -> LagunaNormalizationError {
    LagunaNormalizationError::MissingRequiredField {
        field_name: field_name.to_owned(),
    }
}
