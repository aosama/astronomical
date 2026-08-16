use serde_json::{Map, Value};

use crate::strict_json::{DUPLICATE_JSON_FIELD_MARKER, DuplicateAwareJsonValue};

use super::error::LagunaNormalizationError;

pub(super) const MAX_CONFIG_BYTES: usize = 32 * 1024 * 1024;
pub(super) const MAX_LAYER_COUNT: u32 = 4_096;
pub(super) const MAX_MODEL_DIMENSION: u32 = 16_777_216;
pub(super) const MAX_POSITION_COUNT: u32 = 1_000_000_000;
const MAXIMUM_ERROR_VALUE_CHARACTERS: usize = 256;

/// Private merged view of root metadata and authoritative nested language geometry.
pub(super) struct LagunaConfigurationDocument {
    fields: Map<String, Value>,
    quantization_documents: Vec<Value>,
}

impl LagunaConfigurationDocument {
    pub(super) fn from_json_bytes(config_bytes: &[u8]) -> Result<Self, LagunaNormalizationError> {
        if config_bytes.len() > MAX_CONFIG_BYTES {
            return Err(LagunaNormalizationError::ConfigTooLarge {
                actual_bytes: config_bytes.len(),
                maximum_bytes: MAX_CONFIG_BYTES,
            });
        }
        let root_value = serde_json::from_slice::<DuplicateAwareJsonValue>(config_bytes)
            .map_err(|json_error| {
                if json_error.to_string().contains(DUPLICATE_JSON_FIELD_MARKER) {
                    LagunaNormalizationError::DuplicateConfigField
                } else {
                    LagunaNormalizationError::MalformedJson(json_error)
                }
            })?
            .0;
        let root_fields =
            root_value
                .as_object()
                .ok_or_else(|| LagunaNormalizationError::ExpectedObject {
                    field_name: "root".to_owned(),
                })?;
        let nested_fields = match root_fields.get("text_config") {
            Some(Value::Object(fields)) => Some(fields),
            Some(_) => {
                return Err(LagunaNormalizationError::ExpectedObject {
                    field_name: "text_config".to_owned(),
                });
            }
            None => None,
        };

        // Nested language geometry is authoritative. Root inheritance is limited to
        // family identity, architecture metadata, dtype, and quantization documents.
        let mut merged_fields = nested_fields
            .cloned()
            .unwrap_or_else(|| root_fields.clone());
        if nested_fields.is_some() {
            for (field_name, root_field_value) in root_fields {
                if field_name == "text_config" || is_quantization_field(field_name) {
                    continue;
                }
                match merged_fields.get(field_name) {
                    Some(nested_field_value)
                        if is_normalized_field(field_name)
                            && !semantically_equivalent(
                                field_name,
                                root_field_value,
                                nested_field_value,
                            ) =>
                    {
                        return Err(LagunaNormalizationError::ConflictingEnvelopeField {
                            field_name: field_name.clone(),
                        });
                    }
                    Some(_) => {}
                    None if is_inherited_root_metadata(field_name) => {
                        let _previous_inherited_value =
                            merged_fields.insert(field_name.clone(), root_field_value.clone());
                    }
                    None => {}
                }
            }
        }
        let _text_config = merged_fields.remove("text_config");
        let _quantization = merged_fields.remove("quantization");
        let _quantization_config = merged_fields.remove("quantization_config");

        // Every copy is normalized separately so wrappers and omitted affine modes compare canonically.
        let mut quantization_documents = Vec::new();
        collect_quantization_documents(root_fields, &mut quantization_documents);
        if let Some(nested_fields) = nested_fields {
            collect_quantization_documents(nested_fields, &mut quantization_documents);
        }
        Ok(Self {
            fields: merged_fields,
            quantization_documents,
        })
    }

    pub(super) fn field(&self, field_name: &str) -> Option<&Value> {
        self.fields.get(field_name)
    }

    pub(super) fn required_u32(
        &self,
        field_name: &str,
        maximum_value: u32,
    ) -> Result<u32, LagunaNormalizationError> {
        self.optional_u32(field_name, maximum_value)?
            .ok_or_else(|| LagunaNormalizationError::MissingRequiredField {
                field_name: field_name.to_owned(),
            })
    }

    pub(super) fn optional_u32(
        &self,
        field_name: &str,
        maximum_value: u32,
    ) -> Result<Option<u32>, LagunaNormalizationError> {
        let Some(field_value) = self.field(field_name) else {
            return Ok(None);
        };
        let unsigned_value = field_value
            .as_u64()
            .ok_or_else(|| invalid_numeric(field_name, "must be a positive integer"))?;
        let bounded_value = u32::try_from(unsigned_value)
            .map_err(|_| invalid_numeric(field_name, "exceeds the supported integer range"))?;
        if bounded_value == 0 || bounded_value > maximum_value {
            return Err(invalid_numeric(
                field_name,
                "must be positive and within the supported safety bound",
            ));
        }
        Ok(Some(bounded_value))
    }

    pub(super) fn optional_non_negative_u32(
        &self,
        field_name: &str,
        maximum_value: u32,
    ) -> Result<Option<u32>, LagunaNormalizationError> {
        let Some(field_value) = self.field(field_name) else {
            return Ok(None);
        };
        let unsigned_value = field_value
            .as_u64()
            .ok_or_else(|| invalid_numeric(field_name, "must be a non-negative integer"))?;
        let bounded_value = u32::try_from(unsigned_value)
            .map_err(|_| invalid_numeric(field_name, "exceeds the supported integer range"))?;
        if bounded_value > maximum_value {
            return Err(invalid_numeric(
                field_name,
                "exceeds the supported safety bound",
            ));
        }
        Ok(Some(bounded_value))
    }

    pub(super) fn required_f64(&self, field_name: &str) -> Result<f64, LagunaNormalizationError> {
        self.optional_f64(field_name)?.ok_or_else(|| {
            LagunaNormalizationError::MissingRequiredField {
                field_name: field_name.to_owned(),
            }
        })
    }

    pub(super) fn optional_f64(
        &self,
        field_name: &str,
    ) -> Result<Option<f64>, LagunaNormalizationError> {
        let Some(field_value) = self.field(field_name) else {
            return Ok(None);
        };
        let numeric_value = field_value
            .as_f64()
            .ok_or_else(|| invalid_numeric(field_name, "must be a finite JSON number"))?;
        if !numeric_value.is_finite() {
            return Err(invalid_numeric(field_name, "must be finite"));
        }
        Ok(Some(numeric_value))
    }

    pub(super) fn optional_bool(
        &self,
        field_name: &str,
        default_value: bool,
    ) -> Result<bool, LagunaNormalizationError> {
        match self.field(field_name) {
            None => Ok(default_value),
            Some(Value::Bool(boolean_value)) => Ok(*boolean_value),
            Some(actual_value) => Err(LagunaNormalizationError::UnsupportedValue {
                field_name: field_name.to_owned(),
                actual_value: bounded_value_label(actual_value),
            }),
        }
    }

    pub(super) fn required_string(
        &self,
        field_name: &str,
    ) -> Result<&str, LagunaNormalizationError> {
        match self.field(field_name) {
            Some(Value::String(string_value)) => Ok(string_value),
            Some(actual_value) => Err(LagunaNormalizationError::UnsupportedValue {
                field_name: field_name.to_owned(),
                actual_value: bounded_value_label(actual_value),
            }),
            None => Err(LagunaNormalizationError::MissingRequiredField {
                field_name: field_name.to_owned(),
            }),
        }
    }

    pub(super) fn quantization_documents(&self) -> &[Value] {
        &self.quantization_documents
    }
}

fn collect_quantization_documents(fields: &Map<String, Value>, documents: &mut Vec<Value>) {
    for field_name in ["quantization", "quantization_config"] {
        if let Some(document) = fields.get(field_name) {
            documents.push(document.clone());
        }
    }
}

fn is_quantization_field(field_name: &str) -> bool {
    matches!(field_name, "quantization" | "quantization_config")
}

fn is_inherited_root_metadata(field_name: &str) -> bool {
    matches!(
        field_name,
        "model_type"
            | "architectures"
            | "torch_dtype"
            | "qkv_bias"
            | "hidden_act"
            | "rope_style"
            | "swa_attention_sink_enabled"
            | "use_bidirectional_attention"
            | "moe_router_use_sigmoid"
    )
}

fn is_normalized_field(field_name: &str) -> bool {
    matches!(
        field_name,
        "model_type"
            | "architectures"
            | "torch_dtype"
            | "vocab_size"
            | "hidden_size"
            | "intermediate_size"
            | "num_hidden_layers"
            | "num_attention_heads"
            | "num_attention_heads_per_layer"
            | "num_key_value_heads"
            | "head_dim"
            | "max_position_embeddings"
            | "sliding_window"
            | "attention_bias"
            | "qkv_bias"
            | "hidden_act"
            | "attention_dropout"
            | "rms_norm_eps"
            | "tie_word_embeddings"
            | "use_cache"
            | "layer_types"
            | "gating"
            | "gating_types"
            | "mlp_layer_types"
            | "num_experts"
            | "num_experts_per_tok"
            | "moe_intermediate_size"
            | "shared_expert_intermediate_size"
            | "norm_topk_prob"
            | "decoder_sparse_step"
            | "mlp_only_layers"
            | "scoring_func"
            | "moe_routed_scaling_factor"
            | "moe_apply_router_weight_on_input"
            | "moe_router_logit_softcapping"
            | "moe_router_use_sigmoid"
            | "use_bidirectional_attention"
            | "rope_style"
            | "swa_attention_sink_enabled"
            | "rope_parameters"
            | "swa_rope_parameters"
            | "rope_scaling"
            | "rope_theta"
            | "partial_rotary_factor"
    )
}

fn semantically_equivalent(field_name: &str, first: &Value, second: &Value) -> bool {
    canonical_alias_value(field_name, first) == canonical_alias_value(field_name, second)
}

fn canonical_alias_value(field_name: &str, source_value: &Value) -> Value {
    match source_value {
        Value::Bool(false) if matches!(field_name, "gating" | "gating_types") => {
            Value::String("none".to_owned())
        }
        Value::String(string_value) => {
            Value::String(canonical_alias_string(field_name, string_value))
        }
        Value::Array(source_values) => Value::Array(
            source_values
                .iter()
                .map(|source_value| canonical_alias_value(field_name, source_value))
                .collect(),
        ),
        Value::Object(source_fields) => {
            let mut canonical_fields = Map::new();
            for (source_name, source_value) in source_fields {
                let canonical_name = if source_name == "type" {
                    "rope_type"
                } else {
                    source_name
                };
                let _previous_alias_value = canonical_fields.insert(
                    canonical_name.to_owned(),
                    canonical_alias_value(canonical_name, source_value),
                );
            }
            Value::Object(canonical_fields)
        }
        scalar_value => scalar_value.clone(),
    }
}

fn canonical_alias_string(field_name: &str, source_value: &str) -> String {
    let underscored_value = source_value.replace('-', "_");
    match (field_name, underscored_value.as_str()) {
        ("layer_types", "full_attention") => "full".to_owned(),
        ("layer_types", "sliding_attention") => "sliding".to_owned(),
        ("mlp_layer_types", "moe") => "sparse".to_owned(),
        ("gating" | "gating_types", "false") => "none".to_owned(),
        ("torch_dtype", "bf16") => "bfloat16".to_owned(),
        ("torch_dtype", "fp16" | "float16_t") => "float16".to_owned(),
        ("torch_dtype", "fp32") => "float32".to_owned(),
        _ => underscored_value,
    }
}

pub(super) fn invalid_numeric(
    field_name: &str,
    description: &'static str,
) -> LagunaNormalizationError {
    LagunaNormalizationError::InvalidNumericValue {
        field_name: field_name.to_owned(),
        description,
    }
}

/// Keeps a rejected config subtree actionable without echoing the complete user document.
pub(super) fn bounded_value_label(field_value: &Value) -> String {
    let unbounded_label = field_value
        .as_str()
        .map(str::to_owned)
        .unwrap_or_else(|| field_value.to_string());
    bounded_text_label(&unbounded_label)
}

pub(super) fn bounded_text_label(unbounded_label: &str) -> String {
    let mut label_characters = unbounded_label.chars();
    let bounded_label = label_characters
        .by_ref()
        .take(MAXIMUM_ERROR_VALUE_CHARACTERS)
        .collect::<String>();
    if label_characters.next().is_some() {
        format!("{bounded_label}…")
    } else {
        bounded_label
    }
}
