//! The folded-ternary decoder-cache layout contract for persistent prompt
//! storage.
//!
//! Dtypes come from the pack's own declared execution contract rather than a
//! probe: every packed-module computation runs in `float16`
//! (`FOLDED_MODULE_DTYPE`), so full-attention key/value slabs and the linear
//! convolution rolling buffer persist as Float16, while the gated-delta
//! recurrent accumulator is pinned to `float32` by the pack schema
//! (`FOLDED_SSM_DTYPE`) exactly as the live GDN kernels require. Pinning the
//! storage contract to the schema the pack already enforces keeps a future
//! dtype change from silently producing unreadable cache files.

use crate::{
    DecoderCacheLayerLayout, DecoderCacheLayout, DecoderCacheLayoutError, DecoderCacheTensorDtype,
    DecoderCacheTensorLayout,
};

use super::configuration::FoldedPackConfig;

pub(crate) const PRISM_CONVOLUTION_TENSOR_ROLE: &str = "linear.convolution";
pub(crate) const PRISM_RECURRENCE_TENSOR_ROLE: &str = "linear.gated_delta_recurrent";
pub(crate) const PRISM_ATTENTION_KEYS_TENSOR_ROLE: &str = "attention.keys";
pub(crate) const PRISM_ATTENTION_VALUES_TENSOR_ROLE: &str = "attention.values";

/// The KV slab growth granularity the straight forward pass allocates with.
pub(crate) const PRISM_FULL_ATTENTION_KEY_VALUE_GROWTH_TOKENS: usize = 256;

/// Builds the validated persistent decoder-cache layout for one folded pack.
///
/// Layer ordering follows `layer_types` so persisted tensor names
/// (`layer_<decoder_layer_index>_<role>`) use decoder-layer indices, while the
/// live `StraightGenerationState` slots number layers within their attention
/// kind; the mapping between the two is rebuilt from the same list.
pub fn prism_hadamard_qwen35_decoder_cache_layout(
    folded_pack_config: &FoldedPackConfig,
) -> Result<DecoderCacheLayout, DecoderCacheLayoutError> {
    let text = folded_pack_config.text();
    let full_attention_key_value_dimensions = vec![
        1,
        text.num_key_value_heads as usize,
        0,
        text.head_dim as usize,
    ];
    let linear_convolution_dimensions = vec![
        1,
        (text.linear_conv_kernel_dim as usize).saturating_sub(1),
        (2 * text.linear_num_key_heads * text.linear_key_head_dim
            + text.linear_num_value_heads * text.linear_value_head_dim) as usize,
    ];
    let linear_recurrent_dimensions = vec![
        1,
        text.linear_num_value_heads as usize,
        text.linear_value_head_dim as usize,
        text.linear_key_head_dim as usize,
    ];
    let decoder_layer_layouts = text
        .layer_types
        .iter()
        .enumerate()
        .map(
            |(decoder_layer_index, layer_type)| match layer_type.as_str() {
                "full_attention" => Ok(DecoderCacheLayerLayout::append_only_attention(
                    DecoderCacheTensorLayout::sequence(
                        PRISM_ATTENTION_KEYS_TENSOR_ROLE,
                        DecoderCacheTensorDtype::Float16,
                        full_attention_key_value_dimensions.clone(),
                        2,
                    ),
                    DecoderCacheTensorLayout::sequence(
                        PRISM_ATTENTION_VALUES_TENSOR_ROLE,
                        DecoderCacheTensorDtype::Float16,
                        full_attention_key_value_dimensions.clone(),
                        2,
                    ),
                    PRISM_FULL_ATTENTION_KEY_VALUE_GROWTH_TOKENS,
                )),
                "linear_attention" => Ok(DecoderCacheLayerLayout::composite(vec![
                    DecoderCacheLayerLayout::recurrent_tensor(DecoderCacheTensorLayout::fixed(
                        PRISM_CONVOLUTION_TENSOR_ROLE,
                        DecoderCacheTensorDtype::Float16,
                        linear_convolution_dimensions.clone(),
                    )),
                    DecoderCacheLayerLayout::recurrent_tensor(DecoderCacheTensorLayout::fixed(
                        PRISM_RECURRENCE_TENSOR_ROLE,
                        DecoderCacheTensorDtype::Float32,
                        linear_recurrent_dimensions.clone(),
                    )),
                ])),
                // `validated_layer_types` already refused anything else, so this
                // arm only exists to keep the match exhaustive.
                _other_layer_type => {
                    Err(DecoderCacheLayoutError::ExecutionDtypeLayerFamilyMismatch {
                        layer_index: decoder_layer_index,
                    })
                }
            },
        )
        .collect::<Result<Vec<_>, _>>()?;
    DecoderCacheLayout::new(decoder_layer_layouts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// A tiny pack document shaped exactly like the shipped 27B pack (same
    /// keys and wire conventions) with configurable layer kinds.
    fn shaped_pack_config(layer_types: &[&str]) -> FoldedPackConfig {
        let document = json!({
            "schema_version": 2,
            "model_type": "prism_hadamard_qwen35",
            "base_model_type": "qwen3_5",
            "requires_runtime": "runtime/artifact.py",
            "hadamard_config": "hadamard.json",
            "tensor_namespace": "mlx-vlm-qwen3_5",
            "gdn_activation_layout": "grouped",
            "components": {"text": true, "vision": false, "mtp": false},
            "text_config": {
                "model_type": "qwen3_5_text",
                "hidden_act": "silu",
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_hidden_layers": layer_types.len(),
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 16,
                "vocab_size": 256,
                "max_position_embeddings": 4096,
                "full_attention_interval": 4,
                "layer_types": layer_types,
                "linear_num_value_heads": 8,
                "linear_num_key_heads": 4,
                "linear_key_head_dim": 16,
                "linear_value_head_dim": 16,
                "linear_conv_kernel_dim": 4,
                "rms_norm_eps": 0.000001,
                "rope_parameters": {
                    "type": "default",
                    "rope_theta": 10000000.0,
                    "partial_rotary_factor": 0.25,
                    "mrope_interleaved": true,
                    "mrope_section": [5, 5, 6],
                },
                "eos_token_id": 248044,
                "attention_bias": false,
                "tie_word_embeddings": false,
                "attn_output_gate": true,
                "output_gate_type": "swish",
                "mamba_ssm_dtype": "float32",
            },
            "quantization": {"bits": 2, "group_size": 128, "mode": "affine"},
            "modules": [
                {"path": "lm_head", "block": 1024, "embedding": false, "dtype": "float16"},
                {"path": "model.embed_tokens", "block": 1024, "embedding": true, "dtype": "float16"},
                {"path": "model.layers.0.mlp.down_proj", "block": 1024, "embedding": false, "dtype": "float16"},
            ],
            "tie_word_embeddings": false,
            "vision_config": {"model_type": "qwen3_5", "hidden_size": 64},
            "image_token_id": 151164,
            "video_token_id": 151165,
            "vision_start_token_id": 151161,
            "vision_end_token_id": 151162,
        });
        FoldedPackConfig::from_json_bytes(document.to_string().as_bytes())
            .expect("the fixture document must parse and validate")
    }

    #[test]
    fn should_declare_sequence_state_only_for_full_attention_layers() {
        let decoder_cache_layout =
            prism_hadamard_qwen35_decoder_cache_layout(&shaped_pack_config(&[
                "linear_attention",
                "linear_attention",
                "full_attention",
            ]))
            .expect("the fixture layout must validate");
        assert_eq!(decoder_cache_layout.layer_count(), 3);
        assert_eq!(decoder_cache_layout.sequence_tensor_count(), 2);
        // One convolution plus one gated-delta recurrent tensor per linear layer.
        assert_eq!(decoder_cache_layout.boundary_tensor_count(), 4);
    }

    #[test]
    fn should_name_persisted_tensors_by_decoder_layer_index() {
        let decoder_cache_layout =
            prism_hadamard_qwen35_decoder_cache_layout(&shaped_pack_config(&[
                "linear_attention",
                "linear_attention",
                "full_attention",
            ]))
            .expect("the fixture layout must validate");
        let full_attention_layer = decoder_cache_layout
            .layer(2)
            .expect("the fixture has a full-attention layer at index 2");
        let persisted_tensor_names = match full_attention_layer {
            DecoderCacheLayerLayout::AppendOnlyAttention { keys, values, .. } => {
                // `persistent_tensor_name` composes `layer_<decoder_layer_index>_
                // <tensor_role_name>`; assert the role placement per layer index.
                vec![
                    format!("layer_2_{}", keys.tensor_role_name()),
                    format!("layer_2_{}", values.tensor_role_name()),
                ]
            }
            _ => panic!("the full-attention layer must be append-only attention"),
        };
        assert_eq!(
            persisted_tensor_names,
            vec!["layer_2_attention.keys", "layer_2_attention.values"]
        );
    }

    #[test]
    fn should_name_linear_boundary_tensors_by_decoder_layer_index() {
        let decoder_cache_layout =
            prism_hadamard_qwen35_decoder_cache_layout(&shaped_pack_config(&[
                "linear_attention",
                "linear_attention",
                "full_attention",
            ]))
            .expect("the fixture layout must validate");
        let linear_layer = decoder_cache_layout
            .layer(0)
            .expect("the fixture has a linear layer at index 0");
        let persisted_tensor_names = match linear_layer {
            DecoderCacheLayerLayout::Composite { components } => components
                .iter()
                .flat_map(|component| match component {
                    DecoderCacheLayerLayout::RecurrentTensor { tensor } => {
                        vec![format!("layer_0_{}", tensor.tensor_role_name())]
                    }
                    _ => panic!("the linear layer composite holds only recurrent tensors"),
                })
                .collect::<Vec<_>>(),
            _ => panic!("the linear layer must be composite"),
        };
        assert_eq!(
            persisted_tensor_names,
            vec![
                "layer_0_linear.convolution",
                "layer_0_linear.gated_delta_recurrent"
            ]
        );
    }
}
