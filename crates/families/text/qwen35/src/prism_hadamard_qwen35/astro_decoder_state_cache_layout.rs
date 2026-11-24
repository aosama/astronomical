//! The folded-ternary astro decoder state cache layout contract for persistent prompt
//! storage.
//!
//! Dtypes come from the pack's own declared execution contract rather than a
//! probe: every packed-module computation runs in `float16`
//! (`FOLDED_MODULE_DTYPE`), so full-attention key/value slabs and the linear
//! convolution rolling buffer persist as Float16, while the gated-delta
//! recurrent accumulator is pinned to `float32` by the pack schema
//! (`FOLDED_SSM_DTYPE`) exactly as the live GDN kernels require. Pinning the
//! storage contract to the schema the pack already enforces keeps a future
//! dtype change from silently producing unreadable astro decoder state cache files.

use runtime_integration::MlxRuntimeError;

use crate::{
    AstroDecoderStateCacheLayerLayout, AstroDecoderStateCacheLayout,
    AstroDecoderStateCacheLayoutError, AstroDecoderStateCacheTensorDtype,
    AstroDecoderStateCacheTensorLayout, RequestDecoderStateStack,
};

use super::configuration::FoldedPackConfig;

pub const PRISM_CONVOLUTION_TENSOR_ROLE: &str = "linear.convolution";
pub const PRISM_RECURRENCE_TENSOR_ROLE: &str = "linear.gated_delta_recurrent";
pub const PRISM_ATTENTION_KEYS_TENSOR_ROLE: &str = "attention.keys";
pub const PRISM_ATTENTION_VALUES_TENSOR_ROLE: &str = "attention.values";

/// The KV slab growth granularity the straight forward pass allocates with.
pub const PRISM_FULL_ATTENTION_KEY_VALUE_GROWTH_TOKENS: usize = 256;

/// Creates one request's live `astro_decoder_state_cache` owners, one per
/// decoder layer, from the validated layout.
///
/// The growth step is pinned through the shared stack's override rather than
/// read back from the layout, so the KV slabs grow in exactly the 256-token
/// steps Prism has always allocated: a slab's capacity decides when MLX
/// reallocates and copies it, and a different step would move those copies
/// and change decode speed even though the attended tokens stay the same.
pub fn prism_request_decoder_state(
    astro_decoder_state_cache_layout: &AstroDecoderStateCacheLayout,
) -> Result<RequestDecoderStateStack, MlxRuntimeError> {
    RequestDecoderStateStack::empty_from_astro_decoder_state_cache_layout_with_full_attention_kv_state_growth_tokens(
        astro_decoder_state_cache_layout,
        PRISM_FULL_ATTENTION_KEY_VALUE_GROWTH_TOKENS as i32,
    )
}

/// Builds the validated persistent astro decoder state cache layout for one folded pack.
///
/// Layer ordering follows `layer_types`, so persisted tensor names
/// (`layer_<decoder_layer_index>_<role>`) and the live
/// `RequestDecoderStateStack` both address state by the same decoder-layer
/// index.
pub fn prism_hadamard_qwen35_astro_decoder_state_cache_layout(
    folded_pack_config: &FoldedPackConfig,
) -> Result<AstroDecoderStateCacheLayout, AstroDecoderStateCacheLayoutError> {
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
                "full_attention" => Ok(AstroDecoderStateCacheLayerLayout::append_only_attention(
                    AstroDecoderStateCacheTensorLayout::sequence(
                        PRISM_ATTENTION_KEYS_TENSOR_ROLE,
                        AstroDecoderStateCacheTensorDtype::Float16,
                        full_attention_key_value_dimensions.clone(),
                        2,
                    ),
                    AstroDecoderStateCacheTensorLayout::sequence(
                        PRISM_ATTENTION_VALUES_TENSOR_ROLE,
                        AstroDecoderStateCacheTensorDtype::Float16,
                        full_attention_key_value_dimensions.clone(),
                        2,
                    ),
                    PRISM_FULL_ATTENTION_KEY_VALUE_GROWTH_TOKENS,
                )),
                "linear_attention" => Ok(AstroDecoderStateCacheLayerLayout::composite(vec![
                    AstroDecoderStateCacheLayerLayout::recurrent_tensor(
                        AstroDecoderStateCacheTensorLayout::fixed(
                            PRISM_CONVOLUTION_TENSOR_ROLE,
                            AstroDecoderStateCacheTensorDtype::Float16,
                            linear_convolution_dimensions.clone(),
                        ),
                    ),
                    AstroDecoderStateCacheLayerLayout::recurrent_tensor(
                        AstroDecoderStateCacheTensorLayout::fixed(
                            PRISM_RECURRENCE_TENSOR_ROLE,
                            AstroDecoderStateCacheTensorDtype::Float32,
                            linear_recurrent_dimensions.clone(),
                        ),
                    ),
                ])),
                // `validated_layer_types` already refused anything else, so this
                // arm only exists to keep the match exhaustive.
                _other_layer_type => Err(
                    AstroDecoderStateCacheLayoutError::ExecutionDtypeLayerFamilyMismatch {
                        layer_index: decoder_layer_index,
                    },
                ),
            },
        )
        .collect::<Result<Vec<_>, _>>()?;
    AstroDecoderStateCacheLayout::new(decoder_layer_layouts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AstroDecoderStateCacheState;
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
        let astro_decoder_state_cache_layout =
            prism_hadamard_qwen35_astro_decoder_state_cache_layout(&shaped_pack_config(&[
                "linear_attention",
                "linear_attention",
                "full_attention",
            ]))
            .expect("the fixture layout must validate");
        assert_eq!(astro_decoder_state_cache_layout.layer_count(), 3);
        assert_eq!(astro_decoder_state_cache_layout.sequence_tensor_count(), 2);
        // One convolution plus one gated-delta recurrent tensor per linear layer.
        assert_eq!(astro_decoder_state_cache_layout.boundary_tensor_count(), 4);
    }

    #[test]
    fn should_name_persisted_tensors_by_decoder_layer_index() {
        let astro_decoder_state_cache_layout =
            prism_hadamard_qwen35_astro_decoder_state_cache_layout(&shaped_pack_config(&[
                "linear_attention",
                "linear_attention",
                "full_attention",
            ]))
            .expect("the fixture layout must validate");
        let full_attention_layer = astro_decoder_state_cache_layout
            .layer(2)
            .expect("the fixture has a full-attention layer at index 2");
        let persisted_tensor_names = match full_attention_layer {
            AstroDecoderStateCacheLayerLayout::AppendOnlyAttention { keys, values, .. } => {
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
        let astro_decoder_state_cache_layout =
            prism_hadamard_qwen35_astro_decoder_state_cache_layout(&shaped_pack_config(&[
                "linear_attention",
                "linear_attention",
                "full_attention",
            ]))
            .expect("the fixture layout must validate");
        let linear_layer = astro_decoder_state_cache_layout
            .layer(0)
            .expect("the fixture has a linear layer at index 0");
        let persisted_tensor_names = match linear_layer {
            AstroDecoderStateCacheLayerLayout::Composite { components } => components
                .iter()
                .flat_map(|component| match component {
                    AstroDecoderStateCacheLayerLayout::RecurrentTensor { tensor } => {
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

    #[test]
    fn should_build_one_live_state_owner_per_decoder_layer_in_layer_order() {
        let astro_decoder_state_cache_layout =
            prism_hadamard_qwen35_astro_decoder_state_cache_layout(&shaped_pack_config(&[
                "linear_attention",
                "full_attention",
                "linear_attention",
            ]))
            .expect("the fixture layout must validate");
        let request_decoder_state = prism_request_decoder_state(&astro_decoder_state_cache_layout)
            .expect("the validated layout must create live state");
        let layer_families = request_decoder_state
            .layers()
            .iter()
            .map(live_state_family_name)
            .collect::<Vec<_>>();
        assert_eq!(
            layer_families,
            vec!["composite", "append_only_attention", "composite"]
        );
        // No tensor is allocated before the first forward, so the context
        // share of the memory breakdown starts at zero.
        assert_eq!(request_decoder_state.payload_byte_count(), 0);
    }

    #[test]
    fn should_grow_full_attention_slabs_in_prism_steps_of_256_tokens() {
        let astro_decoder_state_cache_layout =
            prism_hadamard_qwen35_astro_decoder_state_cache_layout(&shaped_pack_config(&[
                "full_attention",
            ]))
            .expect("the fixture layout must validate");
        let request_decoder_state = prism_request_decoder_state(&astro_decoder_state_cache_layout)
            .expect("the validated layout must create live state");
        let Some(AstroDecoderStateCacheState::AppendOnlyAttention { attention }) =
            request_decoder_state.layer(0)
        else {
            panic!("the only layer must be append-only attention");
        };
        // One token and exactly one step both fit the first slab; one token
        // past it needs a second whole slab.
        for (update_token_count, expected_growth_tokens) in [(1, 256), (256, 256), (257, 512)] {
            assert_eq!(
                attention
                    .projected_capacity_growth_tokens(update_token_count)
                    .expect("the projection must succeed"),
                expected_growth_tokens,
                "an empty slab receiving {update_token_count} tokens"
            );
        }
    }

    #[test]
    fn should_refuse_a_layer_layout_the_qwen_state_owners_cannot_hold() {
        // A recurrent tensor outside a composite layer is valid layout
        // vocabulary, but no Qwen decoder layer owns a lone recurrent tensor.
        let standalone_recurrent_layout = AstroDecoderStateCacheLayout::new(vec![
            AstroDecoderStateCacheLayerLayout::recurrent_tensor(
                AstroDecoderStateCacheTensorLayout::fixed(
                    PRISM_RECURRENCE_TENSOR_ROLE,
                    AstroDecoderStateCacheTensorDtype::Float32,
                    vec![1, 8, 16, 16],
                ),
            ),
        ])
        .expect("a standalone recurrent tensor layout is valid on its own");
        assert!(prism_request_decoder_state(&standalone_recurrent_layout).is_err());
    }

    fn live_state_family_name(layer_state: &AstroDecoderStateCacheState) -> &'static str {
        match layer_state {
            AstroDecoderStateCacheState::AppendOnlyAttention { .. } => "append_only_attention",
            AstroDecoderStateCacheState::Composite { .. } => "composite",
        }
    }
}
