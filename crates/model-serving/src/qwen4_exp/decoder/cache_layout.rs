//! Per-request decoder state for `qwen4_exp`.
//!
//! One request owns four structurally different state streams, and this
//! module is where they are named, laid out, and measured together:
//!
//! 1. Gated-delta recurrent plus convolution state on the linear-attention
//!    layers, reusing the architecture-neutral composite state kind.
//! 2. Append-only key-value state on the full-attention layers.
//! 3. The sparse-attention index-key cache on the same full-attention
//!    layers — an auxiliary key/value pair at the indexer's geometry, kept
//!    beside the main key-value state so the indexer can score without
//!    rereading the main cache.
//! 4. Hyper-connection residual streams, which are transient per-forward
//!    activations rather than persisted state; they are measured as request
//!    workspace, never as decoder cache.
//!
//! Geometry comes only from validated configuration. The live execution
//! dtypes arrive separately from the bound graph, exactly like the Qwen3.5
//! family, so a nominal activation dtype cannot silently narrow persistent
//! state.

use crate::{
    DecoderCacheLayerLayout, DecoderCacheLayout, DecoderCacheLayoutError, DecoderCacheTensorDtype,
    DecoderCacheTensorLayout,
};

use crate::qwen4_exp::configuration::{Qwen4ExpConfig, Qwen4ExpLayerKind};

pub(crate) const QWEN4_EXP_CONVOLUTION_TENSOR_ROLE: &str = "linear.convolution";
pub(crate) const QWEN4_EXP_RECURRENCE_TENSOR_ROLE: &str = "linear.gated_delta_recurrent";
pub(crate) const QWEN4_EXP_ATTENTION_KEYS_TENSOR_ROLE: &str = "attention.keys";
pub(crate) const QWEN4_EXP_ATTENTION_VALUES_TENSOR_ROLE: &str = "attention.values";
pub(crate) const QWEN4_EXP_INDEX_KEYS_TENSOR_ROLE: &str = "attention.index_keys";
pub(crate) const QWEN4_EXP_INDEX_VALUES_TENSOR_ROLE: &str = "attention.index_values";

/// Exact live execution dtypes for one `qwen4_exp` decoder layer's state.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Qwen4ExpDecoderLayerCacheDtypes {
    LinearAttention {
        convolution: DecoderCacheTensorDtype,
    },
    FullAttention {
        keys: DecoderCacheTensorDtype,
        values: DecoderCacheTensorDtype,
        index_keys: DecoderCacheTensorDtype,
        index_values: DecoderCacheTensorDtype,
    },
}

/// Combines model geometry with dtypes derived from the bound execution graph.
///
/// # Errors
/// When the dtype slice does not cover every declared layer, a dimension
/// exceeds the layout arithmetic range, a required subsystem group is absent
/// from the configuration, or a layer kind disagrees with its dtype variant.
pub fn qwen4_exp_decoder_cache_layout(
    config: &Qwen4ExpConfig,
    full_attention_key_value_growth_tokens: usize,
    decoder_layer_cache_dtypes: &[Qwen4ExpDecoderLayerCacheDtypes],
) -> Result<DecoderCacheLayout, DecoderCacheLayoutError> {
    let decoder_layer_count = config.decoder_layers as usize;
    if decoder_layer_cache_dtypes.len() != decoder_layer_count {
        return Err(DecoderCacheLayoutError::ExecutionDtypeLayerCountMismatch {
            expected_layer_count: decoder_layer_count,
            actual_layer_count: decoder_layer_cache_dtypes.len(),
        });
    }
    let linear_attention = config.linear_attention.ok_or(
        DecoderCacheLayoutError::ModelConfigurationDimensionOutsideUsizeRange {
            dimension_name: "linear attention geometry",
        },
    )?;
    let indexer = config.sparse_attention.ok_or(
        DecoderCacheLayoutError::ModelConfigurationDimensionOutsideUsizeRange {
            dimension_name: "sparse-attention indexer geometry",
        },
    )?;
    let linear_convolution_state_dimension = usize::try_from(
        linear_attention.key_heads * linear_attention.key_head_dim
            + linear_attention.value_heads * linear_attention.value_head_dim,
    )
    .map_err(
        |_| DecoderCacheLayoutError::ModelConfigurationDimensionOutsideUsizeRange {
            dimension_name: "linear convolution state",
        },
    )?;
    let full_attention_key_value_dimensions = vec![
        1,
        config.key_value_heads as usize,
        0,
        config.head_dim as usize,
    ];
    let index_key_value_dimensions = vec![
        1,
        indexer.key_value_heads as usize,
        0,
        indexer.head_dim as usize,
    ];
    let linear_convolution_dimensions = vec![
        1,
        (linear_attention.conv_kernel_dim as usize).saturating_sub(1),
        linear_convolution_state_dimension,
    ];
    let linear_recurrent_dimensions = vec![
        1,
        linear_attention.value_heads as usize,
        linear_attention.value_head_dim as usize,
        linear_attention.key_head_dim as usize,
    ];
    let decoder_layer_layouts = decoder_layer_cache_dtypes
        .iter()
        .zip(config.layer_schedule.iter())
        .enumerate()
        .map(|(layer_index, (decoder_layer_cache_dtypes, layer_kind))| {
            match (layer_kind, decoder_layer_cache_dtypes) {
                (
                    Qwen4ExpLayerKind::FullAttention,
                    Qwen4ExpDecoderLayerCacheDtypes::FullAttention {
                        keys,
                        values,
                        index_keys,
                        index_values,
                    },
                ) => Ok(DecoderCacheLayerLayout::composite(vec![
                    DecoderCacheLayerLayout::append_only_attention(
                        DecoderCacheTensorLayout::sequence(
                            QWEN4_EXP_ATTENTION_KEYS_TENSOR_ROLE,
                            *keys,
                            full_attention_key_value_dimensions.clone(),
                            2,
                        ),
                        DecoderCacheTensorLayout::sequence(
                            QWEN4_EXP_ATTENTION_VALUES_TENSOR_ROLE,
                            *values,
                            full_attention_key_value_dimensions.clone(),
                            2,
                        ),
                        full_attention_key_value_growth_tokens,
                    ),
                    DecoderCacheLayerLayout::append_only_attention(
                        DecoderCacheTensorLayout::sequence(
                            QWEN4_EXP_INDEX_KEYS_TENSOR_ROLE,
                            *index_keys,
                            index_key_value_dimensions.clone(),
                            2,
                        ),
                        DecoderCacheTensorLayout::sequence(
                            QWEN4_EXP_INDEX_VALUES_TENSOR_ROLE,
                            *index_values,
                            index_key_value_dimensions.clone(),
                            2,
                        ),
                        full_attention_key_value_growth_tokens,
                    ),
                ])),
                (
                    Qwen4ExpLayerKind::LinearAttention,
                    Qwen4ExpDecoderLayerCacheDtypes::LinearAttention { convolution },
                ) => Ok(DecoderCacheLayerLayout::composite(vec![
                    DecoderCacheLayerLayout::recurrent_tensor(DecoderCacheTensorLayout::fixed(
                        QWEN4_EXP_CONVOLUTION_TENSOR_ROLE,
                        *convolution,
                        linear_convolution_dimensions.clone(),
                    )),
                    DecoderCacheLayerLayout::recurrent_tensor(DecoderCacheTensorLayout::fixed(
                        // The gated-delta recurrent accumulator stays in the
                        // configured state dtype, independently from the
                        // activation dtype above.
                        QWEN4_EXP_RECURRENCE_TENSOR_ROLE,
                        DecoderCacheTensorDtype::Float32,
                        linear_recurrent_dimensions.clone(),
                    )),
                ])),
                _ => {
                    Err(DecoderCacheLayoutError::ExecutionDtypeLayerFamilyMismatch { layer_index })
                }
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    DecoderCacheLayout::new(decoder_layer_layouts)
}
