//! Per-stream state geometry for `qwen4_exp`, measured in bytes.
//!
//! The family measures; `memory/` decides. This owner turns validated
//! configuration into the per-token byte facts each state stream contributes,
//! so admission projects real totals instead of guesses and the utilization
//! split can attribute persistent state through the existing context-state
//! category. Hyper-connection streams are transient per-forward activations:
//! they are measured here as request workspace so nothing silently vanishes
//! from the accounting, and they never become persisted decoder state.

use super::cache_layout::{
    QWEN4_EXP_INDEX_KEYS_TENSOR_ROLE, Qwen4ExpDecoderLayerCacheDtypes,
    qwen4_exp_decoder_cache_layout,
};
use crate::qwen4_exp::configuration::{Qwen4ExpConfig, Qwen4ExpLayerKind};

/// Byte facts for one request's state streams, derived from configuration.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Qwen4ExpStateGeometry {
    /// Per-token bytes of the gated-delta convolution plus recurrent state
    /// across every linear-attention layer. Fixed state, paid once.
    pub linear_attention_fixed_bytes: u64,
    /// Per-token bytes of the main key-value state across every
    /// full-attention layer.
    pub full_attention_bytes_per_token: u64,
    /// Per-token bytes of the sparse-attention index-key cache across every
    /// full-attention layer.
    pub index_key_cache_bytes_per_token: u64,
    /// Transient hyper-connection stream workspace for one forward, across
    /// every layer boundary. Never persisted.
    pub hyper_connection_stream_workspace_bytes: u64,
    /// Declared full-attention layer count, for callers that project growth
    /// per layer.
    pub full_attention_layers: u32,
    /// Declared linear-attention layer count.
    pub linear_attention_layers: u32,
}

impl Qwen4ExpStateGeometry {
    /// Measures the four streams from validated configuration and dtypes.
    ///
    /// # Errors
    /// When the layout arithmetic rejects a dimension or the dtype slice does
    /// not cover every layer.
    pub fn measure(
        config: &Qwen4ExpConfig,
        decoder_layer_cache_dtypes: &[Qwen4ExpDecoderLayerCacheDtypes],
        full_attention_key_value_growth_tokens: usize,
    ) -> Result<Self, crate::DecoderCacheLayoutError> {
        let layout = qwen4_exp_decoder_cache_layout(
            config,
            full_attention_key_value_growth_tokens,
            decoder_layer_cache_dtypes,
        )?;
        let mut linear_attention_fixed_bytes = 0_u64;
        let mut full_attention_bytes_per_token = 0_u64;
        let mut index_key_cache_bytes_per_token = 0_u64;
        for layer_index in 0..layout.layer_count() {
            let layer = layout
                .layer(layer_index)
                .expect("layer index comes from the layout itself");
            match layer {
                crate::DecoderCacheLayerLayout::Composite { components } => {
                    for component in components {
                        match component {
                            crate::DecoderCacheLayerLayout::RecurrentTensor { tensor } => {
                                linear_attention_fixed_bytes +=
                                    tensor.fixed_payload_byte_count()? as u64;
                            }
                            crate::DecoderCacheLayerLayout::AppendOnlyAttention {
                                keys,
                                values,
                                ..
                            } => {
                                let per_token = (keys.sequence_payload_byte_count_per_token()?
                                    + values.sequence_payload_byte_count_per_token()?)
                                    as u64;
                                if keys.tensor_role_name() == QWEN4_EXP_INDEX_KEYS_TENSOR_ROLE {
                                    index_key_cache_bytes_per_token += per_token;
                                } else {
                                    full_attention_bytes_per_token += per_token;
                                }
                            }
                            _ => {
                                return Err(
                                    crate::DecoderCacheLayoutError::ModelConfigurationDimensionOutsideUsizeRange {
                                        dimension_name: "unexpected state component",
                                    },
                                );
                            }
                        }
                    }
                }
                _ => {
                    return Err(
                        crate::DecoderCacheLayoutError::ModelConfigurationDimensionOutsideUsizeRange {
                            dimension_name: "unexpected top-level layer layout",
                        },
                    );
                }
            }
        }
        let hyper_connections = config.hyper_connections.ok_or(
            crate::DecoderCacheLayoutError::ModelConfigurationDimensionOutsideUsizeRange {
                dimension_name: "hyper-connection geometry",
            },
        )?;
        let hyper_connection_stream_workspace_bytes =
            u64::from(
                config.decoder_layers * 2 * hyper_connections.stream_count * config.hidden_size,
            ) * activation_dtype_width(&config.activation_dtype);
        let full_attention_layers = config
            .layer_schedule
            .iter()
            .filter(|kind| **kind == Qwen4ExpLayerKind::FullAttention)
            .count() as u32;
        Ok(Self {
            linear_attention_fixed_bytes,
            full_attention_bytes_per_token,
            index_key_cache_bytes_per_token,
            hyper_connection_stream_workspace_bytes,
            full_attention_layers,
            linear_attention_layers: config.decoder_layers - full_attention_layers,
        })
    }

    /// Per-token bytes of every persisted stream combined.
    #[must_use]
    pub fn persisted_bytes_per_token(self) -> u64 {
        self.full_attention_bytes_per_token + self.index_key_cache_bytes_per_token
    }

    /// Total persisted bytes at a token count: fixed state plus growth.
    #[must_use]
    pub fn persisted_bytes_at(self, token_count: u64) -> u64 {
        self.linear_attention_fixed_bytes
            + self.persisted_bytes_per_token().saturating_mul(token_count)
    }
}

/// Bytes per element for the configured activation dtype.
fn activation_dtype_width(activation_dtype: &str) -> u64 {
    match activation_dtype {
        "float32" => 4,
        // Bfloat16 and float16 store two bytes per element.
        _ => 2,
    }
}
