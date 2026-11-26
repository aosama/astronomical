//! What the load-time MLX working-buffer probes share: the shape they run at
//! and the layers they run.
//!
//! A dense model and an expert model measure different things
//! (`dense_layer_mlx_working_buffer_probe`, `expert_layer_mlx_working_buffer_probe`),
//! but both run one layer of each attention kind over a chunk of the size
//! the engine will serve.

use super::Qwen3_5Model;

/// The forward shape the engine will serve, which a probe reproduces.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LayerProbeShape {
    /// The largest prefill chunk the engine's chunk sizer will submit.
    pub prefill_chunk_token_count: usize,
    /// The attention slab granularity requests allocate state with.
    pub full_attention_kv_state_growth_tokens: i32,
}

impl Qwen3_5Model {
    /// The first linear-attention layer and the first full-attention layer.
    pub(super) fn probe_layer_indexes(&self) -> Vec<usize> {
        let layer_count = self.config.layer_count() as usize;
        let mut probe_layer_indexes = Vec::with_capacity(2);
        for wants_full_attention in [false, true] {
            for layer_index in 0..layer_count {
                if self.config.decoder_layer_is_full_attention(layer_index) == wants_full_attention
                {
                    probe_layer_indexes.push(layer_index);
                    break;
                }
            }
        }
        probe_layer_indexes
    }
}
