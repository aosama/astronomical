//! Bound weight tensors for one DFlash draft model.
//!
//! The struct owns both the materialized arrays and the safetensors reader
//! they were loaded through: MLX arrays stay lazy after loading and may read
//! through the reader vtable at first evaluation, so the reader must outlive
//! every array it produced. All sixty-nine shipped tensors are native
//! BF16 (bfloat16); the embedding table and language-model head are not
//! shipped at all — the draft borrows both from its target model, which is
//! what makes the token-id space match by construction.

use mlx_c_rust::MlxArray;

use super::draft_configuration::DFlashDraftConfiguration;

/// Learned weights for one draft layer, in checkpoint tensor layout.
#[derive(Debug)]
pub struct DFlashDraftLayerWeights {
    pub query_projection: MlxArray,
    pub key_projection: MlxArray,
    pub value_projection: MlxArray,
    pub output_projection: MlxArray,
    pub query_head_normalization: MlxArray,
    pub key_head_normalization: MlxArray,
    pub gate_projection: MlxArray,
    pub up_projection: MlxArray,
    pub down_projection: MlxArray,
    pub input_normalization: MlxArray,
    pub post_attention_normalization: MlxArray,
}

/// Learned weights for the whole draft trunk plus the target-conditioning
/// projection. Shapes are validated against the configuration at bind time.
#[derive(Debug)]
pub struct DFlashDraftWeights {
    pub layers: Vec<DFlashDraftLayerWeights>,
    /// Fuses the concatenated target hidden-state taps into draft width:
    /// `[draft hidden size, tap count × target hidden size]`.
    pub target_conditioning_projection: MlxArray,
    pub target_conditioning_normalization: MlxArray,
    pub final_normalization: MlxArray,
    /// Exact safetensors payload bytes; the memory-governance budget charges
    /// the draft through this number.
    pub weights_payload_bytes: u64,
}

impl DFlashDraftWeights {
    /// Total logical bytes across every bound tensor, for admission budgets.
    #[must_use]
    pub fn bound_tensor_bytes(&self) -> u64 {
        let mut bound_tensor_bytes = 0_u64;
        for layer_weights in &self.layers {
            for tensor in [
                &layer_weights.query_projection,
                &layer_weights.key_projection,
                &layer_weights.value_projection,
                &layer_weights.output_projection,
                &layer_weights.query_head_normalization,
                &layer_weights.key_head_normalization,
                &layer_weights.gate_projection,
                &layer_weights.up_projection,
                &layer_weights.down_projection,
                &layer_weights.input_normalization,
                &layer_weights.post_attention_normalization,
            ] {
                bound_tensor_bytes += tensor.byte_count() as u64;
            }
        }
        bound_tensor_bytes
            + self.target_conditioning_projection.byte_count() as u64
            + self.target_conditioning_normalization.byte_count() as u64
            + self.final_normalization.byte_count() as u64
    }

    /// Expected weight matrix shape helper shared with the loader's checks.
    #[must_use]
    pub fn expected_projection_shape(configuration: &DFlashDraftConfiguration) -> (i32, i32, i32) {
        (
            configuration.attention_head_count() * configuration.head_dimension(),
            configuration.key_value_head_count() * configuration.head_dimension(),
            configuration.intermediate_size(),
        )
    }
}
