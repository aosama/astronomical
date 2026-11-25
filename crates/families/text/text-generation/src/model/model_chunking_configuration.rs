use crate::execution_error::TextExecutionError;

/// Validated model-local work boundaries supplied by the standard user configuration.
///
/// This is the native-shape representation of the worker's broader chunking
/// contract. Construction performs the signed and platform-sized conversions
/// once so hot forward paths use values that cannot fail conversion mid-request.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TextModelChunkingConfiguration {
    /// Append-only attention capacity growth passed to MLX tensor shapes.
    pub full_attention_key_value_growth_tokens: i32,
    /// Resident multi-token prefill command-buffer interval. Zero is one lazy tape.
    pub prefill_graph_submission_layer_interval: u32,
    /// Experimental one-token layer interval used only during solid-state-drive paging.
    pub experimental_ssd_paging_generation_graph_submission_layer_interval: u32,
}

impl TextModelChunkingConfiguration {
    pub fn new(
        full_attention_key_value_growth_tokens: u32,
        prefill_graph_submission_layer_interval: u32,
        experimental_ssd_paging_generation_graph_submission_layer_interval: u32,
    ) -> Result<Self, TextExecutionError> {
        if full_attention_key_value_growth_tokens == 0 {
            return Err(TextExecutionError::InvalidInput {
                description: "full-attention key/value growth tokens must be positive",
            });
        }
        Ok(Self {
            full_attention_key_value_growth_tokens: i32::try_from(
                full_attention_key_value_growth_tokens,
            )
            .map_err(|_| TextExecutionError::InvalidInput {
                description: "full-attention key/value growth tokens exceed the Int32 range",
            })?,
            prefill_graph_submission_layer_interval,
            experimental_ssd_paging_generation_graph_submission_layer_interval,
        })
    }
}
