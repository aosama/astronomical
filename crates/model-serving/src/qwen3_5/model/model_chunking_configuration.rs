use super::Qwen3_5ExecutionError;

/// Validated model-local work boundaries supplied by the standard user configuration.
///
/// This is the native-shape representation of the worker's broader chunking
/// contract. Construction performs the signed and platform-sized conversions
/// once so hot forward paths use values that cannot fail conversion mid-request.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Qwen3_5ModelChunkingConfiguration {
    /// Append-only attention capacity growth passed to MLX tensor shapes.
    pub(crate) full_attention_key_value_growth_tokens: i32,
    /// Resident multi-token prefill command-buffer interval. Zero is one lazy tape.
    pub(crate) prefill_graph_submission_layer_interval: u32,
    /// SSD-paged multi-token prefill command-buffer interval. Zero is one lazy tape.
    pub(crate) experimental_ssd_paging_prefill_graph_submission_layer_interval: u32,
    /// Experimental one-token layer interval used only during solid-state-drive paging.
    pub(crate) experimental_ssd_paging_generation_graph_submission_layer_interval: u32,
    /// Maximum prompt rows supplied to one drafter forward.
    pub(crate) speculative_prefill_draft_forward_tokens: usize,
}

impl Qwen3_5ModelChunkingConfiguration {
    pub fn new(
        full_attention_key_value_growth_tokens: u32,
        prefill_graph_submission_layer_interval: u32,
        experimental_ssd_paging_prefill_graph_submission_layer_interval: u32,
        experimental_ssd_paging_generation_graph_submission_layer_interval: u32,
        speculative_prefill_draft_forward_tokens: u32,
    ) -> Result<Self, Qwen3_5ExecutionError> {
        if speculative_prefill_draft_forward_tokens == 0 {
            return Err(Qwen3_5ExecutionError::InvalidInput {
                description: "speculative-prefill draft forward tokens must be positive",
            });
        }
        if full_attention_key_value_growth_tokens == 0 {
            return Err(Qwen3_5ExecutionError::InvalidInput {
                description: "full-attention key/value growth tokens must be positive",
            });
        }
        Ok(Self {
            full_attention_key_value_growth_tokens: i32::try_from(
                full_attention_key_value_growth_tokens,
            )
            .map_err(|_| Qwen3_5ExecutionError::InvalidInput {
                description: "full-attention key/value growth tokens exceed the Int32 range",
            })?,
            prefill_graph_submission_layer_interval,
            experimental_ssd_paging_prefill_graph_submission_layer_interval,
            experimental_ssd_paging_generation_graph_submission_layer_interval,
            speculative_prefill_draft_forward_tokens: usize::try_from(
                speculative_prefill_draft_forward_tokens,
            )
            .map_err(|_| Qwen3_5ExecutionError::InvalidInput {
                description: "speculative-prefill draft forward tokens exceed the usize range",
            })?,
        })
    }
}
