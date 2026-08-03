use super::Qwen3_5MoEConfig;

const BFLOAT16_ELEMENT_SIZE_BYTES: usize = 2;
const FULL_ATTENTION_KEY_VALUE_STATE_TENSOR_COUNT: usize = 2;

impl Qwen3_5MoEConfig {
    /// Reserves context-growing full-attention key/value state for each context token.
    #[must_use]
    pub fn context_memory_reservation_bytes(&self, context_token_count: usize) -> Option<usize> {
        let full_attention_layer_count = self.full_attention_decoder_layer_indexes().len();
        context_token_count
            .checked_mul(full_attention_layer_count)?
            .checked_mul(FULL_ATTENTION_KEY_VALUE_STATE_TENSOR_COUNT)?
            .checked_mul(self.key_value_head_count() as usize)?
            .checked_mul(self.head_dimension() as usize)?
            .checked_mul(BFLOAT16_ELEMENT_SIZE_BYTES)
    }

    #[cfg(feature = "direct-mlx")]
    /// Returns one full-attention layer's exact key/value state bytes per token.
    #[must_use]
    pub fn full_attention_key_value_state_bytes_per_layer_token(&self) -> Option<usize> {
        FULL_ATTENTION_KEY_VALUE_STATE_TENSOR_COUNT
            .checked_mul(self.key_value_head_count() as usize)?
            .checked_mul(self.head_dimension() as usize)?
            .checked_mul(BFLOAT16_ELEMENT_SIZE_BYTES)
    }
}
