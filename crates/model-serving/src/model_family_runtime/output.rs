use crate::Qwen3_5RequestOutput;

/// Family-tagged request-local output state used by the generic worker.
pub enum ModelFamilyRequestOutput {
    Qwen3_5(Qwen3_5RequestOutput),
    /// The folded ternary family reuses the Qwen3.5 output state.
    PrismHadamardQwen35(Qwen3_5RequestOutput),
}
