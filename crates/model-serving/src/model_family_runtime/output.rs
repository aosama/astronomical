use crate::{DeepSeekV4UnavailableRequestOutput, LagunaRequestOutput, Qwen3_5RequestOutput};

/// Family-tagged request-local output state used by the generic worker.
pub enum ModelFamilyRequestOutput {
    Qwen3_5(Qwen3_5RequestOutput),
    Laguna(LagunaRequestOutput),
    DeepSeekV4(DeepSeekV4UnavailableRequestOutput),
}
