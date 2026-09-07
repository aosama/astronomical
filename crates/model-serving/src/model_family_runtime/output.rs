use crate::{
    DeepSeekV4UnavailableRequestOutput, K2HorizonMoVARequestOutput, LagunaRequestOutput,
    Qwen3_5RequestOutput,
};

/// Family-tagged request-local output state used by the generic worker.
pub enum ModelFamilyRequestOutput {
    Qwen3_5(Qwen3_5RequestOutput),
    Laguna(LagunaRequestOutput),
    K2HorizonMoVA(K2HorizonMoVARequestOutput),
    DeepSeekV4(DeepSeekV4UnavailableRequestOutput),
}
