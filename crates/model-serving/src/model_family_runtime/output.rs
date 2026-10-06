use crate::{K2HorizonMoVARequestOutput, Qwen3_5RequestOutput};

/// Family-tagged request-local output state used by the generic worker.
pub enum ModelFamilyRequestOutput {
    Qwen3_5(Qwen3_5RequestOutput),
    K2HorizonMoVA(K2HorizonMoVARequestOutput),
}
