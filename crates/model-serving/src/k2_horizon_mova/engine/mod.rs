//! K2 Horizon MoVA inference engine.

mod attribution;
mod execution;
mod memory;
mod prefill;
mod prompt_cache;

pub use execution::K2HorizonMoVAInferenceExecution;
pub(super) use execution::K2HorizonMoVAPendingStartup;

use crate::MlxInferenceEngine;

pub type K2HorizonMoVAEngine = MlxInferenceEngine<K2HorizonMoVAInferenceExecution>;
