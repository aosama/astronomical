//! The resident Qwen3.5 model as a DFlash2 drafter's target: the drafter
//! borrows its embedding rows and output head, so it has no vocabulary of its
//! own and its logits are the target's.

use astronomical_runtime_integration::{MlxRuntime, MlxRuntimeError};
use mlx_bridge::{MlxArray, MlxDtype};

use crate::InferenceEngineError;
use crate::gpu_token_sampling::sampling_probability_rows;
use crate::speculative_decoding::dflash2_drafter::DFlash2DraftTarget;

use super::model::Qwen3_5ResidentModel;

impl DFlash2DraftTarget for Qwen3_5ResidentModel {
    fn embedded_rows(
        &self,
        _runtime: &MlxRuntime,
        token_ids: &MlxArray,
    ) -> Result<MlxArray, MlxRuntimeError> {
        let signed_token_ids = self.runtime.astype(token_ids, MlxDtype::Int32)?;
        self.embedding_lookup(&signed_token_ids)
            .map_err(runtime_error)
    }

    fn head_logits(
        &self,
        _runtime: &MlxRuntime,
        hidden_rows: &MlxArray,
    ) -> Result<MlxArray, MlxRuntimeError> {
        self.quantized_linear(hidden_rows, &self.weights.language_model_head_weights)
            .map_err(runtime_error)
    }
}

impl Qwen3_5ResidentModel {
    /// The probabilities this engine's sampler would draw each row of
    /// `logits` [1, rows, vocabulary] from, for speculative verification.
    pub fn sampling_probability_rows(
        &self,
        logits: &MlxArray,
        temperature_thousandths: u16,
        top_p_thousandths: u16,
        top_k: u16,
    ) -> Result<MlxArray, InferenceEngineError> {
        sampling_probability_rows(
            &self.runtime,
            logits,
            temperature_thousandths,
            top_p_thousandths,
            top_k,
        )
    }
}

fn runtime_error(execution_error: impl std::fmt::Display) -> MlxRuntimeError {
    MlxRuntimeError::RuntimeOperation {
        operation: "lend the resident target's rows to the DFlash2 drafter",
        description: execution_error.to_string(),
    }
}
