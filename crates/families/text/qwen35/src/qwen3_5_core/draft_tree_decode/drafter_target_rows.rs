//! The shared model as a DFlash drafter's target: the drafter borrows its
//! embedding rows and output head, so it has no vocabulary of its own and
//! its logits are the target's. Both engines' models are this base.

use mlx_bridge::{MlxArray, MlxDtype};
use runtime_integration::{MlxRuntime, MlxRuntimeError};

use crate::qwen3_5_core::model::Qwen3_5ModelBase;
use engine_core::speculative_decoding::dflash_drafter::DFlashDraftTarget;

impl DFlashDraftTarget for Qwen3_5ModelBase {
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

fn runtime_error(execution_error: impl std::fmt::Display) -> MlxRuntimeError {
    MlxRuntimeError::RuntimeOperation {
        operation: "lend the target's rows to the DFlash drafter",
        description: execution_error.to_string(),
    }
}
