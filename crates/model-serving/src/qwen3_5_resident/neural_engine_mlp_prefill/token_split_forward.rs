//! One dense MLP layer split by token between the Neural Engine and the GPU.

use astronomical_neural_engine::mlx_hand_off::{
    MlxHandOffError, OutputHandOff, SplitAxis, channel_major_fp16, merge_surface_with_gpu_slice,
    write_prepared_input,
};
use mlx_c_rust::MlxArray;
use thiserror::Error;

use super::{NEURAL_ENGINE_FINISH_TIMEOUT, NeuralEngineMlpPrefill};
use crate::qwen3_5_core::dense::mlp::Qwen3_5DenseMlpWeights;
use crate::qwen3_5_core::model_math::error::Qwen3_5ExecutionError;
use crate::qwen3_5_resident::model::Qwen3_5ResidentModel;

#[derive(Debug, Error)]
enum TokenSplitForwardError {
    #[error(transparent)]
    HandOff(#[from] MlxHandOffError),
    #[error(transparent)]
    Gpu(#[from] Qwen3_5ExecutionError),
    #[error(transparent)]
    Mlx(#[from] mlx_c_rust::MlxCError),
    #[error(transparent)]
    Program(#[from] astronomical_neural_engine::core_ml::CoreMlProgramError),
    #[error("no Neural Engine program exists for decoder layer {layer_index}")]
    MissingLayerProgram { layer_index: usize },
}

impl TokenSplitForwardError {
    const fn is_about_the_programs(&self) -> bool {
        match self {
            Self::HandOff(hand_off_error) => !hand_off_error.is_from_mlx(),
            Self::Program(_) | Self::MissingLayerProgram { .. } => true,
            Self::Gpu(_) | Self::Mlx(_) => false,
        }
    }
}

impl Qwen3_5ResidentModel {
    /// The dense MLP, on both engines when this chunk fits the programs.
    ///
    /// A runtime failure falls back to the GPU for this layer and every later
    /// chunk, so a misbehaving Neural Engine costs speed, never the request.
    pub(crate) fn forward_resident_dense_mlp(
        &self,
        normalized_attention: &MlxArray,
        dense_mlp_weights: &Qwen3_5DenseMlpWeights,
        token_count: i32,
        layer_index: usize,
    ) -> Result<MlxArray, Qwen3_5ExecutionError> {
        if let Some(prefill) = self.neural_engine_mlp_prefill.as_ref()
            && prefill.handles(token_count)
        {
            match self.forward_dense_mlp_split_by_token(
                normalized_attention,
                dense_mlp_weights,
                layer_index,
                prefill,
            ) {
                Ok(merged) => return Ok(merged),
                // An MLX error says nothing about the programs, and is left to surface
                // from the GPU path below, where memory failures are handled.
                Err(token_split_error) if token_split_error.is_about_the_programs() => {
                    tracing::warn!(
                        layer_index,
                        error = %token_split_error,
                        "Neural Engine MLP prefill failed; remaining prefill stays on the GPU"
                    );
                    prefill.has_failed.set(true);
                }
                Err(token_split_error) => {
                    tracing::warn!(
                        layer_index,
                        error = %token_split_error,
                        "this layer stays on the GPU after an error outside the Neural Engine"
                    );
                }
            }
        }
        self.base
            .forward_qwen3_5_dense_mlp(normalized_attention, dense_mlp_weights)
    }

    fn forward_dense_mlp_split_by_token(
        &self,
        normalized_attention: &MlxArray,
        dense_mlp_weights: &Qwen3_5DenseMlpWeights,
        layer_index: usize,
        prefill: &NeuralEngineMlpPrefill,
    ) -> Result<MlxArray, TokenSplitForwardError> {
        let shape = normalized_attention.shape();
        let (token_count, channel_count) = (shape[1], shape[2]);
        let neural_engine_tokens = prefill.neural_engine_token_count as i32;
        let neural_engine_input = self.runtime.slice(
            normalized_attention,
            &[0, 0, 0],
            &[1, neural_engine_tokens, channel_count],
            &[1, 1, 1],
        )?;
        let prepared = channel_major_fp16(&self.runtime, &neural_engine_input)?;
        self.runtime.evaluate_arrays(&[&prepared])?;
        let gpu_input = self.runtime.slice(
            normalized_attention,
            &[0, neural_engine_tokens, 0],
            &[1, token_count, channel_count],
            &[1, 1, 1],
        )?;
        let gpu_slice = self
            .base
            .forward_qwen3_5_dense_mlp(&gpu_input, dense_mlp_weights)?;
        // The GPU starts first so the host copy below does not leave it idle.
        self.runtime.async_eval_arrays(&[&gpu_slice])?;
        let mut layer_programs = prefill.layer_programs.borrow_mut();
        let program = layer_programs
            .get_mut(layer_index)
            .ok_or(TokenSplitForwardError::MissingLayerProgram { layer_index })?;
        write_prepared_input(&prepared, program)?;
        let pending = program.begin_prediction()?;
        self.runtime.evaluate_arrays(&[&gpu_slice])?;
        pending.finish(NEURAL_ENGINE_FINISH_TIMEOUT)?;
        Ok(merge_surface_with_gpu_slice(
            &self.runtime,
            program,
            &gpu_slice,
            SplitAxis::Tokens,
            OutputHandOff::ZeroCopySurface,
        )?)
    }
}

#[cfg(test)]
#[path = "token_split_forward_tests.rs"]
mod tests;
