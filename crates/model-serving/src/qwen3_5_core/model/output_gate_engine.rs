//! The seam that lets another engine compute a linear-attention layer's output gate.
//!
//! Only the gate's rows are offered because they never enter the recurrent state:
//! they multiply the recurrence's output afterwards, so rounding in them stays in
//! this token's output instead of compounding across the chunk through the state.
//! The core model does not know which engine this is, which keeps Core ML out of it.

use astronomical_runtime_integration::MlxRuntime;
use mlx_c_rust::MlxArray;

pub(crate) trait LinearAttentionOutputGateEngine: Send + std::fmt::Debug {
    /// Whether this chunk fits the engine's fixed-shape programs.
    fn handles(&self, token_count: i32) -> bool;

    /// Projects the gate while the GPU evaluates `gpu_work`, returning an evaluated
    /// `[1, tokens, channels]` array in `gpu_work`'s dtype. An error leaves the GPU
    /// to compute the gate instead.
    fn project_beside_gpu_work(
        &self,
        runtime: &MlxRuntime,
        hidden_states: &MlxArray,
        decoder_layer_index: usize,
        gpu_work: &[&MlxArray],
    ) -> Result<MlxArray, String>;

    /// Sends full chunks to the GPU alone while paused, so one loaded model can
    /// alternate between the arms of a measurement.
    fn set_paused(&self, is_paused: bool);
}
