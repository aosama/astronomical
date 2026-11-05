//! Loads everything the opt-in Neural Engine prefill runs, behind its one setting.
//!
//! The MLP programs carry nearly all of the speedup and decide the status a user
//! sees. The output gate programs add a few percent on top (1.24× to 1.28× on a 9B
//! model, issue #161), so a failure there is logged and leaves the gates on the
//! GPU instead of turning the MLP programs off.

use std::path::Path;

use astronomical_ipc_protocol::{NeuralEnginePrefillStatus, WorkerChunkingConfiguration};

use super::model::Qwen3_5ResidentModel;
use super::neural_engine_mlp_prefill::prepare_neural_engine_mlp_prefill;
use super::neural_engine_output_gate_prefill::prepare_neural_engine_output_gate_prefill;

pub(crate) fn prepare_neural_engine_prefill(
    model: &mut Qwen3_5ResidentModel,
    chunking: &WorkerChunkingConfiguration,
    maximum_context_tokens: u32,
    program_cache_directory: Option<&Path>,
) -> NeuralEnginePrefillStatus {
    let status = prepare_neural_engine_mlp_prefill(
        model,
        chunking,
        maximum_context_tokens,
        program_cache_directory,
    );
    if !matches!(status, NeuralEnginePrefillStatus::Active { .. }) {
        return status;
    }
    match prepare_neural_engine_output_gate_prefill(model, chunking, program_cache_directory) {
        Ok(output_gate_program_count) => tracing::info!(
            output_gate_program_count,
            "Neural Engine output gate programs loaded"
        ),
        Err(reason) => tracing::warn!(
            reason,
            "Neural Engine output gate programs are unavailable; the gates stay on the GPU"
        ),
    }
    status
}
