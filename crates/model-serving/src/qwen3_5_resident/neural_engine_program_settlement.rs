//! Acts between requests on what a request learned about running programs.
//!
//! Programs are only released with no request active, because a chunk borrows them, so a
//! request only leaves this to do: it notes a failure or finishes a comparison, and the
//! worker settles it when it next goes idle.

use astronomical_ipc_protocol::NeuralEnginePrefillStatus;

use super::model::Qwen3_5ResidentModel;
use super::neural_engine_chunk_comparison::{ChunkComparisonVerdict, not_worthwhile_reason};
use super::neural_engine_mlp_prefill::unavailable;

const PROGRAM_FAILED_REASON: &str =
    "a Neural Engine program failed, so prefill stays on the GPU until the model next loads";

/// What the last request left to do, apart from the model so every case can be tested.
#[derive(Clone, Copy, Debug, PartialEq)]
enum ProgramSettlement {
    ReleaseAfterFailure,
    ReportVerdict(ChunkComparisonVerdict),
}

/// A failure beats a verdict, which a failed run could not be trusted to have reached, and
/// each verdict reaches the status and the record once.
fn program_settlement_of(
    has_failed: bool,
    has_reported_verdict: bool,
    verdict: Option<ChunkComparisonVerdict>,
) -> Option<ProgramSettlement> {
    if has_failed {
        return Some(ProgramSettlement::ReleaseAfterFailure);
    }
    if has_reported_verdict {
        return None;
    }
    verdict.map(ProgramSettlement::ReportVerdict)
}

/// `current_status` with the comparison's gain, or the refusal the verdict calls for.
fn status_after_verdict(
    verdict: ChunkComparisonVerdict,
    current_status: &NeuralEnginePrefillStatus,
) -> NeuralEnginePrefillStatus {
    match (verdict, current_status) {
        (ChunkComparisonVerdict::NotWorthwhile { gain }, _) => {
            unavailable(&not_worthwhile_reason(gain))
        }
        (
            ChunkComparisonVerdict::Worthwhile { .. },
            NeuralEnginePrefillStatus::Active {
                layer_program_count,
                neural_engine_tokens_per_chunk,
                chunk_token_count,
                measured_memory_bytes,
                ..
            },
        ) => NeuralEnginePrefillStatus::Active {
            layer_program_count: *layer_program_count,
            neural_engine_tokens_per_chunk: *neural_engine_tokens_per_chunk,
            chunk_token_count: *chunk_token_count,
            measured_memory_bytes: *measured_memory_bytes,
            measured_speed_percent: Some(verdict.speed_percent()),
        },
        (ChunkComparisonVerdict::Worthwhile { .. }, other_status) => other_status.clone(),
    }
}

/// Returns a status to report when anything changed. Failed or slower programs are
/// released, which gives their memory back within the driver's 10 to 20 seconds. Failed
/// gates go alone, because the MLP programs carry nearly all of the speedup.
pub(crate) fn settle_neural_engine_programs(
    model: &mut Qwen3_5ResidentModel,
    current_status: &NeuralEnginePrefillStatus,
) -> Option<NeuralEnginePrefillStatus> {
    if let Some(output_gate_engine) = model.base.output_gate_engine.as_ref()
        && output_gate_engine.has_failed()
    {
        tracing::warn!("releasing the Neural Engine output gate programs after a failure");
        model.base.output_gate_engine = None;
    }
    let prefill = model.neural_engine_mlp_prefill.as_ref()?;
    let settlement = program_settlement_of(
        prefill.has_failed.get(),
        prefill.has_reported_verdict.get(),
        prefill.chunk_comparison.borrow().verdict(),
    )?;
    let verdict = match settlement {
        ProgramSettlement::ReleaseAfterFailure => {
            release_programs(model);
            return Some(unavailable(PROGRAM_FAILED_REASON));
        }
        ProgramSettlement::ReportVerdict(verdict) => verdict,
    };
    prefill.has_reported_verdict.set(true);
    if let Some(speed_record) = prefill.speed_record.as_ref() {
        speed_record.record_gain(verdict.gain());
    }
    tracing::info!(
        gain = verdict.gain(),
        "Neural Engine programs compared with the GPU alone"
    );
    if matches!(verdict, ChunkComparisonVerdict::NotWorthwhile { .. }) {
        release_programs(model);
    }
    Some(status_after_verdict(verdict, current_status))
}

fn release_programs(model: &mut Qwen3_5ResidentModel) {
    model.neural_engine_mlp_prefill = None;
    model.base.output_gate_engine = None;
}

#[cfg(test)]
#[path = "neural_engine_program_settlement_tests.rs"]
mod tests;
