//! Prepares everything the opt-in Neural Engine prefill runs, behind its one setting,
//! beside a model that is already serving on the GPU.
//!
//! The MLP programs carry nearly all of the speedup and decide the status a user
//! sees. The output gate programs add a few percent on top (1.24× to 1.28× on a 9B
//! model, issue #161), so a failure there is logged and leaves the gates on the
//! GPU instead of turning the MLP programs off.

use std::path::{Path, PathBuf};
use std::time::Duration;

use astronomical_ipc_protocol::{NeuralEnginePrefillStatus, WorkerChunkingConfiguration};
use astronomical_neural_engine::core_ml::CoreMlProgram;

use super::model::Qwen3_5ResidentModel;
use super::neural_engine_mlp_prefill::{
    MlpProgramPlan, plan_neural_engine_mlp_programs, unavailable,
};
use super::neural_engine_output_gate_prefill::{
    OutputGateProgramPlan, plan_neural_engine_output_gate_programs,
};
use super::neural_engine_program_cache::CoreMlProgramBuilder;
use super::neural_engine_program_preparation::{
    NeuralEngineProgramFamily, ProgramIdentity, ProgramPackageWriter, ProgramPreparation,
};

/// How often the blocking path looks for finished programs. Short beside a build,
/// which takes from a tenth of a second to several seconds.
const BLOCKING_PREPARATION_POLL_INTERVAL: Duration = Duration::from_millis(20);

/// Reads compile input from the model on the MLX owner thread for the plans in hand.
struct ModelProgramPackages<'preparation> {
    model: &'preparation Qwen3_5ResidentModel,
    mlp_plan: &'preparation MlpProgramPlan,
    output_gate_plan: Option<&'preparation OutputGateProgramPlan>,
}

impl ProgramPackageWriter for ModelProgramPackages<'_> {
    fn write_package(
        &self,
        identity: ProgramIdentity,
        package_directory: &Path,
    ) -> Result<PathBuf, &'static str> {
        let decoder_layer_index = identity.decoder_layer_index;
        match (identity.family, self.output_gate_plan) {
            (NeuralEngineProgramFamily::Mlp, _) => {
                self.mlp_plan
                    .write_package(self.model, decoder_layer_index, package_directory)
            }
            (NeuralEngineProgramFamily::OutputGate, Some(output_gate_plan)) => {
                output_gate_plan.write_package(self.model, decoder_layer_index, package_directory)
            }
            (NeuralEngineProgramFamily::OutputGate, None) => {
                Err("no output gate programs were planned")
            }
        }
    }
}

/// Programs on their way to a loaded model. Dropping this abandons them and waits
/// for the build thread, so nothing outlives the model they were meant for.
pub(crate) struct NeuralEnginePrefillPreparation {
    mlp_plan: MlpProgramPlan,
    output_gate_plan: Option<OutputGateProgramPlan>,
    program_preparation: ProgramPreparation<CoreMlProgram>,
    last_reported_completed_program_count: Option<usize>,
}

/// Decides whether the programs may run and starts building them. The status is the
/// one to report now; a preparation comes back only while it says `Preparing`.
pub(crate) fn begin_neural_engine_prefill_preparation(
    model: &Qwen3_5ResidentModel,
    chunking: &WorkerChunkingConfiguration,
    maximum_context_tokens: u32,
    program_cache_directory: Option<&Path>,
) -> (
    NeuralEnginePrefillStatus,
    Option<NeuralEnginePrefillPreparation>,
) {
    let mlp_plan = match plan_neural_engine_mlp_programs(
        model,
        chunking,
        maximum_context_tokens,
        program_cache_directory,
    ) {
        Ok(mlp_plan) => mlp_plan,
        Err(settled_status) => return (settled_status, None),
    };
    let output_gate_plan =
        plan_output_gates_or_stay_on_the_gpu(model, chunking, program_cache_directory);
    let mut planned_programs = mlp_plan.planned_programs();
    if let Some(output_gate_plan) = &output_gate_plan {
        planned_programs.extend(output_gate_plan.planned_programs());
    }
    let Some(program_preparation) =
        ProgramPreparation::start(planned_programs, CoreMlProgramBuilder)
    else {
        return (
            unavailable("the Neural Engine program thread could not start"),
            None,
        );
    };
    let mut preparation = NeuralEnginePrefillPreparation {
        mlp_plan,
        output_gate_plan,
        program_preparation,
        last_reported_completed_program_count: None,
    };
    let opening_status = preparation.preparing_status();
    preparation.last_reported_completed_program_count = Some(0);
    (opening_status, Some(preparation))
}

fn plan_output_gates_or_stay_on_the_gpu(
    model: &Qwen3_5ResidentModel,
    chunking: &WorkerChunkingConfiguration,
    program_cache_directory: Option<&Path>,
) -> Option<OutputGateProgramPlan> {
    match plan_neural_engine_output_gate_programs(model, chunking, program_cache_directory) {
        Ok(output_gate_plan) => Some(output_gate_plan),
        Err(reason) => {
            warn_output_gates_stay_on_the_gpu(reason);
            None
        }
    }
}

fn warn_output_gates_stay_on_the_gpu(reason: &str) {
    tracing::warn!(
        reason,
        "Neural Engine output gate programs are unavailable; the gates stay on the GPU"
    );
}

/// What one step of a preparation produced.
pub(crate) enum NeuralEnginePreparationStep {
    /// Nothing a reader has not already seen.
    Unchanged,
    Progressed(NeuralEnginePrefillStatus),
    /// Every program has an answer, so [`NeuralEnginePrefillPreparation::finish`] applies.
    ReadyToFinish,
}

impl NeuralEnginePrefillPreparation {
    /// Moves the programs along. Called on the MLX owner thread between requests,
    /// so a chunk already running never sees a program appear part way through.
    pub(crate) fn advance(&mut self, model: &Qwen3_5ResidentModel) -> NeuralEnginePreparationStep {
        let package_writer = ModelProgramPackages {
            model,
            mlp_plan: &self.mlp_plan,
            output_gate_plan: self.output_gate_plan.as_ref(),
        };
        self.program_preparation.advance(&package_writer);
        self.release_weight_copies(model);
        if let Some(reason) = self
            .program_preparation
            .first_failure_of(NeuralEngineProgramFamily::OutputGate)
        {
            self.program_preparation
                .fail_waiting_programs_of(NeuralEngineProgramFamily::OutputGate, reason);
        }
        let has_failed_mlp_program = self
            .program_preparation
            .first_failure_of(NeuralEngineProgramFamily::Mlp)
            .is_some();
        if has_failed_mlp_program || self.program_preparation.is_finished() {
            return NeuralEnginePreparationStep::ReadyToFinish;
        }
        let completed_program_count = self.program_preparation.completed_program_count();
        if self.last_reported_completed_program_count == Some(completed_program_count) {
            return NeuralEnginePreparationStep::Unchanged;
        }
        self.last_reported_completed_program_count = Some(completed_program_count);
        let preparing_status = self.preparing_status();
        log_preparation_progress(&preparing_status);
        NeuralEnginePreparationStep::Progressed(preparing_status)
    }

    /// Reading a layer leaves its fp16 copy in MLX's pool. Released at once, because
    /// a request can arrive before the next step and should not compete with it. A
    /// failure costs that request some memory and nothing else, so it stays a warning.
    fn release_weight_copies(&mut self, model: &Qwen3_5ResidentModel) {
        if self.program_preparation.take_has_written_package()
            && let Err(release_error) = model
                .runtime()
                .synchronize_gpu_stream_and_clear_allocator_cache()
        {
            tracing::warn!(error = %release_error, "could not release Neural Engine weight copies");
        }
    }

    /// Engages the programs, or says why the model stays on the GPU.
    pub(crate) fn finish(self, model: &mut Qwen3_5ResidentModel) -> NeuralEnginePrefillStatus {
        let Self {
            mlp_plan,
            output_gate_plan,
            mut program_preparation,
            ..
        } = self;
        if let Some(reason) = program_preparation.first_failure_of(NeuralEngineProgramFamily::Mlp) {
            return unavailable(reason);
        }
        if let Some(refused_status) = mlp_plan.refusal_at_current_ceiling(model) {
            return refused_status;
        }
        let mut mlp_programs = Vec::new();
        for (_decoder_layer_index, program) in
            program_preparation.take_built_programs_of(NeuralEngineProgramFamily::Mlp)
        {
            mlp_programs.push(program);
        }
        let status = mlp_plan.attach(model, mlp_programs);
        attach_output_gates(model, output_gate_plan, &mut program_preparation);
        status
    }

    fn preparing_status(&self) -> NeuralEnginePrefillStatus {
        let completed_program_count = self.program_preparation.completed_program_count();
        let total_program_count = self.program_preparation.total_program_count();
        let elapsed_ms =
            u64::try_from(self.program_preparation.elapsed().as_millis()).unwrap_or(u64::MAX);
        NeuralEnginePrefillStatus::Preparing {
            completed_program_count: u32::try_from(completed_program_count).unwrap_or(u32::MAX),
            total_program_count: u32::try_from(total_program_count).unwrap_or(u32::MAX),
            elapsed_ms,
            estimated_remaining_ms: estimated_remaining_ms(
                elapsed_ms,
                completed_program_count as u64,
                total_program_count as u64,
            ),
        }
    }
}

fn attach_output_gates(
    model: &mut Qwen3_5ResidentModel,
    output_gate_plan: Option<OutputGateProgramPlan>,
    program_preparation: &mut ProgramPreparation<CoreMlProgram>,
) {
    if let Some(reason) =
        program_preparation.first_failure_of(NeuralEngineProgramFamily::OutputGate)
    {
        warn_output_gates_stay_on_the_gpu(reason);
        return;
    }
    let Some(output_gate_plan) = output_gate_plan else {
        return;
    };
    let output_gate_program_count = output_gate_plan.attach(
        model,
        program_preparation.take_built_programs_of(NeuralEngineProgramFamily::OutputGate),
    );
    tracing::info!(
        output_gate_program_count,
        "Neural Engine output gate programs loaded"
    );
}

/// Serving reports this through the status too, but a measurement that blocks on the
/// programs has only the log to show that a long wait is moving.
fn log_preparation_progress(preparing_status: &NeuralEnginePrefillStatus) {
    if let NeuralEnginePrefillStatus::Preparing {
        completed_program_count,
        total_program_count,
        elapsed_ms,
        estimated_remaining_ms,
    } = preparing_status
    {
        tracing::info!(
            completed_program_count,
            total_program_count,
            percent_complete = u64::from(*completed_program_count) * 100
                / u64::from((*total_program_count).max(1)),
            elapsed_ms,
            eta_ms = estimated_remaining_ms,
            "Neural Engine programs preparing"
        );
    }
}

/// Assumes the remaining programs cost what the finished ones did. Reused programs
/// take milliseconds and compiled ones seconds, so a mixed set can run slower than
/// this says.
fn estimated_remaining_ms(
    elapsed_ms: u64,
    completed_program_count: u64,
    total_program_count: u64,
) -> Option<u64> {
    if completed_program_count == 0 {
        return None;
    }
    let remaining_program_count = total_program_count.saturating_sub(completed_program_count);
    Some(elapsed_ms.saturating_mul(remaining_program_count) / completed_program_count)
}

/// Builds the programs and waits for them. For measurements that need a known
/// starting point; serving uses the preparation so a user never waits on this.
pub(crate) fn prepare_neural_engine_prefill(
    model: &mut Qwen3_5ResidentModel,
    chunking: &WorkerChunkingConfiguration,
    maximum_context_tokens: u32,
    program_cache_directory: Option<&Path>,
) -> NeuralEnginePrefillStatus {
    let (settled_status, preparation) = begin_neural_engine_prefill_preparation(
        model,
        chunking,
        maximum_context_tokens,
        program_cache_directory,
    );
    let Some(mut preparation) = preparation else {
        return settled_status;
    };
    loop {
        if matches!(
            preparation.advance(model),
            NeuralEnginePreparationStep::ReadyToFinish
        ) {
            return preparation.finish(model);
        }
        std::thread::sleep(BLOCKING_PREPARATION_POLL_INTERVAL);
    }
}

#[cfg(test)]
#[path = "neural_engine_prefill_loading_tests.rs"]
mod tests;
