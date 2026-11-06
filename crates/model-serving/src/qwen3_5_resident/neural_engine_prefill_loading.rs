//! Prepares everything the opt-in Neural Engine prefill runs, behind its one setting,
//! beside a model that is already serving on the GPU.
//!
//! The MLP programs carry nearly all of the speedup and decide the status a user
//! sees. The output gate programs add a few percent on top (1.24× to 1.28× on a 9B
//! model, issue #161), so a failure there is logged and leaves the gates on the
//! GPU instead of turning the MLP programs off.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use astronomical_ipc_protocol::{NeuralEnginePrefillStatus, WorkerChunkingConfiguration};
use astronomical_neural_engine::core_ml::CoreMlProgram;

use super::model::Qwen3_5ResidentModel;
use super::neural_engine_idle_watch::IdleWatch;
use super::neural_engine_mlp_prefill::{
    MlpProgramPlan, plan_neural_engine_mlp_programs, unavailable,
};
use super::neural_engine_output_gate_prefill::{
    OutputGateProgramPlan, plan_neural_engine_output_gate_programs,
};
use super::neural_engine_program_cache::CoreMlProgramBuilder;
use super::neural_engine_program_measurement::{
    FinishedMeasurement, MeasurementAction, MeasurementOutcome, ProgramMeasurement,
    ProgramsToMeasure, SystemWiredMemory, action_after_measurement,
};
use super::neural_engine_program_preparation::{
    NeuralEngineProgramFamily, ProgramIdentity, ProgramPackageWriter, ProgramPreparation,
};

/// How often the blocking path looks for finished programs. Short beside a build,
/// which takes from a tenth of a second to several seconds.
const BLOCKING_PREPARATION_POLL_INTERVAL: Duration = Duration::from_millis(20);
/// What the status adds for the measurement itself after the idle wait.
const TYPICAL_MEASURING_DURATION: Duration = Duration::from_secs(3);
const WAITING_REPORT_INTERVAL: Duration = Duration::from_secs(5);
/// Each try takes up to 10 seconds of the Neural Engine, and a Mac this busy will not
/// have settled a few seconds later.
const MOST_UNSETTLED_MEASUREMENTS: u32 = 3;

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

/// Where a preparation has got to, once its programs are built.
enum ProgramMemoryPhase {
    Building,
    WaitingForIdle(ProgramsToMeasure<CoreMlProgram>),
    Measuring(ProgramMeasurement<CoreMlProgram>),
    Known {
        programs: ProgramsToMeasure<CoreMlProgram>,
        wired_bytes: u64,
    },
    Finished(NeuralEnginePrefillStatus),
    Failed(&'static str),
}

/// Programs on their way to a loaded model. Dropping this abandons them and waits
/// for the build thread, so nothing outlives the model they were meant for.
pub(crate) struct NeuralEnginePrefillPreparation {
    mlp_plan: MlpProgramPlan,
    output_gate_plan: Option<OutputGateProgramPlan>,
    program_preparation: ProgramPreparation<CoreMlProgram>,
    last_reported_completed_program_count: Option<usize>,
    built_program_count: usize,
    idle_watch: IdleWatch,
    phase: ProgramMemoryPhase,
    last_waiting_report_at: Instant,
    unsettled_measurement_count: u32,
}

/// Decides whether the programs may run and starts building them. The status is the
/// one to report now; a preparation comes back only while it says `Preparing`.
pub(crate) fn begin_neural_engine_prefill_preparation(
    model: &Qwen3_5ResidentModel,
    chunking: &WorkerChunkingConfiguration,
    program_cache_directory: Option<&Path>,
    neural_engine_token_count_override: Option<u32>,
) -> (
    NeuralEnginePrefillStatus,
    Option<NeuralEnginePrefillPreparation>,
) {
    let mlp_plan = match plan_neural_engine_mlp_programs(
        model,
        chunking,
        program_cache_directory,
        neural_engine_token_count_override,
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
        built_program_count: 0,
        idle_watch: IdleWatch::new(Instant::now()),
        phase: ProgramMemoryPhase::Building,
        last_waiting_report_at: Instant::now(),
        unsettled_measurement_count: 0,
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
    /// The programs are built and measured, or will never be, so
    /// [`NeuralEnginePrefillPreparation::finish`] applies.
    ReadyToFinish,
}

impl NeuralEnginePrefillPreparation {
    /// A forward is about to use the GPU, so any reading in progress is no longer good.
    pub(crate) fn note_gpu_used(&mut self) {
        self.idle_watch.note_use(Instant::now());
        if let ProgramMemoryPhase::Measuring(measurement) = &self.phase {
            measurement.cancel();
        }
    }

    /// Moves the programs along. Called on the MLX owner thread between requests,
    /// so a chunk already running never sees a program appear part way through.
    pub(crate) fn advance(
        &mut self,
        model: &mut Qwen3_5ResidentModel,
    ) -> NeuralEnginePreparationStep {
        let was_used_this_step = self.advance_build(model);
        let is_idle_long_enough = self.idle_watch.step(Instant::now(), was_used_this_step);
        let phase = std::mem::replace(&mut self.phase, ProgramMemoryPhase::Building);
        self.phase = match phase {
            ProgramMemoryPhase::Building => {
                if !self.is_build_over() {
                    return self.report_build_progress();
                }
                self.begin_memory_phase(model)
            }
            ProgramMemoryPhase::WaitingForIdle(programs) if is_idle_long_enough => {
                self.begin_measuring(programs)
            }
            ProgramMemoryPhase::Measuring(mut measurement) => {
                // Not idle here means a gap between steps showed work nothing announced,
                // and the reading ran through it.
                if !is_idle_long_enough {
                    measurement.cancel();
                }
                match measurement.take_finished() {
                    Some(FinishedMeasurement::Ended(programs, outcome)) => {
                        self.take_measurement(programs, outcome, !is_idle_long_enough)
                    }
                    Some(FinishedMeasurement::Lost) => {
                        ProgramMemoryPhase::Failed("the Neural Engine measurement stopped")
                    }
                    None => ProgramMemoryPhase::Measuring(measurement),
                }
            }
            ProgramMemoryPhase::Known {
                programs,
                wired_bytes,
            } => self.engage_programs(model, programs, wired_bytes),
            unchanged_phase => unchanged_phase,
        };
        match self.phase {
            ProgramMemoryPhase::Finished(_) | ProgramMemoryPhase::Failed(_) => {
                NeuralEnginePreparationStep::ReadyToFinish
            }
            _ => self.report_waiting_progress(),
        }
    }

    /// Steps the build and says whether anything it did used the GPU or Neural Engine.
    /// A kept program's first run on the build thread wires its weights with no package
    /// to write, which is why a finished program counts too.
    fn advance_build(&mut self, model: &Qwen3_5ResidentModel) -> bool {
        if !matches!(self.phase, ProgramMemoryPhase::Building) {
            return false;
        }
        let completed_before = self.program_preparation.completed_program_count();
        let package_writer = ModelProgramPackages {
            model,
            mlp_plan: &self.mlp_plan,
            output_gate_plan: self.output_gate_plan.as_ref(),
        };
        self.program_preparation.advance(&package_writer);
        let has_read_weights = self.release_weight_copies(model);
        if let Some(reason) = self
            .program_preparation
            .first_failure_of(NeuralEngineProgramFamily::OutputGate)
        {
            self.program_preparation
                .fail_waiting_programs_of(NeuralEngineProgramFamily::OutputGate, reason);
        }
        has_read_weights || self.program_preparation.completed_program_count() != completed_before
    }

    fn is_build_over(&self) -> bool {
        self.program_preparation.is_finished()
            || self
                .program_preparation
                .first_failure_of(NeuralEngineProgramFamily::Mlp)
                .is_some()
    }

    fn report_build_progress(&mut self) -> NeuralEnginePreparationStep {
        let completed_program_count = self.program_preparation.completed_program_count();
        if self.last_reported_completed_program_count == Some(completed_program_count) {
            return NeuralEnginePreparationStep::Unchanged;
        }
        self.last_reported_completed_program_count = Some(completed_program_count);
        let preparing_status = self.preparing_status();
        log_preparation_progress(&preparing_status);
        NeuralEnginePreparationStep::Progressed(preparing_status)
    }

    /// The set is built but not attached yet, so say what it is waiting for at the same
    /// cadence as any other long job.
    fn report_waiting_progress(&mut self) -> NeuralEnginePreparationStep {
        if self.last_waiting_report_at.elapsed() < WAITING_REPORT_INTERVAL {
            return NeuralEnginePreparationStep::Unchanged;
        }
        self.last_waiting_report_at = Instant::now();
        let preparing_status = self.preparing_status();
        log_preparation_progress(&preparing_status);
        NeuralEnginePreparationStep::Progressed(preparing_status)
    }

    /// Takes the built programs and decides how their memory will become known.
    fn begin_memory_phase(&mut self, model: &Qwen3_5ResidentModel) -> ProgramMemoryPhase {
        if let Some(reason) = self
            .program_preparation
            .first_failure_of(NeuralEngineProgramFamily::Mlp)
        {
            return ProgramMemoryPhase::Failed(reason);
        }
        self.built_program_count = self.program_preparation.completed_program_count();
        let programs = self.take_built_programs();
        let weight_bytes = self.weight_bytes_of(&programs);
        let recorded_wired_bytes = self.mlp_plan.recorded_wired_bytes(weight_bytes);
        match recorded_wired_bytes {
            // A record is only worth trusting if it admits the programs: one that refuses
            // them would otherwise stay wrong until the next macOS build.
            Some(wired_bytes)
                if self.mlp_plan.fits_ceiling(
                    model,
                    wired_bytes.saturating_add(self.surface_bytes_of(&programs)),
                ) =>
            {
                ProgramMemoryPhase::Known {
                    programs,
                    wired_bytes,
                }
            }
            _ => ProgramMemoryPhase::WaitingForIdle(programs),
        }
    }

    fn take_built_programs(&mut self) -> ProgramsToMeasure<CoreMlProgram> {
        let mut mlp_programs = Vec::new();
        for (_decoder_layer_index, program) in self
            .program_preparation
            .take_built_programs_of(NeuralEngineProgramFamily::Mlp)
        {
            mlp_programs.push(program);
        }
        let mut output_gate_programs = self
            .program_preparation
            .take_built_programs_of(NeuralEngineProgramFamily::OutputGate);
        if let Some(reason) = self
            .program_preparation
            .first_failure_of(NeuralEngineProgramFamily::OutputGate)
        {
            warn_output_gates_stay_on_the_gpu(reason);
            self.output_gate_plan = None;
            // The gates that did build will never attach, so they must neither stay in
            // memory nor be cycled into a reading that does not count them.
            output_gate_programs.clear();
        }
        ProgramsToMeasure {
            mlp_programs,
            output_gate_programs,
        }
    }

    fn begin_measuring(
        &mut self,
        programs: ProgramsToMeasure<CoreMlProgram>,
    ) -> ProgramMemoryPhase {
        let weight_bytes = self.weight_bytes_of(&programs);
        match ProgramMeasurement::start(programs, weight_bytes, SystemWiredMemory) {
            Some(measurement) => ProgramMemoryPhase::Measuring(measurement),
            None => ProgramMemoryPhase::Failed("the Neural Engine measurement could not start"),
        }
    }

    /// Only a settled reading is ever used.
    fn take_measurement(
        &mut self,
        mut programs: ProgramsToMeasure<CoreMlProgram>,
        outcome: MeasurementOutcome,
        has_gpu_been_used: bool,
    ) -> ProgramMemoryPhase {
        match action_after_measurement(outcome, has_gpu_been_used) {
            MeasurementAction::Keep { wired_bytes } => {
                self.mlp_plan
                    .record_wired_bytes(self.weight_bytes_of(&programs), wired_bytes);
                ProgramMemoryPhase::Known {
                    programs,
                    wired_bytes,
                }
            }
            MeasurementAction::MeasureAgain => ProgramMemoryPhase::WaitingForIdle(programs),
            MeasurementAction::MeasureAgainAfterUnsettled => {
                self.unsettled_measurement_count += 1;
                if self.unsettled_measurement_count >= MOST_UNSETTLED_MEASUREMENTS {
                    return ProgramMemoryPhase::Failed(
                        "this Mac was too busy to measure the programs' memory; it will try again when the model next loads",
                    );
                }
                ProgramMemoryPhase::WaitingForIdle(programs)
            }
            MeasurementAction::MeasureAgainWithoutOutputGates => {
                warn_output_gates_stay_on_the_gpu("an output gate program failed its first runs");
                programs.output_gate_programs.clear();
                self.output_gate_plan = None;
                self.idle_watch.note_use(Instant::now());
                ProgramMemoryPhase::WaitingForIdle(programs)
            }
            MeasurementAction::GiveUp => {
                ProgramMemoryPhase::Failed("a Neural Engine program failed while it was measured")
            }
        }
    }

    /// Reading a layer leaves its fp16 copy in MLX's pool. Released at once, because
    /// a request can arrive before the next step and should not compete with it. A
    /// failure costs that request some memory and nothing else, so it stays a warning.
    /// Returns whether a layer was read, which used the GPU.
    fn release_weight_copies(&mut self, model: &Qwen3_5ResidentModel) -> bool {
        let has_written_package = self.program_preparation.take_has_written_package();
        if has_written_package
            && let Err(release_error) = model
                .runtime()
                .synchronize_gpu_stream_and_clear_allocator_cache()
        {
            tracing::warn!(error = %release_error, "could not release Neural Engine weight copies");
        }
        has_written_package
    }

    /// What these programs hold as fp16, the floor of what they can wire.
    fn weight_bytes_of(&self, programs: &ProgramsToMeasure<CoreMlProgram>) -> u64 {
        let mlp_weight_bytes = (programs.mlp_programs.len() as u64)
            .saturating_mul(self.mlp_plan.weight_bytes_per_program());
        match &self.output_gate_plan {
            Some(output_gate_plan) => mlp_weight_bytes.saturating_add(
                (programs.output_gate_programs.len() as u64)
                    .saturating_mul(output_gate_plan.weight_bytes_per_program()),
            ),
            None => mlp_weight_bytes,
        }
    }

    /// These programs' surfaces, which were wired at load and so before any reading.
    fn surface_bytes_of(&self, programs: &ProgramsToMeasure<CoreMlProgram>) -> u64 {
        let mlp_surface_bytes = (programs.mlp_programs.len() as u64)
            .saturating_mul(self.mlp_plan.surface_bytes_per_program());
        match &self.output_gate_plan {
            Some(output_gate_plan) => mlp_surface_bytes.saturating_add(
                (programs.output_gate_programs.len() as u64)
                    .saturating_mul(output_gate_plan.surface_bytes_per_program()),
            ),
            None => mlp_surface_bytes,
        }
    }

    /// Attaches the programs, or says why the model stays on the GPU. A program dropped
    /// on refusal gives its memory straight back.
    fn engage_programs(
        &mut self,
        model: &mut Qwen3_5ResidentModel,
        programs: ProgramsToMeasure<CoreMlProgram>,
        wired_bytes: u64,
    ) -> ProgramMemoryPhase {
        let surface_bytes = self.surface_bytes_of(&programs);
        let measured_program_bytes = wired_bytes.saturating_add(surface_bytes);
        tracing::info!(
            measured_program_bytes,
            wired_bytes,
            "Neural Engine program memory is known"
        );
        if let Some(refused_status) = self
            .mlp_plan
            .refusal_for_measured_bytes(model, measured_program_bytes)
        {
            return ProgramMemoryPhase::Finished(refused_status);
        }
        let active_status = self.mlp_plan.attach(
            model,
            programs.mlp_programs,
            measured_program_bytes,
            surface_bytes,
        );
        if let Some(output_gate_plan) = self.output_gate_plan.take() {
            let output_gate_program_count =
                output_gate_plan.attach(model, programs.output_gate_programs);
            tracing::info!(
                output_gate_program_count,
                "Neural Engine output gate programs loaded"
            );
        }
        ProgramMemoryPhase::Finished(active_status)
    }

    /// The status a reader should now see.
    pub(crate) fn finish(self) -> NeuralEnginePrefillStatus {
        match self.phase {
            ProgramMemoryPhase::Finished(status) => status,
            ProgramMemoryPhase::Failed(reason) => unavailable(reason),
            _ => unavailable("the Neural Engine programs were not ready"),
        }
    }

    fn preparing_status(&self) -> NeuralEnginePrefillStatus {
        let is_building = matches!(self.phase, ProgramMemoryPhase::Building);
        let completed_program_count = if is_building {
            self.program_preparation.completed_program_count()
        } else {
            self.built_program_count
        };
        let total_program_count = self.program_preparation.total_program_count();
        let elapsed_ms =
            u64::try_from(self.program_preparation.elapsed().as_millis()).unwrap_or(u64::MAX);
        let remaining_ms = if is_building {
            estimated_remaining_ms(
                elapsed_ms,
                completed_program_count as u64,
                total_program_count as u64,
            )
        } else {
            let remaining = self.idle_watch.remaining(Instant::now()) + TYPICAL_MEASURING_DURATION;
            u64::try_from(remaining.as_millis()).ok()
        };
        NeuralEnginePrefillStatus::Preparing {
            completed_program_count: u32::try_from(completed_program_count).unwrap_or(u32::MAX),
            total_program_count: u32::try_from(total_program_count).unwrap_or(u32::MAX),
            elapsed_ms,
            estimated_remaining_ms: remaining_ms,
            is_waiting_to_measure: !is_building,
        }
    }
}

/// Serving reports this through the status too, but a measurement that blocks on the
/// programs has only the log to show that a long wait is moving.
fn log_preparation_progress(preparing_status: &NeuralEnginePrefillStatus) {
    if let NeuralEnginePrefillStatus::Preparing {
        completed_program_count,
        total_program_count,
        elapsed_ms,
        estimated_remaining_ms,
        ..
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
    program_cache_directory: Option<&Path>,
    neural_engine_token_count_override: Option<u32>,
) -> NeuralEnginePrefillStatus {
    let (settled_status, preparation) = begin_neural_engine_prefill_preparation(
        model,
        chunking,
        program_cache_directory,
        neural_engine_token_count_override,
    );
    let Some(mut preparation) = preparation else {
        return settled_status;
    };
    loop {
        if matches!(
            preparation.advance(model),
            NeuralEnginePreparationStep::ReadyToFinish
        ) {
            return preparation.finish();
        }
        std::thread::sleep(BLOCKING_PREPARATION_POLL_INTERVAL);
    }
}

#[cfg(test)]
#[path = "neural_engine_prefill_loading_tests.rs"]
mod tests;
