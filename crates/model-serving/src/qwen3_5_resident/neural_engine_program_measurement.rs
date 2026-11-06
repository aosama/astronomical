//! Measures what built Neural Engine programs wire on this Mac, on a thread of its own.
//!
//! The reading is only meaningful while MLX sits idle, and it needs the programs to
//! cycle for a few seconds. On its own thread it never delays a request and a stuck
//! Neural Engine cannot stop one being served: a request that arrives simply cancels it,
//! and the measurement starts again the next time the GPU settles. Unloading the model
//! does wait for this thread, as it already waits for the build thread, for at most one
//! prediction's timeout.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use astronomical_memory_policy::neural_engine_program_memory_watch::ProgramMemoryWatch;
use astronomical_neural_engine::core_ml::{CoreMlProgram, system_wired_bytes, wired_growth_bytes};

use super::neural_engine_mlp_prefill::NEURAL_ENGINE_FINISH_TIMEOUT;

/// Every model measured settled within about 2 seconds, so this only bounds a Mac too
/// busy to give a steady reading.
const PROGRAM_MEASURING_LIMIT: Duration = Duration::from_secs(10);

/// A program this measurement can run. Separate from Core ML so its outcomes can be
/// proven without a Neural Engine.
pub(crate) trait MeasurableProgram: Send + 'static {
    /// Runs once on any input, because the driver wires the weights whatever it is.
    fn run_once(&mut self) -> bool;
}

impl MeasurableProgram for CoreMlProgram {
    /// Runs on whatever the input surface holds, because the driver wires the weights
    /// whatever it is, and a zeroed copy per run would only slow a cancel down.
    fn run_once(&mut self) -> bool {
        self.predict_within(NEURAL_ENGINE_FINISH_TIMEOUT).is_ok()
    }
}

/// Where the reading comes from. Separate for the same reason.
pub(crate) trait WiredMemorySource: Send + 'static {
    fn wired_bytes(&mut self) -> u64;
}

pub(crate) struct SystemWiredMemory;

impl WiredMemorySource for SystemWiredMemory {
    fn wired_bytes(&mut self) -> u64 {
        system_wired_bytes()
    }
}

/// The programs a measurement runs, handed back when it ends.
pub(crate) struct ProgramsToMeasure<Program> {
    pub(crate) mlp_programs: Vec<Program>,
    pub(crate) output_gate_programs: Vec<(usize, Program)>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum MeasurementOutcome {
    /// The reading stopped rising, so it can be kept for later loads.
    Settled {
        wired_bytes: u64,
    },
    /// The limit passed first, so the figure may be far below the truth.
    Unsettled,
    /// A request needs the GPU, which would disturb the reading.
    Cancelled,
    MlpProgramFailed,
    OutputGateFailed,
}

/// What a preparation does with an outcome.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum MeasurementAction {
    Keep {
        wired_bytes: u64,
    },
    MeasureAgain,
    /// As `MeasureAgain`, counted because a Mac that never settles must end somewhere.
    MeasureAgainAfterUnsettled,
    MeasureAgainWithoutOutputGates,
    GiveUp,
}

/// A reading the GPU ran through is worthless however it ended, because the growth it
/// saw may be MLX's. A failed program is a fact about the program regardless. An unsettled
/// reading is never used: it can sit at the weights alone, about 10 GB on a model whose
/// programs hold 15 GB, and memory decisions would then trust that for the whole load.
pub(crate) const fn action_after_measurement(
    outcome: MeasurementOutcome,
    has_gpu_been_used: bool,
) -> MeasurementAction {
    match outcome {
        MeasurementOutcome::Settled { .. }
        | MeasurementOutcome::Unsettled
        | MeasurementOutcome::Cancelled
            if has_gpu_been_used =>
        {
            MeasurementAction::MeasureAgain
        }
        MeasurementOutcome::Settled { wired_bytes } => MeasurementAction::Keep { wired_bytes },
        MeasurementOutcome::Unsettled => MeasurementAction::MeasureAgainAfterUnsettled,
        MeasurementOutcome::Cancelled => MeasurementAction::MeasureAgain,
        MeasurementOutcome::MlpProgramFailed => MeasurementAction::GiveUp,
        MeasurementOutcome::OutputGateFailed => MeasurementAction::MeasureAgainWithoutOutputGates,
    }
}

/// Runs every program once, reporting why it stopped early if it did. A cancel is looked
/// for between programs, because a request should not share the Neural Engine with a
/// whole set still to run.
fn run_every_program<Program: MeasurableProgram>(
    programs: &mut ProgramsToMeasure<Program>,
    should_cancel: &AtomicBool,
) -> Option<MeasurementOutcome> {
    for program in &mut programs.mlp_programs {
        if should_cancel.load(Ordering::Relaxed) {
            return Some(MeasurementOutcome::Cancelled);
        }
        if !program.run_once() {
            return Some(MeasurementOutcome::MlpProgramFailed);
        }
    }
    for (_decoder_layer_index, program) in &mut programs.output_gate_programs {
        if should_cancel.load(Ordering::Relaxed) {
            return Some(MeasurementOutcome::Cancelled);
        }
        if !program.run_once() {
            return Some(MeasurementOutcome::OutputGateFailed);
        }
    }
    None
}

/// Cycles the programs until the wired memory they hold stops rising. A failed first
/// reading would make all of the system's wired memory look like growth, so it can
/// only end unsettled.
fn measure_until_settled<Program: MeasurableProgram>(
    programs: &mut ProgramsToMeasure<Program>,
    weight_bytes: u64,
    wired_memory_source: &mut dyn WiredMemorySource,
    should_cancel: &AtomicBool,
    limit: Duration,
) -> MeasurementOutcome {
    let wired_before_bytes = wired_memory_source.wired_bytes();
    let mut watch = ProgramMemoryWatch::new(weight_bytes);
    let started_at = Instant::now();
    loop {
        if should_cancel.load(Ordering::Relaxed) {
            return MeasurementOutcome::Cancelled;
        }
        if let Some(early_outcome) = run_every_program(programs, should_cancel) {
            return early_outcome;
        }
        // Again here: a cancel during those runs means this reading could include MLX.
        if should_cancel.load(Ordering::Relaxed) {
            return MeasurementOutcome::Cancelled;
        }
        if wired_before_bytes > 0 {
            let wired_growth =
                wired_growth_bytes(wired_before_bytes, wired_memory_source.wired_bytes());
            if watch.observe(wired_growth, started_at.elapsed()) {
                return MeasurementOutcome::Settled {
                    wired_bytes: watch.wired_bytes(),
                };
            }
        }
        if started_at.elapsed() >= limit {
            return MeasurementOutcome::Unsettled;
        }
    }
}

/// How a measurement's thread ended.
pub(crate) enum FinishedMeasurement<Program> {
    Ended(ProgramsToMeasure<Program>, MeasurementOutcome),
    /// The thread panicked, which took the programs with it.
    Lost,
}

/// A measurement in progress. Dropping it cancels and waits, so no program outlives the
/// model it was built for.
pub(crate) struct ProgramMeasurement<Program> {
    should_cancel: Arc<AtomicBool>,
    measuring_thread: Option<JoinHandle<(ProgramsToMeasure<Program>, MeasurementOutcome)>>,
}

impl<Program: MeasurableProgram> ProgramMeasurement<Program> {
    /// Starts measuring, or drops the programs when no thread could start.
    pub(crate) fn start(
        mut programs: ProgramsToMeasure<Program>,
        weight_bytes: u64,
        mut wired_memory_source: impl WiredMemorySource,
    ) -> Option<Self> {
        let should_cancel = Arc::new(AtomicBool::new(false));
        let should_cancel_for_thread = Arc::clone(&should_cancel);
        let measuring_thread = std::thread::Builder::new()
            .name("astronomical-neural-engine-measurement".to_owned())
            .spawn(move || {
                let outcome = measure_until_settled(
                    &mut programs,
                    weight_bytes,
                    &mut wired_memory_source,
                    &should_cancel_for_thread,
                    PROGRAM_MEASURING_LIMIT,
                );
                (programs, outcome)
            })
            .ok()?;
        Some(Self {
            should_cancel,
            measuring_thread: Some(measuring_thread),
        })
    }

    pub(crate) fn cancel(&self) {
        self.should_cancel.store(true, Ordering::Relaxed);
    }

    /// How it ended, once the thread has, and only the first time this is asked.
    pub(crate) fn take_finished(&mut self) -> Option<FinishedMeasurement<Program>> {
        if !self.measuring_thread.as_ref()?.is_finished() {
            return None;
        }
        match self.measuring_thread.take()?.join() {
            Ok((programs, outcome)) => Some(FinishedMeasurement::Ended(programs, outcome)),
            Err(_) => Some(FinishedMeasurement::Lost),
        }
    }
}

impl<Program> Drop for ProgramMeasurement<Program> {
    fn drop(&mut self) {
        self.should_cancel.store(true, Ordering::Relaxed);
        if let Some(measuring_thread) = self.measuring_thread.take() {
            let _thread_outcome = measuring_thread.join();
        }
    }
}

#[cfg(test)]
#[path = "neural_engine_program_measurement_tests.rs"]
mod tests;
