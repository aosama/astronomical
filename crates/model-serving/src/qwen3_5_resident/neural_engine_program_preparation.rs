//! Moves Neural Engine programs from planned to built without holding up a model.
//!
//! Building a set takes a minute or more whenever macOS has discarded its prepared
//! copies, and a model serves on the GPU without them, so nobody should wait for it.
//! Compiling and loading run on a thread of their own because neither touches MLX.
//! Only reading a layer's weights needs the MLX owner thread, which is why packages
//! are written there, one per call, from between requests. That keeps each pause
//! short and leaves no weight read competing with a generation.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use astronomical_neural_engine::program_shape::ProgramShape;

/// A program's first run wires its weights outside the memory ceiling, and this only
/// tops up between requests. So this also bounds what a request starting now can find
/// wired beside it: a few programs, rather than a whole set of 15 GB.
const MOST_PROGRAMS_AT_BUILD_THREAD: usize = 4;
const BUILD_THREAD_STOPPED: &str = "the Neural Engine program thread stopped";
const PACKAGE_DIRECTORY_PREFIX: &str = "astronomical-neural-engine-";
const NO_SCRATCH_DIRECTORY: &str = "no scratch directory for the Core ML program";

static NEXT_PACKAGE_NUMBER: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum NeuralEngineProgramFamily {
    Mlp,
    OutputGate,
}

impl NeuralEngineProgramFamily {
    const fn label(self) -> &'static str {
        match self {
            Self::Mlp => "mlp",
            Self::OutputGate => "output-gate",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ProgramIdentity {
    pub(crate) family: NeuralEngineProgramFamily,
    pub(crate) decoder_layer_index: usize,
}

pub(crate) struct PlannedProgram {
    pub(crate) identity: ProgramIdentity,
    pub(crate) shape: ProgramShape,
    /// Where the compiled program stays between loads; `None` builds it from a
    /// temporary package and keeps nothing.
    pub(crate) kept_program_path: Option<PathBuf>,
}

impl PlannedProgram {
    fn needs_package(&self) -> bool {
        match &self.kept_program_path {
            Some(kept_program_path) => !kept_program_path.is_dir(),
            None => true,
        }
    }
}

/// Compile input for one program, removed as soon as Core ML holds its own compiled
/// copy, because each one is nearly a whole layer's weights on disk.
pub(crate) struct ProgramPackageDirectory {
    pub(crate) path: PathBuf,
}

impl ProgramPackageDirectory {
    pub(crate) fn create(identity: ProgramIdentity) -> Option<Self> {
        // The counter keeps two preparations apart: a replaced engine's package can
        // still be on its way out when the next engine writes the same layer's.
        let package_number = NEXT_PACKAGE_NUMBER.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "{PACKAGE_DIRECTORY_PREFIX}{}-{package_number}-{}-{}",
            std::process::id(),
            identity.family.label(),
            identity.decoder_layer_index,
        ));
        std::fs::create_dir_all(&path).ok()?;
        Some(Self { path })
    }
}

impl Drop for ProgramPackageDirectory {
    fn drop(&mut self) {
        if let Err(removal_error) = std::fs::remove_dir_all(&self.path) {
            tracing::warn!(error = %removal_error, "could not remove a Neural Engine program package");
        }
    }
}

pub(crate) struct WrittenProgramPackage {
    // Held so the package outlives the build that reads it.
    _package_directory: ProgramPackageDirectory,
    pub(crate) package_path: PathBuf,
}

pub(crate) struct ProgramBuildJob {
    pub(crate) identity: ProgramIdentity,
    pub(crate) shape: ProgramShape,
    pub(crate) kept_program_path: Option<PathBuf>,
    pub(crate) package: Option<WrittenProgramPackage>,
}

pub(crate) enum ProgramBuildOutcome<Program> {
    Built(Program),
    /// The kept path could not produce a usable program, so this load falls back to
    /// a temporary package, which needs the weights read on the MLX owner thread.
    NeedsUncachedPackage,
    Failed(&'static str),
}

/// Turns one job into a program. Runs on the build thread, so it must not touch MLX.
pub(crate) trait ProgramBuilder: Send + 'static {
    type Program: Send + 'static;

    fn build(&self, job: &ProgramBuildJob) -> ProgramBuildOutcome<Self::Program>;
}

/// Writes one program's compile input. Runs on the MLX owner thread, because this is
/// the one step that reads the model's weights.
pub(crate) trait ProgramPackageWriter {
    fn write_package(
        &self,
        identity: ProgramIdentity,
        package_directory: &Path,
    ) -> Result<PathBuf, &'static str>;
}

/// One finished build. Carries the shape so a program sent back for a temporary
/// package can be planned again without any other record of it.
struct FinishedProgramBuild<Program> {
    identity: ProgramIdentity,
    shape: ProgramShape,
    outcome: ProgramBuildOutcome<Program>,
}

pub(crate) struct ProgramPreparation<Program> {
    waiting_programs: VecDeque<PlannedProgram>,
    job_sender: Option<Sender<ProgramBuildJob>>,
    finished_build_receiver: Receiver<FinishedProgramBuild<Program>>,
    /// Only one package exists at a time, because each is about a layer's weights
    /// on disk and the build thread can only use them one by one.
    package_in_flight: Option<ProgramIdentity>,
    /// Sent and not yet answered, so a dead build thread fails each under its own
    /// family: losing only gates must not turn the MLP programs off.
    programs_at_build_thread: Vec<ProgramIdentity>,
    should_stop: Arc<AtomicBool>,
    build_thread: Option<JoinHandle<()>>,
    total_program_count: usize,
    built_programs: Vec<(ProgramIdentity, Program)>,
    failed_programs: Vec<(ProgramIdentity, &'static str)>,
    has_written_package: bool,
    started_at: Instant,
}

/// The worker that wrote a package, read from the directory's name.
fn package_owner_process_id(directory_name: &str) -> Option<u32> {
    let name_after_prefix = directory_name.strip_prefix(PACKAGE_DIRECTORY_PREFIX)?;
    let (process_id, _rest_of_name) = name_after_prefix.split_once('-')?;
    process_id.parse().ok()
}

/// Through the `kill` tool because this crate holds no unsafe code. It runs at most
/// once per directory left in the temporary folder, once per preparation, so its cost
/// stays far below what it guards against. A failure to ask keeps the package.
fn process_has_stopped(process_id: u32) -> bool {
    let probe_outcome = std::process::Command::new("/bin/kill")
        .arg("-0")
        .arg(process_id.to_string())
        .stderr(std::process::Stdio::null())
        .status();
    match probe_outcome {
        Ok(exit_status) => !exit_status.success(),
        Err(_) => false,
    }
}

/// The daemon stops a worker with SIGKILL, so `Drop` never removes the package it was
/// using, and the model now reports ready while packages are still being written.
/// Each one left behind is nearly a layer's weights in the temporary directory.
fn remove_packages_of_stopped_workers(temporary_directory: &Path) {
    let Ok(directory_entries) = std::fs::read_dir(temporary_directory) else {
        return;
    };
    for directory_entry in directory_entries.flatten() {
        let directory_name = directory_entry.file_name();
        let Some(owner_process_id) = package_owner_process_id(&directory_name.to_string_lossy())
        else {
            continue;
        };
        if !process_has_stopped(owner_process_id) {
            continue;
        }
        if let Err(removal_error) = std::fs::remove_dir_all(directory_entry.path()) {
            tracing::warn!(error = %removal_error, "could not remove a stopped worker's Neural Engine package");
        }
    }
}

fn run_program_builds<Builder: ProgramBuilder>(
    builder: Builder,
    job_receiver: Receiver<ProgramBuildJob>,
    finished_build_sender: Sender<FinishedProgramBuild<Builder::Program>>,
    should_stop: Arc<AtomicBool>,
) {
    while let Ok(build_job) = job_receiver.recv() {
        // Checked per job because a stop must not sit through a queue of programs:
        // a swap waits for this thread before it loads the next model.
        if should_stop.load(Ordering::Relaxed) {
            return;
        }
        let outcome = builder.build(&build_job);
        let (identity, shape) = (build_job.identity, build_job.shape);
        // Before the outcome goes out, so the next package never lands beside this
        // one and the disk holds one layer's worth at a time.
        drop(build_job);
        let finished_build = FinishedProgramBuild {
            identity,
            shape,
            outcome,
        };
        if finished_build_sender.send(finished_build).is_err() {
            return;
        }
    }
}

impl<Program: Send + 'static> ProgramPreparation<Program> {
    /// `None` when the system cannot start a thread, which leaves the model on the GPU.
    pub(crate) fn start<Builder>(
        planned_programs: Vec<PlannedProgram>,
        builder: Builder,
    ) -> Option<Self>
    where
        Builder: ProgramBuilder<Program = Program>,
    {
        remove_packages_of_stopped_workers(&std::env::temp_dir());
        let (job_sender, job_receiver) = mpsc::channel();
        let (finished_build_sender, finished_build_receiver) = mpsc::channel();
        let should_stop = Arc::new(AtomicBool::new(false));
        let build_thread_stop_signal = Arc::clone(&should_stop);
        let build_thread = thread::Builder::new()
            .name("astronomical-neural-engine-programs".to_owned())
            .spawn(move || {
                run_program_builds(
                    builder,
                    job_receiver,
                    finished_build_sender,
                    build_thread_stop_signal,
                );
            })
            .ok()?;
        Some(Self {
            total_program_count: planned_programs.len(),
            waiting_programs: planned_programs.into(),
            job_sender: Some(job_sender),
            finished_build_receiver,
            package_in_flight: None,
            programs_at_build_thread: Vec::new(),
            should_stop,
            build_thread: Some(build_thread),
            built_programs: Vec::new(),
            failed_programs: Vec::new(),
            has_written_package: false,
            started_at: Instant::now(),
        })
    }

    /// Collects finished builds and starts more. Never waits for a build, so the
    /// only time it costs the caller is reading one layer's weights.
    pub(crate) fn advance(&mut self, package_writer: &dyn ProgramPackageWriter) {
        self.collect_finished_builds();
        self.send_ready_builds(package_writer);
    }

    fn collect_finished_builds(&mut self) {
        loop {
            match self.finished_build_receiver.try_recv() {
                Ok(finished_build) => self.record_finished_build(finished_build),
                Err(TryRecvError::Empty) => return,
                Err(TryRecvError::Disconnected) => {
                    self.fail_everything_unresolved();
                    return;
                }
            }
        }
    }

    fn record_finished_build(&mut self, finished_build: FinishedProgramBuild<Program>) {
        let identity = finished_build.identity;
        if self.package_in_flight == Some(identity) {
            self.package_in_flight = None;
        }
        for sent_position in 0..self.programs_at_build_thread.len() {
            if self.programs_at_build_thread[sent_position] == identity {
                self.programs_at_build_thread.swap_remove(sent_position);
                break;
            }
        }
        match finished_build.outcome {
            ProgramBuildOutcome::Built(program) => self.built_programs.push((identity, program)),
            ProgramBuildOutcome::Failed(failure_reason) => {
                self.failed_programs.push((identity, failure_reason));
            }
            ProgramBuildOutcome::NeedsUncachedPackage => {
                self.waiting_programs.push_front(PlannedProgram {
                    identity,
                    shape: finished_build.shape,
                    kept_program_path: None,
                });
            }
        }
    }

    /// A dead build thread will never answer, and a status stuck on "preparing"
    /// would promise a notification that can never come.
    fn fail_everything_unresolved(&mut self) {
        self.package_in_flight = None;
        for identity in std::mem::take(&mut self.programs_at_build_thread) {
            self.failed_programs.push((identity, BUILD_THREAD_STOPPED));
        }
        for planned_program in std::mem::take(&mut self.waiting_programs) {
            self.failed_programs
                .push((planned_program.identity, BUILD_THREAD_STOPPED));
        }
    }

    /// Gives up on what has not been started in `family`, so a set that can no longer
    /// be used reads no more weights and compiles nothing more for it.
    pub(crate) fn fail_waiting_programs_of(
        &mut self,
        family: NeuralEngineProgramFamily,
        failure_reason: &'static str,
    ) {
        for planned_program in std::mem::take(&mut self.waiting_programs) {
            if planned_program.identity.family == family {
                self.failed_programs
                    .push((planned_program.identity, failure_reason));
            } else {
                self.waiting_programs.push_back(planned_program);
            }
        }
    }

    /// Sends every program that needs no weights, up to the next one that does, and
    /// writes at most one package per call.
    fn send_ready_builds(&mut self, package_writer: &dyn ProgramPackageWriter) {
        while let Some(next_program) = self.waiting_programs.front() {
            if next_program.needs_package() && self.package_in_flight.is_some() {
                return;
            }
            if self.programs_at_build_thread.len() >= MOST_PROGRAMS_AT_BUILD_THREAD {
                return;
            }
            let Some(planned_program) = self.waiting_programs.pop_front() else {
                return;
            };
            if !planned_program.needs_package() {
                self.send_build_job(planned_program, None);
                continue;
            }
            self.send_build_with_package(planned_program, package_writer);
            return;
        }
    }

    fn send_build_with_package(
        &mut self,
        planned_program: PlannedProgram,
        package_writer: &dyn ProgramPackageWriter,
    ) {
        let identity = planned_program.identity;
        match write_program_package(identity, package_writer) {
            Ok(written_package) => {
                self.has_written_package = true;
                self.package_in_flight = Some(identity);
                self.send_build_job(planned_program, Some(written_package));
            }
            Err(failure_reason) => self.failed_programs.push((identity, failure_reason)),
        }
    }

    fn send_build_job(
        &mut self,
        planned_program: PlannedProgram,
        package: Option<WrittenProgramPackage>,
    ) {
        self.programs_at_build_thread.push(planned_program.identity);
        let build_job = ProgramBuildJob {
            identity: planned_program.identity,
            shape: planned_program.shape,
            kept_program_path: planned_program.kept_program_path,
            package,
        };
        let is_sent = match &self.job_sender {
            Some(job_sender) => job_sender.send(build_job).is_ok(),
            None => false,
        };
        if !is_sent {
            self.fail_everything_unresolved();
        }
    }

    /// Finished either way, so the figure a reader sees always reaches the total.
    pub(crate) fn completed_program_count(&self) -> usize {
        self.built_programs.len() + self.failed_programs.len()
    }

    pub(crate) const fn total_program_count(&self) -> usize {
        self.total_program_count
    }

    pub(crate) fn elapsed(&self) -> Duration {
        self.started_at.elapsed()
    }

    pub(crate) fn is_finished(&self) -> bool {
        self.completed_program_count() >= self.total_program_count
    }

    /// Whether a weight was read since the last call, which leaves MLX buffers worth
    /// releasing.
    pub(crate) const fn take_has_written_package(&mut self) -> bool {
        std::mem::replace(&mut self.has_written_package, false)
    }

    /// The first reason a program of `family` failed, if one has.
    pub(crate) fn first_failure_of(
        &self,
        family: NeuralEngineProgramFamily,
    ) -> Option<&'static str> {
        for (identity, failure_reason) in &self.failed_programs {
            if identity.family == family {
                return Some(failure_reason);
            }
        }
        None
    }

    /// Hands over the programs of `family`, in decoder-layer order.
    pub(crate) fn take_built_programs_of(
        &mut self,
        family: NeuralEngineProgramFamily,
    ) -> Vec<(usize, Program)> {
        let mut requested_programs = Vec::new();
        let mut other_programs = Vec::new();
        for (identity, program) in self.built_programs.drain(..) {
            if identity.family == family {
                requested_programs.push((identity.decoder_layer_index, program));
            } else {
                other_programs.push((identity, program));
            }
        }
        self.built_programs = other_programs;
        requested_programs.sort_by_key(decoder_layer_index_of);
        requested_programs
    }
}

fn decoder_layer_index_of<Program>(layer_program: &(usize, Program)) -> usize {
    layer_program.0
}

fn write_program_package(
    identity: ProgramIdentity,
    package_writer: &dyn ProgramPackageWriter,
) -> Result<WrittenProgramPackage, &'static str> {
    let package_directory =
        ProgramPackageDirectory::create(identity).ok_or(NO_SCRATCH_DIRECTORY)?;
    let package_path = package_writer.write_package(identity, &package_directory.path)?;
    Ok(WrittenProgramPackage {
        _package_directory: package_directory,
        package_path,
    })
}

impl<Program> Drop for ProgramPreparation<Program> {
    /// Waits for the thread, because a swap releases this before it loads the next
    /// model and the two must not overlap in memory. At most one program's build
    /// is left to finish, which is a few seconds at worst.
    fn drop(&mut self) {
        self.should_stop.store(true, Ordering::Relaxed);
        drop(self.job_sender.take());
        if let Some(build_thread) = self.build_thread.take() {
            let _build_thread_panicked = build_thread.join().is_err();
        }
    }
}

#[cfg(test)]
#[path = "neural_engine_program_preparation_tests.rs"]
mod tests;
