use std::fs;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use astronomical_neural_engine::program_shape::ProgramShape;

use super::{
    NeuralEngineProgramFamily, PlannedProgram, ProgramBuildJob, ProgramBuildOutcome,
    ProgramBuilder, ProgramIdentity, ProgramPackageWriter, ProgramPreparation,
};
use super::{
    PACKAGE_DIRECTORY_PREFIX, package_owner_process_id, process_has_stopped,
    remove_packages_of_stopped_workers,
};

const WAIT_LIMIT: Duration = Duration::from_secs(10);
const PACKAGE_WRITE_FAILURE: &str = "the weights could not be read";

fn program_shape() -> ProgramShape {
    ProgramShape::new(64, 64, 64).expect("a valid shape")
}

fn identity(family: NeuralEngineProgramFamily, decoder_layer_index: usize) -> ProgramIdentity {
    ProgramIdentity {
        family,
        decoder_layer_index,
    }
}

fn mlp(decoder_layer_index: usize) -> ProgramIdentity {
    identity(NeuralEngineProgramFamily::Mlp, decoder_layer_index)
}

fn planned(identity: ProgramIdentity, kept_program_path: Option<PathBuf>) -> PlannedProgram {
    PlannedProgram {
        identity,
        shape: program_shape(),
        kept_program_path,
    }
}

/// Builds a program named after its layer, and answers one layer with whatever the
/// test asked for, the first time only.
struct ScriptedBuilder {
    first_answers: Arc<Mutex<Vec<(usize, ScriptedAnswer)>>>,
    seen_jobs: Sender<(ProgramIdentity, bool)>,
    build_delay: Duration,
}

#[derive(Clone, Copy)]
enum ScriptedAnswer {
    NeedsUncachedPackage,
    Failed(&'static str),
}

impl ProgramBuilder for ScriptedBuilder {
    type Program = usize;

    fn build(&self, job: &ProgramBuildJob) -> ProgramBuildOutcome<usize> {
        self.seen_jobs
            .send((job.identity, job.package.is_some()))
            .expect("the test is listening");
        std::thread::sleep(self.build_delay);
        let mut first_answers = self.first_answers.lock().expect("the script");
        for scripted_position in 0..first_answers.len() {
            if first_answers[scripted_position].0 == job.identity.decoder_layer_index {
                return match first_answers.remove(scripted_position).1 {
                    ScriptedAnswer::NeedsUncachedPackage => {
                        ProgramBuildOutcome::NeedsUncachedPackage
                    }
                    ScriptedAnswer::Failed(reason) => ProgramBuildOutcome::Failed(reason),
                };
            }
        }
        ProgramBuildOutcome::Built(job.identity.decoder_layer_index)
    }
}

fn start_preparation(
    planned_programs: Vec<PlannedProgram>,
    first_answers: Vec<(usize, ScriptedAnswer)>,
) -> (ProgramPreparation<usize>, Receiver<(ProgramIdentity, bool)>) {
    start_preparation_building_every(Duration::ZERO, planned_programs, first_answers)
}

fn start_preparation_building_every(
    build_delay: Duration,
    planned_programs: Vec<PlannedProgram>,
    first_answers: Vec<(usize, ScriptedAnswer)>,
) -> (ProgramPreparation<usize>, Receiver<(ProgramIdentity, bool)>) {
    let (seen_jobs, seen_job_receiver) = mpsc::channel();
    let builder = ScriptedBuilder {
        first_answers: Arc::new(Mutex::new(first_answers)),
        seen_jobs,
        build_delay,
    };
    let preparation =
        ProgramPreparation::start(planned_programs, builder).expect("the thread starts");
    (preparation, seen_job_receiver)
}

struct WritingPackages;

impl ProgramPackageWriter for WritingPackages {
    fn write_package(
        &self,
        _identity: ProgramIdentity,
        package_directory: &Path,
    ) -> Result<PathBuf, &'static str> {
        let package_path = package_directory.join("program.mlpackage");
        fs::write(&package_path, b"package").expect("the package writes");
        Ok(package_path)
    }
}

struct FailingPackages;

impl ProgramPackageWriter for FailingPackages {
    fn write_package(
        &self,
        _identity: ProgramIdentity,
        _package_directory: &Path,
    ) -> Result<PathBuf, &'static str> {
        Err(PACKAGE_WRITE_FAILURE)
    }
}

fn advance_until_finished(
    preparation: &mut ProgramPreparation<usize>,
    package_writer: &dyn ProgramPackageWriter,
) {
    let deadline = Instant::now() + WAIT_LIMIT;
    while !preparation.is_finished() {
        assert!(Instant::now() < deadline, "the preparation never finished");
        preparation.advance(package_writer);
        std::thread::sleep(Duration::from_millis(1));
    }
}

fn built_layers(
    preparation: &mut ProgramPreparation<usize>,
    family: NeuralEngineProgramFamily,
) -> Vec<usize> {
    let mut layers = Vec::new();
    for (decoder_layer_index, _program) in preparation.take_built_programs_of(family) {
        layers.push(decoder_layer_index);
    }
    layers
}

#[test]
fn should_build_kept_programs_without_reading_any_weights() {
    let kept_programs = tempfile::tempdir().expect("a scratch directory");
    let (mut preparation, _seen_jobs) = start_preparation(
        vec![
            planned(mlp(0), Some(kept_programs.path().to_owned())),
            planned(mlp(1), Some(kept_programs.path().to_owned())),
        ],
        Vec::new(),
    );

    advance_until_finished(&mut preparation, &FailingPackages);

    assert_eq!(preparation.completed_program_count(), 2);
    assert!(!preparation.take_has_written_package());
    assert_eq!(
        preparation.first_failure_of(NeuralEngineProgramFamily::Mlp),
        None
    );
    assert_eq!(
        built_layers(&mut preparation, NeuralEngineProgramFamily::Mlp),
        vec![0, 1]
    );
}

#[test]
fn should_write_at_most_one_package_per_advance() {
    let (mut preparation, seen_jobs) = start_preparation(
        vec![planned(mlp(0), None), planned(mlp(1), None)],
        Vec::new(),
    );

    preparation.advance(&WritingPackages);

    assert_eq!(seen_jobs.recv_timeout(WAIT_LIMIT).unwrap(), (mlp(0), true));
    assert!(seen_jobs.recv_timeout(Duration::from_millis(50)).is_err());
    assert!(!preparation.is_finished());
    advance_until_finished(&mut preparation, &WritingPackages);
    assert!(preparation.take_has_written_package());
}

#[test]
fn should_retry_from_a_temporary_package_when_a_kept_program_is_unusable() {
    let kept_programs = tempfile::tempdir().expect("a scratch directory");
    let (mut preparation, seen_jobs) = start_preparation(
        vec![planned(mlp(0), Some(kept_programs.path().to_owned()))],
        vec![(0, ScriptedAnswer::NeedsUncachedPackage)],
    );

    advance_until_finished(&mut preparation, &WritingPackages);

    assert_eq!(seen_jobs.recv_timeout(WAIT_LIMIT).unwrap(), (mlp(0), false));
    assert_eq!(seen_jobs.recv_timeout(WAIT_LIMIT).unwrap(), (mlp(0), true));
    assert_eq!(
        built_layers(&mut preparation, NeuralEngineProgramFamily::Mlp),
        vec![0]
    );
}

#[test]
fn should_count_a_failed_program_as_finished_and_name_its_family() {
    let gate = identity(NeuralEngineProgramFamily::OutputGate, 3);
    let (mut preparation, _seen_jobs) = start_preparation(
        vec![planned(mlp(0), None), planned(gate, None)],
        vec![(3, ScriptedAnswer::Failed("the gate failed"))],
    );

    advance_until_finished(&mut preparation, &WritingPackages);

    assert_eq!(preparation.completed_program_count(), 2);
    assert_eq!(
        preparation.first_failure_of(NeuralEngineProgramFamily::Mlp),
        None
    );
    assert_eq!(
        preparation.first_failure_of(NeuralEngineProgramFamily::OutputGate),
        Some("the gate failed")
    );
}

#[test]
fn should_fail_a_program_whose_package_cannot_be_written() {
    let (mut preparation, _seen_jobs) = start_preparation(vec![planned(mlp(0), None)], Vec::new());

    advance_until_finished(&mut preparation, &FailingPackages);

    assert_eq!(
        preparation.first_failure_of(NeuralEngineProgramFamily::Mlp),
        Some(PACKAGE_WRITE_FAILURE)
    );
}

#[test]
fn should_finish_at_once_when_nothing_is_planned() {
    let (preparation, _seen_jobs) = start_preparation(Vec::new(), Vec::new());

    assert!(preparation.is_finished());
    assert_eq!(preparation.total_program_count(), 0);
}

#[test]
fn should_hand_over_one_family_in_layer_order_and_keep_the_other() {
    let gate = identity(NeuralEngineProgramFamily::OutputGate, 0);
    let (mut preparation, _seen_jobs) = start_preparation(
        vec![
            planned(mlp(1), None),
            planned(gate, None),
            planned(mlp(0), None),
        ],
        Vec::new(),
    );
    advance_until_finished(&mut preparation, &WritingPackages);

    assert_eq!(
        built_layers(&mut preparation, NeuralEngineProgramFamily::Mlp),
        vec![0, 1]
    );
    assert_eq!(
        built_layers(&mut preparation, NeuralEngineProgramFamily::OutputGate),
        vec![0]
    );
}

#[test]
fn should_stop_promptly_when_released_with_programs_still_queued() {
    const QUEUED_PROGRAM_COUNT: usize = 100;
    let kept_programs = tempfile::tempdir().expect("a scratch directory");
    let mut planned_programs = Vec::new();
    for decoder_layer_index in 0..QUEUED_PROGRAM_COUNT {
        planned_programs.push(planned(
            mlp(decoder_layer_index),
            Some(kept_programs.path().to_owned()),
        ));
    }
    let (mut preparation, seen_jobs) =
        start_preparation_building_every(Duration::from_millis(20), planned_programs, Vec::new());
    preparation.advance(&FailingPackages);

    drop(preparation);

    // Dropping joins the thread, so every job it will ever start has been seen.
    assert!(seen_jobs.iter().count() < QUEUED_PROGRAM_COUNT);
}

/// Stands in for a build thread that dies part way through a set.
struct PanickingBuilder;

impl ProgramBuilder for PanickingBuilder {
    type Program = usize;

    fn build(&self, _job: &ProgramBuildJob) -> ProgramBuildOutcome<usize> {
        panic!("the build thread stops here");
    }
}

#[test]
fn should_fail_each_unanswered_program_under_its_own_family_when_the_thread_dies() {
    let kept_programs = tempfile::tempdir().expect("a scratch directory");
    let gate = identity(NeuralEngineProgramFamily::OutputGate, 0);
    let mut preparation = ProgramPreparation::start(
        vec![
            planned(gate, Some(kept_programs.path().to_owned())),
            planned(identity(NeuralEngineProgramFamily::OutputGate, 1), None),
        ],
        PanickingBuilder,
    )
    .expect("the thread starts");

    advance_until_finished(&mut preparation, &WritingPackages);

    assert_eq!(preparation.completed_program_count(), 2);
    assert_eq!(
        preparation.first_failure_of(NeuralEngineProgramFamily::Mlp),
        None
    );
    assert!(
        preparation
            .first_failure_of(NeuralEngineProgramFamily::OutputGate)
            .is_some()
    );
}

#[test]
fn should_never_report_more_programs_than_the_total_after_a_retry() {
    let kept_programs = tempfile::tempdir().expect("a scratch directory");
    let (mut preparation, _seen_jobs) = start_preparation(
        vec![
            planned(mlp(0), Some(kept_programs.path().to_owned())),
            planned(mlp(1), Some(kept_programs.path().to_owned())),
        ],
        vec![(0, ScriptedAnswer::NeedsUncachedPackage)],
    );

    advance_until_finished(&mut preparation, &WritingPackages);

    assert_eq!(preparation.completed_program_count(), 2);
}

#[test]
fn should_give_up_on_waiting_programs_of_one_family_only() {
    let gate = identity(NeuralEngineProgramFamily::OutputGate, 3);
    let (mut preparation, seen_jobs) =
        start_preparation(vec![planned(gate, None), planned(mlp(0), None)], Vec::new());

    preparation.fail_waiting_programs_of(NeuralEngineProgramFamily::OutputGate, "given up");
    advance_until_finished(&mut preparation, &WritingPackages);

    assert_eq!(seen_jobs.recv_timeout(WAIT_LIMIT).unwrap(), (mlp(0), true));
    assert_eq!(
        preparation.first_failure_of(NeuralEngineProgramFamily::OutputGate),
        Some("given up")
    );
    assert_eq!(preparation.completed_program_count(), 2);
}

#[test]
fn should_read_the_owner_from_a_package_name() {
    assert_eq!(
        package_owner_process_id("astronomical-neural-engine-4242-7-mlp-3"),
        Some(4242)
    );
    assert_eq!(
        package_owner_process_id("astronomical-neural-engine-x-mlp"),
        None
    );
    assert_eq!(
        package_owner_process_id("astronomical-neural-engine-4242"),
        None
    );
    assert_eq!(package_owner_process_id("unrelated-4242-7"), None);
}

fn process_id_that_has_exited() -> u32 {
    let mut exited_child = std::process::Command::new("/usr/bin/true")
        .spawn()
        .expect("the child starts");
    let process_id = exited_child.id();
    exited_child.wait().expect("the child exits");
    process_id
}

#[test]
fn should_tell_a_stopped_process_from_a_running_one() {
    assert!(process_has_stopped(process_id_that_has_exited()));
    assert!(!process_has_stopped(std::process::id()));
}

#[test]
fn should_remove_only_the_packages_of_stopped_workers() {
    let temporary_directory = tempfile::tempdir().expect("a scratch directory");
    let stopped_package = temporary_directory.path().join(format!(
        "{PACKAGE_DIRECTORY_PREFIX}{}-0-mlp-0",
        process_id_that_has_exited()
    ));
    let running_package = temporary_directory.path().join(format!(
        "{PACKAGE_DIRECTORY_PREFIX}{}-0-mlp-0",
        std::process::id()
    ));
    let unrelated_directory = temporary_directory.path().join("unrelated-1-0");
    for directory in [&stopped_package, &running_package, &unrelated_directory] {
        fs::create_dir_all(directory).expect("the directory exists");
    }

    remove_packages_of_stopped_workers(temporary_directory.path());

    assert!(!stopped_package.exists());
    assert!(running_package.is_dir());
    assert!(unrelated_directory.is_dir());
}
