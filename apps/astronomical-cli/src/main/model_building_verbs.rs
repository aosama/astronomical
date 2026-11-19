//! Runners for the verbs that build a new model folder from installed ones:
//! `vision` gives a text model a tower, and `decision` writes a calibrated
//! or trained copy of a decision model. Each turns its outcome into an exit
//! code (2 for a usage-shaped failure).

use std::io::{self, Write};
use std::process::ExitCode;

use astronomical_cli::CliCommand;

use super::{
    DAEMON_VERB_STAGE_TIMEOUT, DAEMON_WORK_POLL_INTERVAL, MODEL_LOAD_STALL_BOUND,
    build_current_thread_runtime, own_runtime_instance_and_ipc_socket_paths, run_vision_verb,
};

/// Runs a verb that builds a new model folder from installed ones: `vision`
/// for a text model with a tower, `decision` for a calibrated copy.
pub(super) fn run_model_building_verb(model_building_command: &CliCommand) -> ExitCode {
    match model_building_command {
        CliCommand::Vision(vision_command) => run_vision_verb(vision_command),
        CliCommand::Decision(decision_command) => run_decision_verb(decision_command),
        // `main` routes only those two verbs here.
        _ => ExitCode::FAILURE,
    }
}

/// How often `decision calibrate` and `decision train` report progress and
/// time left on long steps.
const DECISION_PROGRESS_INTERVAL: std::time::Duration = std::time::Duration::from_secs(5);

fn run_decision_verb(decision_command: &astronomical_cli::DecisionCommand) -> ExitCode {
    let decision_runtime = match build_current_thread_runtime() {
        Ok(decision_runtime) => decision_runtime,
        Err(runtime_error) => {
            eprintln!("astronomical: {runtime_error}");
            return ExitCode::FAILURE;
        }
    };
    let (runtime_instance, candidate_socket_paths) = own_runtime_instance_and_ipc_socket_paths();
    let instance_paths = match config::AstronomicalInstancePaths::default_location_instance_paths(
        runtime_instance,
    ) {
        Ok(instance_paths) => instance_paths,
        Err(instance_error) => {
            eprintln!("astronomical: {instance_error}");
            return ExitCode::FAILURE;
        }
    };
    let mut stdout = io::stdout();
    let mut stderr = io::stderr();
    let mut dependencies = astronomical_cli::DecisionToolDependencies {
        candidate_socket_paths,
        runtime_instance,
        loopback_address: runtime_instance.loopback_socket_addr(),
        instance_paths,
        stdout: &mut stdout,
        stderr: &mut stderr,
        wait_bounds: astronomical_cli::verbs::decide::DecisionWaitBounds {
            request_timeout: DAEMON_VERB_STAGE_TIMEOUT,
            model_load_stall_bound: MODEL_LOAD_STALL_BOUND,
            daemon_work_poll_interval: DAEMON_WORK_POLL_INTERVAL,
        },
        download_stall_bound: astronomical_cli::model_lifecycle::DOWNLOAD_STALL_BOUND,
        download_poll_interval: astronomical_cli::model_lifecycle::DOWNLOAD_POLL_INTERVAL,
        progress_interval: DECISION_PROGRESS_INTERVAL,
    };
    let decision_outcome = match decision_command {
        astronomical_cli::DecisionCommand::Calibrate(calibrate_arguments) => decision_runtime
            .block_on(astronomical_cli::run_calibrate(
                calibrate_arguments,
                &mut dependencies,
            )),
        astronomical_cli::DecisionCommand::Train(train_arguments) => decision_runtime.block_on(
            astronomical_cli::run_train(train_arguments, &mut dependencies),
        ),
    };
    match decision_outcome {
        Ok(()) => ExitCode::SUCCESS,
        Err(decision_error) => {
            let _ = writeln!(stderr, "astronomical: {decision_error}");
            if decision_error.is_usage_error() {
                ExitCode::from(2)
            } else {
                ExitCode::FAILURE
            }
        }
    }
}
