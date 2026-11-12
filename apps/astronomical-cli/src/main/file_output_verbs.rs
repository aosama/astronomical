//! Runners for the verbs that save their result to a file, `image` and
//! `speak`: each builds its runtime and dependencies, runs its journey, and
//! turns the outcome into an exit code (2 for a usage-shaped failure).

use std::io::{self, Write};
use std::process::ExitCode;

use astronomical_cli::CliCommand;

use super::{
    DAEMON_VERB_STAGE_TIMEOUT, build_current_thread_runtime,
    own_runtime_instance_and_ipc_socket_paths,
};

pub(super) fn run_file_output_verb(file_output_command: CliCommand) -> ExitCode {
    match file_output_command {
        CliCommand::Speak(speak_arguments) => run_speak_verb(speak_arguments),
        CliCommand::Image(image_arguments) => run_image_verb(image_arguments),
        // The dispatch routes only those two commands here.
        _ => ExitCode::FAILURE,
    }
}

/// An image is the slowest verb: the daemon stops one at 15 minutes (its own
/// bound), so the CLI waits a little longer than that and reports the daemon's
/// verdict instead of cutting the run off first.
const IMAGE_GENERATION_WAIT_BOUND: std::time::Duration = std::time::Duration::from_secs(16 * 60);

/// How often an image run reports elapsed time on standard error.
const IMAGE_PROGRESS_INTERVAL: std::time::Duration = std::time::Duration::from_secs(5);

fn run_image_verb(image_arguments: astronomical_cli::ImageArguments) -> ExitCode {
    let image_runtime = match build_current_thread_runtime() {
        Ok(image_runtime) => image_runtime,
        Err(runtime_error) => {
            eprintln!("astronomical: {runtime_error}");
            return ExitCode::FAILURE;
        }
    };
    let default_output_directory = match std::env::current_dir() {
        Ok(current_directory) => current_directory,
        Err(directory_error) => {
            eprintln!("astronomical: could not read the working directory: {directory_error}");
            return ExitCode::FAILURE;
        }
    };
    let mut stdout = io::stdout();
    let mut stderr = io::stderr();
    let (image_runtime_instance, image_candidate_socket_paths) =
        own_runtime_instance_and_ipc_socket_paths();
    let mut image_dependencies = astronomical_cli::ImageDependencies {
        candidate_socket_paths: image_candidate_socket_paths,
        runtime_instance: image_runtime_instance,
        default_output_directory,
        stdout: &mut stdout,
        stderr: &mut stderr,
        request_timeout: DAEMON_VERB_STAGE_TIMEOUT,
        generation_bound: IMAGE_GENERATION_WAIT_BOUND,
        progress_interval: IMAGE_PROGRESS_INTERVAL,
        download_stall_bound: astronomical_cli::model_lifecycle::DOWNLOAD_STALL_BOUND,
        download_poll_interval: astronomical_cli::model_lifecycle::DOWNLOAD_POLL_INTERVAL,
    };
    match image_runtime.block_on(astronomical_cli::run_image(
        &image_arguments,
        &mut image_dependencies,
    )) {
        Ok(()) => ExitCode::SUCCESS,
        // A missing model or an unusable output path is fixed by changing the
        // command, so it exits 2 like a bad flag.
        Err(image_error) if image_error.is_usage_error() => {
            let _ = writeln!(stderr, "astronomical: {image_error}");
            ExitCode::from(2)
        }
        Err(image_error) => {
            let _ = writeln!(stderr, "astronomical: {image_error}");
            ExitCode::FAILURE
        }
    }
}

/// The daemon bounds a speech request at 10 minutes of compute; the CLI
/// waits a little longer and reports the daemon's verdict instead.
const SPEECH_WAIT_BOUND: std::time::Duration = std::time::Duration::from_secs(11 * 60);

/// How often a speech run reports elapsed time on standard error.
const SPEECH_PROGRESS_INTERVAL: std::time::Duration = std::time::Duration::from_secs(5);

fn run_speak_verb(speak_arguments: astronomical_cli::SpeakArguments) -> ExitCode {
    let speak_runtime = match build_current_thread_runtime() {
        Ok(speak_runtime) => speak_runtime,
        Err(runtime_error) => {
            eprintln!("astronomical: {runtime_error}");
            return ExitCode::FAILURE;
        }
    };
    let default_output_directory = match std::env::current_dir() {
        Ok(current_directory) => current_directory,
        Err(directory_error) => {
            eprintln!("astronomical: could not read the working directory: {directory_error}");
            return ExitCode::FAILURE;
        }
    };
    let mut stdin = io::stdin().lock();
    let mut stdout = io::stdout();
    let mut stderr = io::stderr();
    let (speak_runtime_instance, speak_candidate_socket_paths) =
        own_runtime_instance_and_ipc_socket_paths();
    let mut speak_dependencies = astronomical_cli::SpeakDependencies {
        candidate_socket_paths: speak_candidate_socket_paths,
        runtime_instance: speak_runtime_instance,
        default_output_directory,
        stdin: &mut stdin,
        stdout: &mut stdout,
        stderr: &mut stderr,
        request_timeout: DAEMON_VERB_STAGE_TIMEOUT,
        generation_bound: SPEECH_WAIT_BOUND,
        progress_interval: SPEECH_PROGRESS_INTERVAL,
        download_stall_bound: astronomical_cli::model_lifecycle::DOWNLOAD_STALL_BOUND,
        download_poll_interval: astronomical_cli::model_lifecycle::DOWNLOAD_POLL_INTERVAL,
    };
    let speak_outcome = speak_runtime.block_on(astronomical_cli::run_speak(
        &speak_arguments,
        &mut speak_dependencies,
    ));
    drop(speak_dependencies);
    match speak_outcome {
        Ok(()) => ExitCode::SUCCESS,
        Err(speak_error) => {
            let _ = writeln!(stderr, "astronomical: {speak_error}");
            if speak_error.is_usage_error() {
                ExitCode::from(2)
            } else {
                ExitCode::FAILURE
            }
        }
    }
}
