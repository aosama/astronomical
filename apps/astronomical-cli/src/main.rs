//! User-facing `astronomical` binary. `launch` execs a harness in the
//! foreground; this process is not the model runner.

#![forbid(unsafe_code)]

use std::{
    env, io,
    io::{IsTerminal, Write},
    process::{Command, ExitCode},
};

use astronomical_cli::{
    CliCommand, LaunchDependencies, candidate_instances, help_text, parse_command, prepare_launch,
    run_schema, run_validate_config, runtime_instance_from_executable_path,
};
use astronomical_config::AstronomicalRuntimeInstance;
use std::os::unix::process::CommandExt;

fn main() -> ExitCode {
    let process_arguments = env::args_os().collect::<Vec<_>>();
    if astronomical_cli::arguments::is_verbose_requested(&process_arguments) {
        let _ = tracing_subscriber::fmt()
            .with_writer(io::stderr)
            .with_max_level(tracing::Level::DEBUG)
            .try_init();
    }
    match parse_command(process_arguments) {
        Ok(CliCommand::Help) => {
            print!("{}", help_text());
            ExitCode::SUCCESS
        }
        Ok(CliCommand::Version) => {
            println!("{}", env!("CARGO_PKG_VERSION"));
            ExitCode::SUCCESS
        }
        Ok(CliCommand::Launch(launch_arguments)) => run_launch(launch_arguments),
        Ok(CliCommand::Schema(schema_arguments)) => run_schema_verb(&schema_arguments),
        Ok(CliCommand::ValidateConfig(validate_arguments)) => {
            run_validate_config_verb(validate_arguments)
        }
        Ok(CliCommand::Respond(respond_arguments)) => run_respond_verb(respond_arguments),
        Ok(CliCommand::Embed(embed_arguments)) => run_embed_verb(embed_arguments),
        Ok(CliCommand::Image(image_arguments)) => run_image_verb(image_arguments),
        Ok(CliCommand::Models(models_command)) => run_models_verb(models_command),
        Ok(CliCommand::Status) => run_status_verb(),
        Ok(CliCommand::Doctor) => run_doctor_verb(),
        Ok(CliCommand::Quantize(quantize_command)) => run_quantize_verb(quantize_command),
        Ok(CliCommand::Vision(vision_command)) => run_vision_verb(&vision_command),
        Ok(
            disk_models_command @ (CliCommand::ModelPrograms | CliCommand::ModelConvertAtlas(_)),
        ) => run_disk_models_command(&disk_models_command),
        Err(usage_error) => {
            eprint!("astronomical: {usage_error}\n\n{}", help_text());
            ExitCode::from(2)
        }
    }
}

fn run_validate_config_verb(
    validate_arguments: astronomical_cli::ValidateConfigArguments,
) -> ExitCode {
    let instance_paths =
        match astronomical_config::AstronomicalInstancePaths::default_location_instance_paths(
            validate_arguments.runtime_instance,
        ) {
            Ok(instance_paths) => instance_paths,
            Err(config_error) => {
                eprintln!("astronomical: {config_error}");
                return ExitCode::FAILURE;
            }
        };
    let mut stdout = io::stdout();
    let mut stderr = io::stderr();
    let mut validate_dependencies = astronomical_cli::ValidateConfigDependencies {
        instance_paths,
        stdout: &mut stdout,
        stderr: &mut stderr,
    };
    match run_validate_config(&validate_arguments, &mut validate_dependencies) {
        Ok(()) => ExitCode::SUCCESS,
        Err(validate_error) => {
            let _ = writeln!(stderr, "{validate_error}");
            ExitCode::FAILURE
        }
    }
}

/// Reads only this Mac's config and disk, so it answers with no daemon running.
fn run_model_programs_verb() -> ExitCode {
    let (runtime_instance, _) = own_runtime_instance_and_ipc_socket_paths();
    let report_outcome = render_model_programs_for(runtime_instance);
    match report_outcome {
        Ok(()) => ExitCode::SUCCESS,
        Err(report_error) => {
            eprintln!("astronomical: {report_error}");
            ExitCode::FAILURE
        }
    }
}

/// The `models` subcommands that read only the disk and need no daemon.
fn run_disk_models_command(disk_models_command: &CliCommand) -> ExitCode {
    match disk_models_command {
        CliCommand::ModelConvertAtlas(convert_arguments) => {
            run_convert_atlas_verb(convert_arguments)
        }
        _ => run_model_programs_verb(),
    }
}

fn run_convert_atlas_verb(convert_arguments: &astronomical_cli::ConvertAtlasArguments) -> ExitCode {
    let (runtime_instance, _) = own_runtime_instance_and_ipc_socket_paths();
    let instance_paths =
        match astronomical_config::AstronomicalInstancePaths::default_location_instance_paths(
            runtime_instance,
        ) {
            Ok(instance_paths) => instance_paths,
            Err(instance_error) => {
                eprintln!("astronomical: {instance_error}");
                return ExitCode::FAILURE;
            }
        };
    let conversion_outcome = astronomical_cli::run_convert_atlas(
        convert_arguments,
        astronomical_cli::ConvertAtlasDependencies {
            models_directory: instance_paths.models_directory(),
            stdout: &mut io::stdout(),
            stderr: &mut io::stderr(),
            progress_interval: std::time::Duration::from_secs(5),
        },
    );
    match conversion_outcome {
        Ok(()) => ExitCode::SUCCESS,
        Err(conversion_error) => {
            eprintln!("astronomical: {conversion_error}");
            ExitCode::FAILURE
        }
    }
}

fn render_model_programs_for(
    runtime_instance: AstronomicalRuntimeInstance,
) -> Result<(), Box<dyn std::error::Error>> {
    let instance_paths =
        astronomical_config::AstronomicalInstancePaths::default_location_instance_paths(
            runtime_instance,
        )?;
    let config =
        astronomical_config::AstronomicalConfig::load_from_instance_paths(instance_paths.clone())?;
    let program_directory = config.neural_engine_program_directory();
    let program_sets = astronomical_config::list_neural_engine_program_sets(&program_directory);
    let mut installed_model_ids = std::collections::BTreeSet::new();
    let discovery = astronomical_config::discover_effective_models(
        &instance_paths.models_directory(),
        config.model_directories(),
    );
    let is_discovery_known = discovery.is_ok();
    if let Ok(discovery) = discovery {
        for discovered_model in discovery.discovered_models {
            installed_model_ids.insert(discovered_model.model_id);
        }
    }
    astronomical_cli::render_model_programs(
        &astronomical_cli::ModelProgramsReport {
            program_directory: &program_directory,
            program_sets: &program_sets,
            installed_model_ids: is_discovery_known.then_some(&installed_model_ids),
        },
        &mut io::stdout(),
    )?;
    Ok(())
}

fn run_schema_verb(schema_arguments: &astronomical_cli::SchemaArguments) -> ExitCode {
    let mut stdout = io::stdout();
    match run_schema(schema_arguments, &mut stdout) {
        Ok(()) => ExitCode::SUCCESS,
        Err(stdout_error) => {
            eprintln!("astronomical: {stdout_error}");
            ExitCode::FAILURE
        }
    }
}

/// The quantize verb is synchronous and in-process: no daemon, no socket,
/// no async runtime. Progress goes to standard error, the report to standard
/// output.
fn run_quantize_verb(quantize_command: astronomical_cli::QuantizeCommand) -> ExitCode {
    let mut stdout = io::stdout();
    let mut stderr = io::stderr();
    let mut quantize_dependencies = astronomical_cli::QuantizeDependencies {
        stdout: &mut stdout,
        stderr: &mut stderr,
        progress_interval: std::time::Duration::from_secs(5),
    };
    let outcome = match &quantize_command {
        astronomical_cli::QuantizeCommand::Plan(plan_arguments) => {
            astronomical_cli::run_quantize_plan(plan_arguments, &mut quantize_dependencies)
        }
        astronomical_cli::QuantizeCommand::Inspect(inspect_arguments) => {
            astronomical_cli::run_quantize_inspect(inspect_arguments, &mut quantize_dependencies)
        }
        astronomical_cli::QuantizeCommand::Convert(convert_arguments) => {
            astronomical_cli::run_quantize_convert(convert_arguments, &mut quantize_dependencies)
        }
    };
    match outcome {
        Ok(()) => ExitCode::SUCCESS,
        Err(quantize_error) => {
            eprintln!("astronomical: {quantize_error}");
            ExitCode::FAILURE
        }
    }
}

/// In-process like quantize, but it needs a runtime for its HTTPS reads. An
/// inspection always exits successfully, because "no tower" is an answer; a match
/// exits with failure unless adding the tower could proceed, so scripts can gate on
/// it; and an add fails whenever it refuses, having written nothing.
fn run_vision_verb(vision_command: &astronomical_cli::VisionCommand) -> ExitCode {
    let vision_runtime = match build_current_thread_runtime() {
        Ok(vision_runtime) => vision_runtime,
        Err(runtime_error) => {
            eprintln!("astronomical: could not start the async runtime: {runtime_error}");
            return ExitCode::FAILURE;
        }
    };
    let mut stdout = io::stdout();
    let mut stderr = io::stderr();
    let mut vision_dependencies = astronomical_cli::VisionDependencies {
        stdout: &mut stdout,
        stderr: &mut stderr,
        progress_interval: std::time::Duration::from_secs(5),
    };
    let outcome = match vision_command {
        astronomical_cli::VisionCommand::Inspect(inspect_arguments) => vision_runtime
            .block_on(astronomical_cli::run_vision_inspect(
                inspect_arguments,
                &mut vision_dependencies,
            ))
            .map(exit_code_for_inspection),
        astronomical_cli::VisionCommand::Match(match_arguments) => vision_runtime
            .block_on(astronomical_cli::run_vision_match(
                match_arguments,
                &mut vision_dependencies,
            ))
            .map(exit_code_for_match),
        astronomical_cli::VisionCommand::Add(add_arguments) => vision_runtime
            .block_on(astronomical_cli::run_vision_add(
                add_arguments,
                &mut vision_dependencies,
            ))
            .map(exit_code_for_addition),
    };
    match outcome {
        Ok(exit_code) => exit_code,
        Err(vision_error) => {
            eprintln!("astronomical: {vision_error}");
            ExitCode::FAILURE
        }
    }
}

fn exit_code_for_inspection(_status: astronomical_vision_tower::TowerStatus) -> ExitCode {
    ExitCode::SUCCESS
}

fn exit_code_for_addition(_report: astronomical_vision_tower::TowerAdditionReport) -> ExitCode {
    ExitCode::SUCCESS
}

fn exit_code_for_match(verdict: astronomical_vision_tower::BaseMatchVerdict) -> ExitCode {
    if verdict.allows_adding() {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

fn run_launch(launch_arguments: astronomical_cli::LaunchArguments) -> ExitCode {
    let executable_path = env::current_exe().ok();
    let preferred_instance = executable_path
        .as_deref()
        .map(runtime_instance_from_executable_path)
        .unwrap_or(AstronomicalRuntimeInstance::Development);
    let candidate_bind_addresses = candidate_instances(preferred_instance)
        .map(AstronomicalRuntimeInstance::loopback_socket_addr)
        .to_vec();
    let path_value = env::var_os("PATH").unwrap_or_default();
    let is_interactive = io::stdin().is_terminal();
    let mut stdin = io::stdin().lock();
    let mut stderr = io::stderr();
    let prepared_launch = match prepare_launch(
        launch_arguments,
        LaunchDependencies {
            candidate_bind_addresses,
            path_value,
            is_interactive,
            stdin: &mut stdin,
            stderr: &mut stderr,
            http_timeout: astronomical_cli::verbs::launch::DEFAULT_LOOPBACK_TIMEOUT,
        },
    ) {
        Ok(prepared_launch) => prepared_launch,
        Err(launch_error) => {
            eprintln!("{launch_error}");
            return ExitCode::FAILURE;
        }
    };

    let mut command = Command::new(&prepared_launch.program);
    command.args(&prepared_launch.arguments);
    for (environment_name, environment_value) in prepared_launch.extra_environment {
        command.env(environment_name, environment_value);
    }
    let exec_error = command.exec();
    eprintln!(
        "{}",
        astronomical_cli::LaunchError::ToolStartFailed {
            program: prepared_launch.program,
            cause: exec_error.to_string(),
        }
    );
    ExitCode::FAILURE
}

/// How long a one-shot daemon verb waits on one protocol stage with nothing heard.
/// Short enough that a wedged daemon cannot hang the calling script forever. A model
/// load or a request making progress counts as hearing from the daemon, so `respond`
/// and `embed` can outlast this through a long prefill or a queue.
const DAEMON_VERB_STAGE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(120);
/// Wait between asking what the daemon is doing. Only a silent wait polls, so a steady
/// token stream sends none of these, and one status request per interval costs the
/// daemon nothing noticeable during prefill.
const DAEMON_WORK_POLL_INTERVAL: std::time::Duration = std::time::Duration::from_secs(2);
/// How long one load may stay reported before a waiting verb gives up. Set above the
/// daemon's own 60-second budget for a load, because the daemon owns deciding that a
/// load has hung; this only ends the wait if the daemon itself wedges while still
/// answering status.
const MODEL_LOAD_STALL_BOUND: std::time::Duration = std::time::Duration::from_secs(120);

/// Current-thread async runtime for one-shot verbs: a single connection at a
/// time never pays the multi-threaded runtime's thread spawn cost.
fn build_current_thread_runtime() -> Result<tokio::runtime::Runtime, std::io::Error> {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
}

/// The executable's own channel plus that channel's IPC socket paths: the
/// daemon verbs probe only their own channel (strict isolation), and the
/// not-running disclosure names this channel so a failure says whose daemon
/// is down instead of leaving the user to guess (issue #1392).
fn own_runtime_instance_and_ipc_socket_paths()
-> (AstronomicalRuntimeInstance, Vec<std::path::PathBuf>) {
    let current_executable_path = env::current_exe()
        .inspect_err(|exe_error| eprintln!("astronomical: {exe_error}"))
        .ok();
    let preferred_instance = current_executable_path
        .as_deref()
        .map(runtime_instance_from_executable_path)
        .unwrap_or(AstronomicalRuntimeInstance::Development);
    let candidate_socket_paths = candidate_instances(preferred_instance)
        .into_iter()
        .filter_map(|runtime_instance| {
            astronomical_config::AstronomicalInstancePaths::default_location_instance_paths(
                runtime_instance,
            )
            .ok()
        })
        .map(|instance_paths| instance_paths.ipc_socket_file_path())
        .collect();
    (preferred_instance, candidate_socket_paths)
}

fn run_respond_verb(respond_arguments: astronomical_cli::RespondArguments) -> ExitCode {
    let respond_runtime = match build_current_thread_runtime() {
        Ok(respond_runtime) => respond_runtime,
        Err(runtime_error) => {
            eprintln!("astronomical: {runtime_error}");
            return ExitCode::FAILURE;
        }
    };
    let mut stdout = io::stdout();
    let mut stderr = io::stderr();
    let (respond_runtime_instance, respond_candidate_socket_paths) =
        own_runtime_instance_and_ipc_socket_paths();
    let mut respond_dependencies = astronomical_cli::RespondDependencies {
        candidate_socket_paths: respond_candidate_socket_paths,
        runtime_instance: respond_runtime_instance,
        stdout: &mut stdout,
        stderr: &mut stderr,
        request_timeout: DAEMON_VERB_STAGE_TIMEOUT,
        download_stall_bound: astronomical_cli::model_lifecycle::DOWNLOAD_STALL_BOUND,
        download_poll_interval: astronomical_cli::model_lifecycle::DOWNLOAD_POLL_INTERVAL,
        daemon_work_poll_interval: DAEMON_WORK_POLL_INTERVAL,
        model_load_stall_bound: MODEL_LOAD_STALL_BOUND,
    };
    match respond_runtime.block_on(astronomical_cli::run_respond(
        &respond_arguments,
        &mut respond_dependencies,
    )) {
        Ok(()) => ExitCode::SUCCESS,
        // A missing or capability-mismatched model is a usage error, so it
        // exits 2 like a bad flag.
        Err(astronomical_cli::RespondError::ModelUnavailable { reason }) => {
            let _ = writeln!(stderr, "astronomical: {reason}");
            ExitCode::from(2)
        }
        Err(respond_error) => {
            let _ = writeln!(stderr, "astronomical: {respond_error}");
            ExitCode::FAILURE
        }
    }
}

fn run_embed_verb(embed_arguments: astronomical_cli::EmbedArguments) -> ExitCode {
    let embed_runtime = match build_current_thread_runtime() {
        Ok(embed_runtime) => embed_runtime,
        Err(runtime_error) => {
            eprintln!("astronomical: {runtime_error}");
            return ExitCode::FAILURE;
        }
    };
    let mut stdin = io::stdin();
    let mut stdout = io::stdout();
    let mut stderr = io::stderr();
    let (embed_runtime_instance, embed_candidate_socket_paths) =
        own_runtime_instance_and_ipc_socket_paths();
    let mut embed_dependencies = astronomical_cli::EmbedDependencies {
        candidate_socket_paths: embed_candidate_socket_paths,
        runtime_instance: embed_runtime_instance,
        stdin: &mut stdin,
        stdout: &mut stdout,
        stderr: &mut stderr,
        request_timeout: DAEMON_VERB_STAGE_TIMEOUT,
        download_stall_bound: astronomical_cli::model_lifecycle::DOWNLOAD_STALL_BOUND,
        download_poll_interval: astronomical_cli::model_lifecycle::DOWNLOAD_POLL_INTERVAL,
        daemon_work_poll_interval: DAEMON_WORK_POLL_INTERVAL,
        model_load_stall_bound: MODEL_LOAD_STALL_BOUND,
    };
    match embed_runtime.block_on(astronomical_cli::run_embed(
        &embed_arguments,
        &mut embed_dependencies,
    )) {
        Ok(()) => ExitCode::SUCCESS,
        // A missing or capability-mismatched model is a usage error, so it
        // exits 2 like a bad flag.
        Err(astronomical_cli::EmbedError::ModelUnavailable { reason }) => {
            let _ = writeln!(stderr, "astronomical: {reason}");
            ExitCode::from(2)
        }
        Err(embed_error) => {
            let _ = writeln!(stderr, "astronomical: {embed_error}");
            ExitCode::FAILURE
        }
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

fn run_models_verb(models_command: astronomical_cli::ModelsCommand) -> ExitCode {
    let models_runtime = match build_current_thread_runtime() {
        Ok(models_runtime) => models_runtime,
        Err(runtime_error) => {
            eprintln!("astronomical: {runtime_error}");
            return ExitCode::FAILURE;
        }
    };
    let mut stdout = io::stdout();
    let mut stderr = io::stderr();
    let (models_runtime_instance, models_candidate_socket_paths) =
        own_runtime_instance_and_ipc_socket_paths();
    let mut models_dependencies = astronomical_cli::ModelsDependencies {
        candidate_socket_paths: models_candidate_socket_paths,
        runtime_instance: models_runtime_instance,
        stdout: &mut stdout,
        stderr: &mut stderr,
        request_timeout: DAEMON_VERB_STAGE_TIMEOUT,
        download_stall_bound: astronomical_cli::model_lifecycle::DOWNLOAD_STALL_BOUND,
        download_poll_interval: astronomical_cli::model_lifecycle::DOWNLOAD_POLL_INTERVAL,
    };
    match models_runtime.block_on(astronomical_cli::run_models(
        &models_command,
        &mut models_dependencies,
    )) {
        Ok(()) => ExitCode::SUCCESS,
        // A missing model on `models download` is a usage error, so it
        // exits 2 like a bad flag.
        Err(astronomical_cli::ModelsError::ModelUnavailable { reason }) => {
            let _ = writeln!(stderr, "astronomical: {reason}");
            ExitCode::from(2)
        }
        Err(models_error) => {
            let _ = writeln!(stderr, "astronomical: {models_error}");
            ExitCode::FAILURE
        }
    }
}

/// Shorter than the daemon-verb stage bound: a doctor probe only needs to
/// learn reachability, and a diagnostic that can hang minutes per channel
/// would defeat its own purpose.
const DOCTOR_PROBE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

fn run_doctor_verb() -> ExitCode {
    let doctor_runtime = match build_current_thread_runtime() {
        Ok(doctor_runtime) => doctor_runtime,
        Err(runtime_error) => {
            eprintln!("astronomical: {runtime_error}");
            return ExitCode::FAILURE;
        }
    };
    let executable_path = match env::current_exe() {
        Ok(raw_executable_path) => raw_executable_path
            .canonicalize()
            .unwrap_or(raw_executable_path),
        Err(exe_error) => {
            eprintln!("astronomical: {exe_error}");
            return ExitCode::FAILURE;
        }
    };
    // Both channels, Stable first: the mismatch a doctor report exists to
    // expose is only visible when both sides are printed side by side.
    let mut channel_socket_paths = Vec::new();
    for runtime_instance in [
        AstronomicalRuntimeInstance::Stable,
        AstronomicalRuntimeInstance::Development,
    ] {
        match astronomical_config::AstronomicalInstancePaths::default_location_instance_paths(
            runtime_instance,
        ) {
            Ok(instance_paths) => {
                channel_socket_paths.push((runtime_instance, instance_paths.ipc_socket_file_path()))
            }
            Err(config_error) => eprintln!("astronomical: {config_error}"),
        }
    }
    let mut stdout = io::stdout();
    let mut doctor_dependencies = astronomical_cli::DoctorDependencies {
        executable_path,
        channel_socket_paths,
        stdout: &mut stdout,
        request_timeout: DOCTOR_PROBE_TIMEOUT,
    };
    match doctor_runtime.block_on(astronomical_cli::run_doctor(&mut doctor_dependencies)) {
        Ok(()) => ExitCode::SUCCESS,
        Err(doctor_error) => {
            eprintln!("astronomical: {doctor_error}");
            ExitCode::FAILURE
        }
    }
}

fn run_status_verb() -> ExitCode {
    let status_runtime = match build_current_thread_runtime() {
        Ok(status_runtime) => status_runtime,
        Err(runtime_error) => {
            eprintln!("astronomical: {runtime_error}");
            return ExitCode::FAILURE;
        }
    };
    let mut stdout = io::stdout();
    let (status_runtime_instance, status_candidate_socket_paths) =
        own_runtime_instance_and_ipc_socket_paths();
    let mut status_dependencies = astronomical_cli::StatusDependencies {
        candidate_socket_paths: status_candidate_socket_paths,
        runtime_instance: status_runtime_instance,
        stdout: &mut stdout,
        request_timeout: DAEMON_VERB_STAGE_TIMEOUT,
    };
    match status_runtime.block_on(astronomical_cli::run_status(&mut status_dependencies)) {
        Ok(()) => ExitCode::SUCCESS,
        Err(status_error) => {
            let _ = writeln!(io::stderr(), "astronomical: {status_error}");
            ExitCode::FAILURE
        }
    }
}
