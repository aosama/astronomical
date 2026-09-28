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
    if env::args_os().any(|argument| argument == "--verbose" || argument == "-v") {
        let _ = tracing_subscriber::fmt()
            .with_writer(io::stderr)
            .with_max_level(tracing::Level::DEBUG)
            .try_init();
    }
    match parse_command(env::args_os()) {
        Ok(CliCommand::Help) => {
            print!("{}", help_text());
            ExitCode::SUCCESS
        }
        Ok(CliCommand::Version) => {
            println!("{}", env!("CARGO_PKG_VERSION"));
            ExitCode::SUCCESS
        }
        Ok(CliCommand::Launch(launch_arguments)) => run_launch(launch_arguments),
        Ok(CliCommand::Schema(schema_arguments)) => {
            let mut stdout = io::stdout();
            match run_schema(&schema_arguments, &mut stdout) {
                Ok(()) => ExitCode::SUCCESS,
                Err(stdout_error) => {
                    eprintln!("astronomical: {stdout_error}");
                    ExitCode::FAILURE
                }
            }
        }
        Ok(CliCommand::ValidateConfig(validate_arguments)) => {
            run_validate_config_verb(validate_arguments)
        }
        Ok(CliCommand::Respond(respond_arguments)) => run_respond_verb(respond_arguments),
        Ok(CliCommand::Embed(embed_arguments)) => run_embed_verb(embed_arguments),
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
            let _ = write!(stderr, "{validate_error}\n");
            ExitCode::FAILURE
        }
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
            http_timeout: astronomical_cli::launch::DEFAULT_LOOPBACK_TIMEOUT,
        },
    ) {
        Ok(prepared_launch) => prepared_launch,
        Err(launch_error) => {
            eprintln!("{launch_error}");
            return ExitCode::FAILURE;
        }
    };

    let mut command = Command::new(&prepared_launch.program);
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

/// Bound for each protocol stage of a one-shot daemon verb. Long enough for
/// a full local answer, short enough that a wedged daemon cannot hang the
/// calling script forever.
const DAEMON_VERB_STAGE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(120);

/// Current-thread async runtime for one-shot verbs: a single connection at a
/// time never pays the multi-threaded runtime's thread spawn cost.
fn build_current_thread_runtime() -> Result<tokio::runtime::Runtime, std::io::Error> {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
}

/// IPC sockets of every instance the running executable could serve, most
/// preferred first: its own instance, then the others.
fn candidate_ipc_socket_paths() -> Vec<std::path::PathBuf> {
    let current_executable_path = env::current_exe()
        .inspect_err(|exe_error| eprintln!("astronomical: {exe_error}"))
        .ok();
    let preferred_instance = current_executable_path
        .as_deref()
        .map(runtime_instance_from_executable_path)
        .unwrap_or(AstronomicalRuntimeInstance::Development);
    candidate_instances(preferred_instance)
        .into_iter()
        .filter_map(|runtime_instance| {
            astronomical_config::AstronomicalInstancePaths::default_location_instance_paths(
                runtime_instance,
            )
            .ok()
        })
        .map(|instance_paths| instance_paths.ipc_socket_file_path())
        .collect()
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
    let mut respond_dependencies = astronomical_cli::RespondDependencies {
        candidate_socket_paths: candidate_ipc_socket_paths(),
        stdout: &mut stdout,
        stderr: &mut stderr,
        request_timeout: DAEMON_VERB_STAGE_TIMEOUT,
    };
    match respond_runtime.block_on(astronomical_cli::run_respond(
        &respond_arguments,
        &mut respond_dependencies,
    )) {
        Ok(()) => ExitCode::SUCCESS,
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
    let mut embed_dependencies = astronomical_cli::EmbedDependencies {
        candidate_socket_paths: candidate_ipc_socket_paths(),
        stdin: &mut stdin,
        stdout: &mut stdout,
        request_timeout: DAEMON_VERB_STAGE_TIMEOUT,
    };
    match embed_runtime.block_on(astronomical_cli::run_embed(
        &embed_arguments,
        &mut embed_dependencies,
    )) {
        Ok(()) => ExitCode::SUCCESS,
        Err(embed_error) => {
            let _ = writeln!(stderr, "astronomical: {embed_error}");
            ExitCode::FAILURE
        }
    }
}
