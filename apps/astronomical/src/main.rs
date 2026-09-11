//! User-facing `astronomical` binary. `launch` execs a harness in the
//! foreground; this process is not the model runner.

#![forbid(unsafe_code)]

use std::{
    env, io,
    io::IsTerminal,
    process::{Command, ExitCode},
};

use astronomical_cli::{
    CliCommand, LaunchDependencies, candidate_instances, help_text, parse_command, prepare_launch,
    runtime_instance_from_executable_path,
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
        Err(usage_error) => {
            eprint!("astronomical: {usage_error}\n\n{}", help_text());
            ExitCode::from(2)
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
