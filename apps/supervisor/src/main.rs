#![forbid(unsafe_code)]

use std::{
    env, io, net::SocketAddr, path::PathBuf, process::ExitCode, sync::Arc, sync::RwLock,
    time::Duration,
};

use astronomical_config::{AstronomicalConfig, AstronomicalConfigError, AstronomicalInstancePaths};
use astronomical_supervisor::{
    ApplicationBuildIdentity, AstronomicalInstanceLock, GenerationPerformanceLog,
    ImageGenerationTimeouts, ResolvedRuntimeConfigError, ResolvedRuntimeConfigResolver,
    ShutdownController, WorkerControlError, WorkerHandle, build_application_with_full_control,
};
use thiserror::Error;
use tokio::{net::TcpListener, signal};
use tracing_subscriber::{EnvFilter, fmt};

const WORKER_MODEL_LOAD_TIMEOUT: Duration = Duration::from_secs(60);

mod daemon_arguments;
use daemon_arguments::{DaemonArguments, DaemonCommand};

#[tokio::main]
async fn main() -> ExitCode {
    let daemon_command = match DaemonArguments::parse(std::env::args_os()) {
        Ok(daemon_command) => daemon_command,
        Err(argument_error) => {
            eprintln!(
                "astronomicald: {argument_error}\n\n{}",
                daemon_arguments::help_text()
            );
            return ExitCode::from(2);
        }
    };
    let daemon_arguments = match daemon_command {
        DaemonCommand::Help => {
            print!("{}", daemon_arguments::help_text());
            return ExitCode::SUCCESS;
        }
        DaemonCommand::Version => {
            println!(
                "{}",
                ApplicationBuildIdentity::current().command_line_version()
            );
            return ExitCode::SUCCESS;
        }
        DaemonCommand::Run(daemon_arguments) => daemon_arguments,
    };
    let instance_paths = match daemon_arguments.resolve_instance_paths() {
        Ok(instance_paths) => instance_paths,
        Err(configuration_error) => {
            eprintln!("astronomicald configuration failed: {configuration_error}");
            return ExitCode::FAILURE;
        }
    };
    let _instance_lock =
        match AstronomicalInstanceLock::acquire(&instance_paths.instance_lock_file_path()) {
            Ok(instance_lock) => instance_lock,
            Err(lock_error) => {
                eprintln!("astronomicald startup failed: {lock_error}");
                return ExitCode::FAILURE;
            }
        };
    let user_config = match AstronomicalConfig::load_from_instance_paths(instance_paths.clone()) {
        Ok(user_config) => user_config,
        Err(configuration_error) => {
            eprintln!("astronomicald configuration failed: {configuration_error}");
            return ExitCode::FAILURE;
        }
    };
    let _logging_guard = match initialize_tracing(&user_config) {
        Ok(logging_guard) => logging_guard,
        Err(initialization_error) => {
            eprintln!("astronomicald logging initialization failed: {initialization_error}");
            return ExitCode::FAILURE;
        }
    };
    match run_daemon(instance_paths, user_config).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(daemon_error) => {
            eprintln!("astronomicald failed: {daemon_error}");
            ExitCode::FAILURE
        }
    }
}

fn initialize_tracing(
    user_config: &AstronomicalConfig,
) -> Result<tracing_appender::non_blocking::WorkerGuard, DaemonError> {
    let logging_config = user_config.logging()?;
    std::fs::create_dir_all(logging_config.directory()).map_err(DaemonError::CreateLogDirectory)?;
    let file_appender = tracing_appender::rolling::Builder::new()
        .rotation(astronomical_supervisor::astronomical_log_rotation())
        .filename_prefix("supervisor")
        .filename_suffix("log")
        .max_log_files(logging_config.retained_files())
        .build(logging_config.directory())
        .map_err(DaemonError::CreateLogAppender)?;
    // The local API must remain responsive when its diagnostic disk is slow.
    // Keep a small lossy queue instead of tracing-appender's much larger default.
    let (file_writer, guard) = tracing_appender::non_blocking::NonBlockingBuilder::default()
        .buffered_lines_limit(logging_config.buffered_line_limit())
        .lossy(true)
        .thread_name("astronomical-supervisor-log-writer")
        .finish(file_appender);
    let filter = EnvFilter::new(logging_config.level().as_str());
    fmt()
        .with_env_filter(filter)
        .with_ansi(false)
        .with_thread_ids(true)
        .with_thread_names(true)
        .with_target(true)
        .with_writer(file_writer)
        .try_init()
        .map_err(|source| DaemonError::InitializeTracing {
            reason: source.to_string(),
        })?;
    Ok(guard)
}

async fn run_daemon(
    instance_paths: AstronomicalInstancePaths,
    user_config: AstronomicalConfig,
) -> Result<(), DaemonError> {
    let runtime_config_resolver = ResolvedRuntimeConfigResolver::for_instance(
        instance_paths.clone(),
        fallback_worker_executable_path()?,
    );
    let resolved_runtime_config = runtime_config_resolver
        .resolve(&user_config)
        .map_err(DaemonError::ResolveRuntimeConfig)?;
    let supervisor_bind_address = resolved_runtime_config
        .bind_address
        .parse::<SocketAddr>()
        .map_err(DaemonError::ParseResolvedBindAddress)?;
    let listener = TcpListener::bind(supervisor_bind_address)
        .await
        .map_err(DaemonError::BindSupervisor)?;
    let bound_supervisor_address = listener
        .local_addr()
        .map_err(DaemonError::ReadBoundSupervisorAddress)?;
    tracing::info!(
        bind_address = %bound_supervisor_address,
        channel = instance_paths
            .runtime_instance()
            .map_or("explicit", astronomical_config::AstronomicalRuntimeInstance::as_str),
        version = ApplicationBuildIdentity::current().version,
        commit = ApplicationBuildIdentity::current().commit,
        configuration_generation = %resolved_runtime_config.configuration_generation,
        "Astronomical REST API listener bound"
    );
    let logging_config = user_config.logging()?;
    let performance_log = GenerationPerformanceLog::open(logging_config.directory())
        .map_err(DaemonError::CreatePerformanceLog)?;
    let worker_handle =
        match WorkerHandle::launch_with_startup_configuration_and_image_generation_timeouts(
            &resolved_runtime_config.worker_executable_path,
            WORKER_MODEL_LOAD_TIMEOUT,
            ImageGenerationTimeouts::DEFAULT,
            performance_log,
            Arc::clone(&resolved_runtime_config.model_policy_catalog),
            resolved_runtime_config.worker_startup_configuration(),
        )
        .await
        {
            Ok(worker_handle) => worker_handle,
            Err(worker_start_error) => {
                eprintln!("astronomicald worker unavailable: {worker_start_error}");
                WorkerHandle::unavailable()
            }
        };
    let reloadable_config = Arc::new(RwLock::new(resolved_runtime_config));
    let shutdown_controller = ShutdownController::new();
    let internal_shutdown_receiver = shutdown_controller.subscribe();
    let application = build_application_with_full_control(
        worker_handle.clone(),
        Arc::clone(&reloadable_config),
        runtime_config_resolver,
        shutdown_controller,
    );

    println!("astronomicald listening on http://{bound_supervisor_address}");

    let serve_result = axum::serve(listener, application)
        .with_graceful_shutdown(shutdown_signal(internal_shutdown_receiver))
        .await
        .map_err(DaemonError::ServeHttp);
    let worker_shutdown_result = worker_handle
        .shutdown()
        .await
        .map(|_| ())
        .map_err(DaemonError::ShutdownWorker);

    serve_result?;
    worker_shutdown_result?;

    Ok(())
}

fn fallback_worker_executable_path() -> Result<PathBuf, DaemonError> {
    let daemon_executable_path =
        env::current_exe().map_err(DaemonError::ResolveCurrentExecutable)?;
    let executable_directory = daemon_executable_path
        .parent()
        .ok_or(DaemonError::MissingExecutableDirectory)?;
    let worker_executable_name = format!(
        "astronomical-inference-worker{}",
        std::env::consts::EXE_SUFFIX
    );
    let fallback_worker_executable_path = executable_directory.join(worker_executable_name);
    Ok(fallback_worker_executable_path)
}

async fn shutdown_signal(mut internal_shutdown_receiver: tokio::sync::watch::Receiver<bool>) {
    let ctrl_c_signal = async {
        if let Err(signal_error) = signal::ctrl_c().await {
            eprintln!("failed to listen for Ctrl+C shutdown signal: {signal_error}");
            std::future::pending::<()>().await;
        }
    };

    #[cfg(unix)]
    let terminate_signal = async {
        match signal::unix::signal(signal::unix::SignalKind::terminate()) {
            Ok(mut signal_stream) => {
                signal_stream.recv().await;
            }
            Err(signal_error) => {
                eprintln!("failed to listen for SIGTERM shutdown signal: {signal_error}");
                std::future::pending::<()>().await;
            }
        }
    };

    #[cfg(not(unix))]
    let terminate_signal = std::future::pending::<()>();

    tokio::select! {
        () = ctrl_c_signal => {}
        () = terminate_signal => {}
        _ = internal_shutdown_receiver.changed() => {
            tracing::info!("internal shutdown signal received from /v1/control/shutdown");
        }
    }
}

#[derive(Debug, Error)]
enum DaemonError {
    #[error("invalid Astronomical configuration: {0}")]
    Configuration(#[from] AstronomicalConfigError),

    #[error("failed to bind supervisor listener: {0}")]
    BindSupervisor(#[source] io::Error),

    #[error("failed to resolve current daemon executable path: {0}")]
    ResolveCurrentExecutable(#[source] io::Error),

    #[error("current daemon executable path does not have a parent directory")]
    MissingExecutableDirectory,

    #[error("failed to read bound supervisor listener address: {0}")]
    ReadBoundSupervisorAddress(#[source] io::Error),

    #[error("failed to serve supervisor HTTP listener: {0}")]
    ServeHttp(#[source] io::Error),

    #[error("failed to shut down inference worker: {0}")]
    ShutdownWorker(#[source] WorkerControlError),
    #[error("failed to create the Astronomical log directory: {0}")]
    CreateLogDirectory(#[source] io::Error),
    #[error("failed to create the Astronomical log appender: {0}")]
    CreateLogAppender(#[source] tracing_appender::rolling::InitError),
    #[error("failed to create the performance log: {0}")]
    CreatePerformanceLog(#[source] io::Error),
    #[error("failed to resolve runtime configuration: {0}")]
    ResolveRuntimeConfig(#[source] ResolvedRuntimeConfigError),
    #[error("resolved supervisor bind address became invalid: {0}")]
    ParseResolvedBindAddress(#[source] std::net::AddrParseError),
    #[error("failed to initialize Astronomical tracing: {reason}")]
    InitializeTracing { reason: String },
}
