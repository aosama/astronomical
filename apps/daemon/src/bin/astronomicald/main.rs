#![forbid(unsafe_code)]

use std::{
    env, io,
    net::SocketAddr,
    path::PathBuf,
    process::ExitCode,
    sync::{Arc, RwLock, atomic::AtomicU64},
    time::Duration,
};

use astronomical_config::{
    AstronomicalConfig, AstronomicalConfigError, AstronomicalInstancePaths, LoggingConfig,
};
use astronomical_daemon::{
    ApplicationBuildIdentity, AstronomicalInstanceLock, CompletionAttributionLog,
    DaemonIpcGenerationContext, DaemonPerformanceAttributionLog, DaemonPerformanceMeasurement,
    DaemonPerformanceOperation, DownloadCatalog, DownloadCatalogError, GenerationPerformanceLog,
    ImageGenerationTimeouts, LibraryDownloadCoordinator, LibraryDownloadCoordinatorError,
    LibraryModelDiscoveryRefresh, ReqwestHubTransportBuildError, ResolvedRuntimeConfig,
    ResolvedRuntimeConfigError, ResolvedRuntimeConfigResolver, ShutdownController,
    StartupFailureStage, WorkerControlError, WorkerHandle,
    build_application_with_full_control_and_library_download, read_instance_lock_holder,
    record_startup_failure, start_daemon_ipc_service,
};
use astronomical_ipc_protocol::DaemonTransportError;
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
                // The menu spawns this process with stderr discarded, so the
                // lock conflict must survive on disk for the UI to explain it.
                record_startup_failure(
                    instance_paths.state_directory(),
                    StartupFailureStage::InstanceLock,
                    &lock_error,
                    read_instance_lock_holder(instance_paths.state_directory()),
                );
                eprintln!("astronomicald startup failed: {lock_error}");
                return ExitCode::FAILURE;
            }
        };
    let user_config = match AstronomicalConfig::load_from_instance_paths(instance_paths.clone()) {
        Ok(user_config) => user_config,
        Err(configuration_error) => {
            record_startup_failure(
                instance_paths.state_directory(),
                StartupFailureStage::Configuration,
                &configuration_error,
                None,
            );
            eprintln!("astronomicald configuration failed: {configuration_error}");
            return ExitCode::FAILURE;
        }
    };
    let _logging_guard = match initialize_tracing(&user_config) {
        Ok(logging_guard) => logging_guard,
        Err(initialization_error) => {
            record_startup_failure(
                instance_paths.state_directory(),
                StartupFailureStage::LoggingInitialization,
                &initialization_error,
                None,
            );
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
        .rotation(tracing_appender::rolling::Rotation::HOURLY)
        .filename_prefix("daemon")
        .filename_suffix("log")
        .max_log_files(logging_config.retained_files())
        .build(logging_config.directory())
        .map_err(DaemonError::CreateLogAppender)?;
    // The local API must remain responsive when its diagnostic disk is slow.
    // Keep a small lossy queue instead of tracing-appender's much larger default.
    let (file_writer, guard) = tracing_appender::non_blocking::NonBlockingBuilder::default()
        .buffered_lines_limit(logging_config.buffered_line_limit())
        .lossy(true)
        .thread_name("astronomical-daemon-log-writer")
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

/// Logging policy and the daemon performance-attribution log every phase writes through.
struct DaemonLogContext {
    logging_config: LoggingConfig,
    daemon_attribution_log: DaemonPerformanceAttributionLog,
}

/// Resolver, resolved runtime snapshot, and the bound REST listener produced at startup.
struct BoundDaemonRuntime {
    runtime_config_resolver: ResolvedRuntimeConfigResolver,
    resolved_runtime_config: ResolvedRuntimeConfig,
    listener: TcpListener,
    bound_daemon_address: SocketAddr,
}

/// Owned singletons that the REST and IPC surfaces must share, wired once at startup.
struct DaemonServices {
    reloadable_config: Arc<RwLock<ResolvedRuntimeConfig>>,
    next_chat_request_id: Arc<AtomicU64>,
    shutdown_controller: ShutdownController,
    internal_shutdown_receiver: tokio::sync::watch::Receiver<bool>,
    download_catalog: Arc<DownloadCatalog>,
    library_download_coordinator: Arc<LibraryDownloadCoordinator>,
}

/// Inputs for building the shared daemon services; owned fields are consumed.
struct DaemonServiceAssembly<'a> {
    instance_paths: &'a AstronomicalInstancePaths,
    user_config: &'a AstronomicalConfig,
    worker_handle: &'a WorkerHandle,
    runtime_config_resolver: &'a ResolvedRuntimeConfigResolver,
    resolved_runtime_config: ResolvedRuntimeConfig,
    download_catalog: DownloadCatalog,
    daemon_attribution_log: &'a DaemonPerformanceAttributionLog,
}

async fn run_daemon(
    instance_paths: AstronomicalInstancePaths,
    user_config: AstronomicalConfig,
) -> Result<(), DaemonError> {
    let DaemonLogContext {
        logging_config,
        daemon_attribution_log,
    } = open_daemon_log_context(&user_config)?;
    let download_catalog = load_bundled_download_catalog(&daemon_attribution_log)?;
    let BoundDaemonRuntime {
        runtime_config_resolver,
        resolved_runtime_config,
        listener,
        bound_daemon_address,
    } = bind_daemon_listener(&instance_paths, &user_config).await?;
    let worker_handle =
        launch_daemon_worker(&resolved_runtime_config, &logging_config, &user_config).await?;
    let DaemonServices {
        reloadable_config,
        next_chat_request_id,
        shutdown_controller,
        internal_shutdown_receiver,
        download_catalog,
        library_download_coordinator,
    } = assemble_daemon_services(DaemonServiceAssembly {
        instance_paths: &instance_paths,
        user_config: &user_config,
        worker_handle: &worker_handle,
        runtime_config_resolver: &runtime_config_resolver,
        resolved_runtime_config,
        download_catalog,
        daemon_attribution_log: &daemon_attribution_log,
    })
    .await?;

    // The daemon IPC service must start before the builder consumes the
    // worker handle: the CLI socket shares the worker and the request-id
    // counter with the REST surface, so both must exist first.
    let daemon_ipc_service = start_daemon_ipc_service(
        &instance_paths,
        DaemonIpcGenerationContext {
            executor: Arc::new(worker_handle.clone()),
            reloadable_config: Some(Arc::clone(&reloadable_config)),
            next_chat_request_id: Arc::clone(&next_chat_request_id),
            download_catalog: Arc::clone(&download_catalog),
            library_download_coordinator: Some(Arc::clone(&library_download_coordinator)),
            instance_paths: instance_paths.clone(),
        },
        &daemon_attribution_log,
    )
    .await
    .map_err(DaemonError::StartDaemonIpc)?;
    let application = build_application_with_full_control_and_library_download(
        worker_handle.clone(),
        next_chat_request_id,
        Arc::clone(&reloadable_config),
        runtime_config_resolver,
        shutdown_controller,
        download_catalog,
        library_download_coordinator,
    );

    println!("astronomicald listening on http://{bound_daemon_address}");

    let serve_result = axum::serve(listener, application)
        .with_graceful_shutdown(shutdown_signal(internal_shutdown_receiver))
        .await
        .map_err(DaemonError::ServeHttp);
    let worker_shutdown_result = worker_handle
        .shutdown()
        .await
        .map(|_| ())
        .map_err(DaemonError::ShutdownWorker);
    let daemon_ipc_shutdown_result = daemon_ipc_service.shutdown().await;

    serve_result?;
    worker_shutdown_result?;
    daemon_ipc_shutdown_result.map_err(DaemonError::ShutdownDaemonIpc)?;

    Ok(())
}

fn open_daemon_log_context(
    user_config: &AstronomicalConfig,
) -> Result<DaemonLogContext, DaemonError> {
    let logging_config = user_config.logging()?;
    let daemon_attribution_log = DaemonPerformanceAttributionLog::open(
        logging_config.directory(),
        user_config.performance_attribution_enabled(),
    )
    .map_err(DaemonError::CreateDaemonPerformanceAttributionLog)?;
    Ok(DaemonLogContext {
        logging_config,
        daemon_attribution_log,
    })
}

fn load_bundled_download_catalog(
    daemon_attribution_log: &DaemonPerformanceAttributionLog,
) -> Result<DownloadCatalog, DaemonError> {
    daemon_attribution_log
        .measure_operation(
            DaemonPerformanceOperation::LibraryCatalogLoad,
            DownloadCatalog::load_bundled,
            |catalog_outcome| match catalog_outcome {
                Ok(download_catalog) => DaemonPerformanceMeasurement::successful_catalog_load(
                    download_catalog.entry_count(),
                ),
                Err(_) => DaemonPerformanceMeasurement::failure(),
            },
        )
        .map_err(DaemonError::RecordDaemonPerformanceAttribution)?
        .map_err(DaemonError::LoadDownloadCatalog)
}

async fn bind_daemon_listener(
    instance_paths: &AstronomicalInstancePaths,
    user_config: &AstronomicalConfig,
) -> Result<BoundDaemonRuntime, DaemonError> {
    let runtime_config_resolver = ResolvedRuntimeConfigResolver::for_instance(
        instance_paths.clone(),
        fallback_worker_executable_path()?,
    );
    let resolved_runtime_config = runtime_config_resolver
        .resolve(user_config)
        .map_err(DaemonError::ResolveRuntimeConfig)?;
    let daemon_bind_address = resolved_runtime_config
        .bind_address
        .parse::<SocketAddr>()
        .map_err(DaemonError::ParseResolvedBindAddress)?;
    let listener = TcpListener::bind(daemon_bind_address)
        .await
        .map_err(DaemonError::BindDaemon)?;
    let bound_daemon_address = listener
        .local_addr()
        .map_err(DaemonError::ReadBoundDaemonAddress)?;
    tracing::info!(
        bind_address = %bound_daemon_address,
        channel = instance_paths
            .runtime_instance()
            .map_or("explicit", astronomical_config::AstronomicalRuntimeInstance::as_str),
        version = ApplicationBuildIdentity::current().version,
        commit = ApplicationBuildIdentity::current().commit,
        configuration_generation = %resolved_runtime_config.configuration_generation,
        "Astronomical REST API listener bound"
    );
    Ok(BoundDaemonRuntime {
        runtime_config_resolver,
        resolved_runtime_config,
        listener,
        bound_daemon_address,
    })
}

async fn launch_daemon_worker(
    resolved_runtime_config: &ResolvedRuntimeConfig,
    logging_config: &LoggingConfig,
    user_config: &AstronomicalConfig,
) -> Result<WorkerHandle, DaemonError> {
    let performance_log = GenerationPerformanceLog::open(logging_config.directory())
        .map_err(DaemonError::CreatePerformanceLog)?;
    let completion_log = CompletionAttributionLog::open(
        logging_config.directory(),
        user_config.completion_attribution_enabled(),
    )
    .map_err(DaemonError::CreateCompletionAttributionLog)?;
    match WorkerHandle::launch_with_startup_configuration_and_image_generation_timeouts(
        &resolved_runtime_config.worker_executable_path,
        WORKER_MODEL_LOAD_TIMEOUT,
        ImageGenerationTimeouts::DEFAULT,
        performance_log,
        completion_log,
        Arc::clone(&resolved_runtime_config.model_policy_catalog),
        resolved_runtime_config.worker_startup_configuration(),
    )
    .await
    {
        Ok(worker_handle) => Ok(worker_handle),
        Err(worker_start_error) => {
            eprintln!("astronomicald worker unavailable: {worker_start_error}");
            Ok(WorkerHandle::unavailable())
        }
    }
}

async fn assemble_daemon_services(
    assembly: DaemonServiceAssembly<'_>,
) -> Result<DaemonServices, DaemonError> {
    let DaemonServiceAssembly {
        instance_paths,
        user_config,
        worker_handle,
        runtime_config_resolver,
        resolved_runtime_config,
        download_catalog,
        daemon_attribution_log,
    } = assembly;
    let reloadable_config = Arc::new(RwLock::new(resolved_runtime_config));
    let next_chat_request_id = Arc::new(AtomicU64::new(1));
    let shutdown_controller = ShutdownController::new();
    let internal_shutdown_receiver = shutdown_controller.subscribe();
    let download_catalog = Arc::new(download_catalog);
    let hub_transport = Arc::new(
        astronomical_daemon::ReqwestHubTransport::production()
            .map_err(DaemonError::BuildHubTransport)?,
    );
    let discovery_refresh = Arc::new(LibraryModelDiscoveryRefresh::new(
        runtime_config_resolver.clone(),
        Arc::clone(&reloadable_config),
        worker_handle.clone(),
    ));
    // The download coordinator is built once and is not rebuilt on config
    // reload, so the override only takes effect on the next application start.
    let models_download_directory = user_config
        .model_download_directory()
        .cloned()
        .unwrap_or_else(|| instance_paths.models_directory());
    let library_download_coordinator =
        Arc::new(astronomical_daemon::LibraryDownloadCoordinator::new(
            Arc::clone(&download_catalog),
            models_download_directory,
            Arc::new(astronomical_daemon::Fs4DiskCapacityQuery),
            hub_transport.clone(),
            hub_transport,
            discovery_refresh,
            daemon_attribution_log.clone(),
        ));
    library_download_coordinator
        .recover_startup_state()
        .await
        .map_err(DaemonError::RecoverLibraryDownload)?;
    Ok(DaemonServices {
        reloadable_config,
        next_chat_request_id,
        shutdown_controller,
        internal_shutdown_receiver,
        download_catalog,
        library_download_coordinator,
    })
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

    #[error("failed to bind daemon listener: {0}")]
    BindDaemon(#[source] io::Error),

    #[error("failed to resolve current daemon executable path: {0}")]
    ResolveCurrentExecutable(#[source] io::Error),

    #[error("current daemon executable path does not have a parent directory")]
    MissingExecutableDirectory,

    #[error("failed to read bound daemon listener address: {0}")]
    ReadBoundDaemonAddress(#[source] io::Error),

    #[error("failed to serve daemon HTTP listener: {0}")]
    ServeHttp(#[source] io::Error),

    #[error("failed to shut down inference worker: {0}")]
    ShutdownWorker(#[source] WorkerControlError),
    #[error("failed to start the daemon IPC service: {0}")]
    StartDaemonIpc(#[source] DaemonTransportError),
    #[error("failed to shut down the daemon IPC service: {0}")]
    ShutdownDaemonIpc(#[source] DaemonTransportError),
    #[error("failed to create the Astronomical log directory: {0}")]
    CreateLogDirectory(#[source] io::Error),
    #[error("failed to create the Astronomical log appender: {0}")]
    CreateLogAppender(#[source] tracing_appender::rolling::InitError),
    #[error("failed to create the performance log: {0}")]
    CreatePerformanceLog(#[source] io::Error),
    #[error("failed to create the completion attribution log: {0}")]
    CreateCompletionAttributionLog(#[source] io::Error),
    #[error("failed to create the daemon performance-attribution log: {0}")]
    CreateDaemonPerformanceAttributionLog(#[source] io::Error),
    #[error("failed to record daemon performance attribution: {0}")]
    RecordDaemonPerformanceAttribution(#[source] io::Error),
    #[error("invalid bundled download catalog: {0}")]
    LoadDownloadCatalog(#[source] DownloadCatalogError),
    #[error("failed to construct the Hugging Face transfer client: {0}")]
    BuildHubTransport(#[source] ReqwestHubTransportBuildError),
    #[error("failed to recover the durable Library download: {0}")]
    RecoverLibraryDownload(#[source] LibraryDownloadCoordinatorError),
    #[error("failed to resolve runtime configuration: {0}")]
    ResolveRuntimeConfig(#[source] ResolvedRuntimeConfigError),
    #[error("resolved daemon bind address became invalid: {0}")]
    ParseResolvedBindAddress(#[source] std::net::AddrParseError),
    #[error("failed to initialize Astronomical tracing: {reason}")]
    InitializeTracing { reason: String },
}
