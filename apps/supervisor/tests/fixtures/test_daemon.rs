#![forbid(unsafe_code)]

use std::{collections::HashMap, env, error::Error, path::PathBuf, sync::Arc, time::Duration};

use astronomical_config::{AstronomicalInstancePaths, AstronomicalRuntimeInstance};
use astronomical_supervisor::{GenerationPerformanceLog, WorkerHandle, build_application};
use tokio::net::TcpListener;

const TEST_WORKER_EXECUTABLE_PATH_ENVIRONMENT_VARIABLE: &str =
    "ASTRONOMICAL_TEST_WORKER_EXECUTABLE_PATH";
#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let state_directory = state_directory_argument()?;
    let instance_paths = AstronomicalInstancePaths::for_state_directory(
        state_directory,
        AstronomicalRuntimeInstance::Development,
    );
    let supervisor_bind_address = instance_paths.default_bind_address();
    let worker_executable_path =
        PathBuf::from(env::var(TEST_WORKER_EXECUTABLE_PATH_ENVIRONMENT_VARIABLE)?);
    let listener = TcpListener::bind(supervisor_bind_address).await?;
    let bound_supervisor_address = listener.local_addr()?;
    let performance_log_directory = instance_paths.logging_directory();
    std::fs::create_dir_all(&performance_log_directory)?;
    let performance_log = GenerationPerformanceLog::open(&performance_log_directory)?;
    let worker_handle = WorkerHandle::launch(
        worker_executable_path,
        Duration::from_secs(60),
        performance_log,
        Arc::new(HashMap::new()),
    )
    .await?;
    let application = build_application(worker_handle.clone());

    println!("astronomicald listening on http://{bound_supervisor_address}");

    let serve_result = axum::serve(listener, application)
        .with_graceful_shutdown(shutdown_signal())
        .await;
    let worker_shutdown_result = worker_handle.shutdown().await;
    serve_result?;
    worker_shutdown_result?;

    Ok(())
}

fn state_directory_argument() -> Result<PathBuf, Box<dyn Error>> {
    let process_arguments = env::args_os().collect::<Vec<_>>();
    let state_directory_position = process_arguments
        .iter()
        .position(|argument| argument == "--state-directory")
        .ok_or("the test daemon requires --state-directory")?;
    process_arguments
        .get(state_directory_position + 1)
        .map(|state_directory| PathBuf::from(state_directory.as_os_str()))
        .ok_or_else(|| "the test daemon requires a state-directory value".into())
}

async fn shutdown_signal() {
    let interrupt_signal = async {
        if tokio::signal::ctrl_c().await.is_err() {
            std::future::pending::<()>().await;
        }
    };
    let terminate_signal = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut signal_stream) => {
                signal_stream.recv().await;
            }
            Err(_) => std::future::pending::<()>().await,
        }
    };

    tokio::select! {
        () = interrupt_signal => {},
        () = terminate_signal => {},
    }
}
