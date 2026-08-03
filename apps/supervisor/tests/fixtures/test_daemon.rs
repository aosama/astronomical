#![forbid(unsafe_code)]

use std::{
    collections::HashMap, env, error::Error, net::SocketAddr, path::PathBuf, sync::Arc,
    time::Duration,
};

use astronomical_supervisor::{GenerationPerformanceLog, WorkerHandle, build_application};
use tokio::net::TcpListener;

const SUPERVISOR_BIND_ADDRESS_ENVIRONMENT_VARIABLE: &str = "ASTRONOMICAL_SUPERVISOR_BIND_ADDRESS";
const TEST_WORKER_EXECUTABLE_PATH_ENVIRONMENT_VARIABLE: &str =
    "ASTRONOMICAL_TEST_WORKER_EXECUTABLE_PATH";
const DEFAULT_MAX_OUTPUT_TOKENS: u32 = 20480;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let supervisor_bind_address =
        env::var(SUPERVISOR_BIND_ADDRESS_ENVIRONMENT_VARIABLE)?.parse::<SocketAddr>()?;
    if !supervisor_bind_address.ip().is_loopback() {
        return Err("the test daemon requires a loopback bind address".into());
    }
    let worker_executable_path =
        PathBuf::from(env::var(TEST_WORKER_EXECUTABLE_PATH_ENVIRONMENT_VARIABLE)?);
    let listener = TcpListener::bind(supervisor_bind_address).await?;
    let bound_supervisor_address = listener.local_addr()?;
    let performance_log_directory = std::env::temp_dir().join("astronomical-test-daemon");
    std::fs::create_dir_all(&performance_log_directory)?;
    let performance_log = GenerationPerformanceLog::open(&performance_log_directory)?;
    let worker_handle = WorkerHandle::launch(
        worker_executable_path,
        Duration::from_secs(60),
        performance_log,
        Arc::new(HashMap::new()),
        DEFAULT_MAX_OUTPUT_TOKENS,
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
