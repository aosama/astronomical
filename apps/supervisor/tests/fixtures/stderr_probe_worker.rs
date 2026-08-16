#![forbid(unsafe_code)]

use std::{error::Error, os::unix::fs::MetadataExt, process::ExitCode};

use astronomical_ipc_protocol::{ProtocolWriter, WorkerEvent};

#[tokio::main]
async fn main() -> ExitCode {
    match run_stderr_probe_worker().await {
        Ok(()) => ExitCode::SUCCESS,
        Err(worker_error) => {
            eprintln!("stderr-probe worker failed: {worker_error}");
            ExitCode::FAILURE
        }
    }
}

async fn run_stderr_probe_worker() -> Result<(), Box<dyn Error + Send + Sync>> {
    if stderr_points_to_dev_null()? {
        return Ok(());
    }
    let oversized_diagnostic_prefix = "x".repeat(16 * 1_024);
    eprintln!("{oversized_diagnostic_prefix}");
    eprintln!("stderr-probe worker observed visible stderr");
    ProtocolWriter::new(tokio::io::stdout())
        .send_event(&WorkerEvent::Idle {
            machine_mlx_memory_ceiling_bytes: 40_000_000_000,
            effective_mlx_memory_ceiling_bytes: 40_000_000_000,
            minimum_mlx_memory_ceiling_bytes: 1,
        })
        .await?;
    Ok(())
}

fn stderr_points_to_dev_null() -> Result<bool, Box<dyn Error + Send + Sync>> {
    let stderr_metadata = std::fs::metadata("/dev/fd/2")?;
    let dev_null_metadata = std::fs::metadata("/dev/null")?;
    Ok(stderr_metadata.dev() == dev_null_metadata.dev()
        && stderr_metadata.ino() == dev_null_metadata.ino())
}
