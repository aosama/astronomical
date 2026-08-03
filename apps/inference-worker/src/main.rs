#![forbid(unsafe_code)]

use std::process::ExitCode;

use astronomical_inference_worker::{worker_process_runtime, worker_startup};

fn main() -> ExitCode {
    let worker_outcome =
        match worker_process_runtime::run_worker_future_with_bounded_runtime_shutdown(|| {
            worker_startup::run_bootstrapped_worker(tokio::io::stdin(), tokio::io::stdout())
        }) {
            Ok(worker_outcome) => worker_outcome,
            Err(runtime_initialization_error) => {
                eprintln!("failed to initialize worker runtime: {runtime_initialization_error}");
                return ExitCode::FAILURE;
            }
        };

    match worker_outcome {
        Ok(()) => ExitCode::SUCCESS,
        Err(worker_error) => {
            eprintln!("model worker stopped with an error: {worker_error}");
            ExitCode::FAILURE
        }
    }
}
