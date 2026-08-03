use std::future::Future;
use std::io;
use std::time::Duration;

const WORKER_RUNTIME_SHUTDOWN_TIMEOUT: Duration = Duration::from_millis(100);

/// Runs the worker future and bounds Tokio shutdown after that future finishes.
///
/// Tokio implements standard input with a blocking read that cannot be cancelled.
/// A plain `#[tokio::main]` shutdown can therefore keep a failed worker process
/// alive until its parent closes standard input. The timeout lets the process
/// exit promptly so the supervisor observes EOF and releases the failed worker.
pub fn run_worker_future_with_bounded_runtime_shutdown<Output, WorkerFuture, WorkerFutureFactory>(
    worker_future_factory: WorkerFutureFactory,
) -> io::Result<Output>
where
    WorkerFuture: Future<Output = Output>,
    WorkerFutureFactory: FnOnce() -> WorkerFuture,
{
    let worker_runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    let worker_output = worker_runtime.block_on(worker_future_factory());
    worker_runtime.shutdown_timeout(WORKER_RUNTIME_SHUTDOWN_TIMEOUT);
    Ok(worker_output)
}
