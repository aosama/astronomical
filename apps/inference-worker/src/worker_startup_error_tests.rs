use std::io;

use crate::worker_startup_error::{WorkerProcessError, WorkerStartupError};

#[test]
fn should_include_the_exact_io_cause_in_a_worker_startup_failure() {
    let worker_process_error =
        WorkerProcessError::Startup(WorkerStartupError::SamplePhysicalMemorySize(
            io::Error::other("synthetic sysctl spawn failure"),
        ));

    let worker_failure_diagnostic = worker_process_error.to_string();

    assert!(worker_failure_diagnostic.contains("failed to sample hw.memsize"));
    assert!(worker_failure_diagnostic.contains("synthetic sysctl spawn failure"));
}
