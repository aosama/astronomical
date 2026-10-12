//! Observation of the worker subprocesses owned by a real-model test harness.
//!
//! The memory-demotion contract crosses a process boundary: the trusted worker
//! must exit and be reaped before any replacement process exists. Proving that
//! from the supervisor's own control flow would only restate the implementation,
//! so these helpers observe the boundary from outside it — by sampling which
//! worker subprocesses the harness itself owns across the transition.

use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};

use tokio::task::JoinHandle;

/// How often the worker-process timeline samples the harness's own children.
const WORKER_PROCESS_SAMPLE_INTERVAL: Duration = Duration::from_millis(20);

/// Identifiers of the worker subprocesses owned by this test harness.
///
/// The scan is deliberately restricted to children of this process. Matching on
/// the executable name alone would also match an independently running Stable
/// worker, which no test may ever inspect or influence.
pub(crate) fn worker_child_process_identifiers() -> Vec<u32> {
    let own_identifier = std::process::id().to_string();
    let Ok(child_listing) = std::process::Command::new("pgrep")
        .args(["-P", own_identifier.as_str()])
        .output()
    else {
        return Vec::new();
    };
    if !child_listing.status.success() {
        return Vec::new();
    }
    String::from_utf8_lossy(&child_listing.stdout)
        .lines()
        .filter_map(|line| line.trim().parse::<u32>().ok())
        .filter(|child_identifier| {
            let Ok(named_listing) = std::process::Command::new("ps")
                .args(["-p", &child_identifier.to_string(), "-o", "comm="])
                .output()
            else {
                return false;
            };
            String::from_utf8_lossy(&named_listing.stdout).contains("astronomical-inference-worker")
        })
        .collect()
}

/// Requires the harness to own exactly one worker process and returns its
/// identifier, so a journey can pin the worker identity before and after a
/// supervisor-driven replacement.
pub(crate) fn sole_worker_child_process_identifier(context: &str) -> u32 {
    let mut worker_identifiers = worker_child_process_identifiers();
    worker_identifiers.sort_unstable();
    assert_eq!(
        worker_identifiers.len(),
        1,
        "{context}, observed {worker_identifiers:?}"
    );
    worker_identifiers[0]
}

/// Samples which worker subprocesses exist across a transition, so a journey
/// can tell a stop-first restart (a gap with no worker) from a candidate-first
/// overlap (two workers at once).
pub(crate) struct WorkerProcessTimeline {
    stop_requested: Arc<AtomicBool>,
    sampler_task: JoinHandle<Vec<Vec<u32>>>,
}

impl WorkerProcessTimeline {
    pub(crate) fn start() -> Self {
        let stop_requested = Arc::new(AtomicBool::new(false));
        let sampler_stop_requested = Arc::clone(&stop_requested);
        let sampler_task = tokio::spawn(async move {
            let mut observed_transitions: Vec<Vec<u32>> = Vec::new();
            let mut previous_identifiers: Option<Vec<u32>> = None;
            while !sampler_stop_requested.load(Ordering::Relaxed) {
                let identifiers = tokio::task::spawn_blocking(worker_child_process_identifiers)
                    .await
                    .unwrap_or_default();
                let identifiers = {
                    let mut identifiers = identifiers;
                    identifiers.sort_unstable();
                    identifiers
                };
                if previous_identifiers.as_ref() != Some(&identifiers) {
                    observed_transitions.push(identifiers.clone());
                    previous_identifiers = Some(identifiers);
                }
                tokio::time::sleep(WORKER_PROCESS_SAMPLE_INTERVAL).await;
            }
            observed_transitions
        });
        Self {
            stop_requested,
            sampler_task,
        }
    }

    pub(crate) async fn stop(self) -> Vec<Vec<u32>> {
        self.stop_requested.store(true, Ordering::Relaxed);
        self.sampler_task
            .await
            .expect("the worker-process sampler should not panic")
    }
}

/// Reports whether the named process is still running without signalling it.
pub(crate) fn is_process_running(process_identifier: u32) -> bool {
    std::process::Command::new("kill")
        .args(["-0", &process_identifier.to_string()])
        .output()
        .is_ok_and(|probe| probe.status.success())
}
