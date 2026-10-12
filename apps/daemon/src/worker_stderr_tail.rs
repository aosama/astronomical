use std::{collections::VecDeque, sync::Arc};

use tokio::sync::Mutex;

/// Maximum worker standard-error context retained for one fatal supervisor diagnostic.
const MAXIMUM_WORKER_STDERR_TAIL_BYTES: usize = 8 * 1_024;

/// Shared bounded tail of diagnostics emitted before a worker process exits.
///
/// The child drain task appends fixed-size reads while the worker-process owner
/// snapshots the same tail after the IPC event stream closes. Keeping only the
/// tail prevents a noisy or compromised child from growing supervisor memory.
#[derive(Clone, Default)]
pub(crate) struct WorkerStderrTail {
    retained_bytes: Arc<Mutex<VecDeque<u8>>>,
}

impl WorkerStderrTail {
    /// Appends one fixed-size read while retaining only the newest bounded bytes.
    pub(crate) async fn append(&self, diagnostic_bytes: &[u8]) {
        let mut retained_bytes = self.retained_bytes.lock().await;
        let bytes_to_skip = diagnostic_bytes
            .len()
            .saturating_sub(MAXIMUM_WORKER_STDERR_TAIL_BYTES);
        retained_bytes.extend(&diagnostic_bytes[bytes_to_skip..]);
        while retained_bytes.len() > MAXIMUM_WORKER_STDERR_TAIL_BYTES {
            retained_bytes.pop_front();
        }
    }

    /// Returns a printable snapshot suitable for the supervisor's local log.
    pub(crate) async fn diagnostic_snapshot(&self) -> String {
        let retained_bytes = self.retained_bytes.lock().await;
        let contiguous_bytes = retained_bytes.iter().copied().collect::<Vec<_>>();
        let diagnostic_text = String::from_utf8_lossy(&contiguous_bytes).trim().to_owned();
        if diagnostic_text.is_empty() {
            "no worker stderr was captured".to_owned()
        } else {
            diagnostic_text
        }
    }
}
