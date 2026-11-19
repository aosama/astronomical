//! The `DownloadStatus` answer a stub daemon scripts for its Nth poll.

use ipc_protocol::DaemonResponse;

use super::stub_daemon::StubDaemonConfig;

/// The scripted job and download list for poll `poll_index`. Each script's
/// last entry repeats once it runs out; an empty job script means no job and
/// an empty list script means nothing waits.
pub fn scripted_download_status(config: &StubDaemonConfig, poll_index: usize) -> DaemonResponse {
    let job = config
        .download_jobs
        .get(poll_index)
        .or(config.download_jobs.last())
        .cloned()
        .unwrap_or(None);
    let waiting_huggingface_ids = config
        .download_waiting_huggingface_ids
        .get(poll_index)
        .or(config.download_waiting_huggingface_ids.last())
        .cloned()
        .unwrap_or_default();
    DaemonResponse::DownloadStatus {
        job,
        waiting_huggingface_ids,
    }
}
