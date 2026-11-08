//! Acceptance journeys for repositories that contain zero-byte files (issue #1896): the transfer
//! creates them itself, never asks the network for them, and survives a pause or a checksum retry.

use std::{fs, sync::Arc, time::Duration};

use astronomical_daemon::{
    DownloadJobState, DownloadJobStore, DownloadPayloadTransfer, DownloadPayloadTransferError,
    DownloadPayloadTransferOutcome, DownloadTransferControl,
};
use tempfile::TempDir;

use super::download_payload_support as support;

use support::{
    RELATIVE_PATH, ScriptedPayloadTransport, disabled_attribution, multi_file_paused_job,
    payload_response, staged_file_path_for_relative_path,
};

const EMPTY_FILE_PATH: &str = "tools/__init__.py";
const ROMEO_AND_JULIET: &[u8] = b"Romeo and Juliet";

fn transfer_over(
    test_directory: &TempDir,
    job_store: &DownloadJobStore,
    transport: Arc<ScriptedPayloadTransport>,
    transfer_control: DownloadTransferControl,
) -> DownloadPayloadTransfer {
    DownloadPayloadTransfer::new(
        job_store.clone(),
        transport,
        disabled_attribution(test_directory),
        transfer_control,
    )
}

fn store_with_an_empty_file_and_weights(test_directory: &TempDir) -> DownloadJobStore {
    let job_store = DownloadJobStore::new(test_directory.path().join("models"));
    job_store
        .create(&multi_file_paused_job(&[
            (EMPTY_FILE_PATH, b""),
            (RELATIVE_PATH, ROMEO_AND_JULIET),
        ]))
        .expect("exact manifest job should persist");
    job_store
}

#[tokio::test]
async fn should_create_the_empty_file_and_request_only_files_with_bytes() {
    tokio::time::timeout(Duration::from_secs(5), async {
        let test_directory = TempDir::new().expect("temporary directory should be available");
        let job_store = store_with_an_empty_file_and_weights(&test_directory);
        let transport = Arc::new(ScriptedPayloadTransport::new([payload_response(
            200,
            None,
            [ROMEO_AND_JULIET],
        )]));
        let transfer = transfer_over(
            &test_directory,
            &job_store,
            transport.clone(),
            DownloadTransferControl::new(),
        );

        let transfer_outcome = transfer
            .resume(200)
            .await
            .expect("the download should verify");

        assert!(matches!(
            transfer_outcome,
            DownloadPayloadTransferOutcome::ReadyToPublish(_)
        ));
        assert_eq!(
            transport.requests().len(),
            1,
            "the empty file needs no request"
        );
        let empty_file =
            staged_file_path_for_relative_path(job_store.models_directory(), EMPTY_FILE_PATH);
        assert_eq!(
            fs::metadata(empty_file)
                .expect("the empty file should be staged")
                .len(),
            0
        );
    })
    .await
    .expect("the empty-file journey should remain bounded");
}

#[tokio::test]
async fn should_pause_before_any_request_and_finish_the_empty_file_on_resume() {
    tokio::time::timeout(Duration::from_secs(5), async {
        let test_directory = TempDir::new().expect("temporary directory should be available");
        let job_store = store_with_an_empty_file_and_weights(&test_directory);
        let paused_control = DownloadTransferControl::new();
        paused_control.request_pause();
        let idle_transport = Arc::new(ScriptedPayloadTransport::new([]));
        let pausing_transfer = transfer_over(
            &test_directory,
            &job_store,
            idle_transport.clone(),
            paused_control,
        );

        let pause_outcome = pausing_transfer
            .resume(200)
            .await
            .expect("a pause is durable");

        assert!(matches!(
            pause_outcome,
            DownloadPayloadTransferOutcome::Paused(_)
        ));
        assert!(idle_transport.requests().is_empty());
        let resuming_transfer = transfer_over(
            &test_directory,
            &job_store,
            Arc::new(ScriptedPayloadTransport::new([payload_response(
                200,
                None,
                [ROMEO_AND_JULIET],
            )])),
            DownloadTransferControl::new(),
        );
        let resume_outcome = resuming_transfer
            .resume(300)
            .await
            .expect("the resumed download should verify");
        assert!(matches!(
            resume_outcome,
            DownloadPayloadTransferOutcome::ReadyToPublish(_)
        ));
    })
    .await
    .expect("the pause-and-resume journey should remain bounded");
}

#[tokio::test]
async fn should_keep_the_empty_file_through_a_checksum_retry() {
    tokio::time::timeout(Duration::from_secs(5), async {
        let test_directory = TempDir::new().expect("temporary directory should be available");
        let job_store = store_with_an_empty_file_and_weights(&test_directory);
        let corrupting_transfer = transfer_over(
            &test_directory,
            &job_store,
            Arc::new(ScriptedPayloadTransport::new([payload_response(
                200,
                None,
                [b"Romeo and Juliex".as_slice()],
            )])),
            DownloadTransferControl::new(),
        );

        let corruption_failure = corrupting_transfer
            .resume(200)
            .await
            .expect_err("corrupt bytes must fail verification");

        assert!(matches!(
            corruption_failure,
            DownloadPayloadTransferError::ChecksumMismatch { .. }
        ));
        let retrying_transfer = transfer_over(
            &test_directory,
            &job_store,
            Arc::new(ScriptedPayloadTransport::new([payload_response(
                200,
                None,
                [ROMEO_AND_JULIET],
            )])),
            DownloadTransferControl::new(),
        );
        let retry_outcome = retrying_transfer
            .resume(300)
            .await
            .expect("the retry should verify, empty file included");
        assert!(matches!(
            retry_outcome,
            DownloadPayloadTransferOutcome::ReadyToPublish(ref ready_job)
                if ready_job.state() == DownloadJobState::Publishing
        ));
    })
    .await
    .expect("the checksum-retry journey should remain bounded");
}
