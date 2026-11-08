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
    RELATIVE_PATH, REPOSITORY_ID, REVISION, ScriptedPayloadTransport, disabled_attribution,
    multi_file_paused_job, payload_response, sha256_hex, staged_file_path_for_relative_path,
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

/// The job 0.2.91 left behind on charlesberret's Mac (#1896): every byte of every non-empty file
/// is staged, the empty file was never created, and verification marked the job
/// `failed: checksum_mismatch`. After the upgrade, running the same command again must create the
/// empty file, verify and publish, without fetching anything.
fn job_failed_by_the_missing_empty_file() -> astronomical_daemon::DownloadJob {
    let romeo_bytes = ROMEO_AND_JULIET.len();
    let job_document = format!(
        "{{\"huggingface_id\":\"{REPOSITORY_ID}\",\"revision\":\"{REVISION}\",\"state\":\"failed\",\
\"bytes_completed\":{romeo_bytes},\"bytes_total\":{romeo_bytes},\"current_file_relative_path\":null,\
\"files\":[{{\"relative_path\":\"{EMPTY_FILE_PATH}\",\"expected_bytes\":0,\
\"expected_digest\":{{\"algorithm\":\"sha256\",\"hex\":\"{empty_digest}\"}},\"bytes_on_disk\":0}},\
{{\"relative_path\":\"{RELATIVE_PATH}\",\"expected_bytes\":{romeo_bytes},\
\"expected_digest\":{{\"algorithm\":\"sha256\",\"hex\":\"{romeo_digest}\"}},\"bytes_on_disk\":{romeo_bytes}}}],\
\"error_code\":\"checksum_mismatch\",\"updated_at\":100}}",
        empty_digest = sha256_hex(b""),
        romeo_digest = sha256_hex(ROMEO_AND_JULIET),
    );
    astronomical_daemon::DownloadJob::parse_json(&job_document)
        .expect("the job left by the previous version should still parse")
}

#[tokio::test]
async fn should_publish_a_job_the_previous_version_failed_for_a_missing_empty_file() {
    tokio::time::timeout(Duration::from_secs(5), async {
        let test_directory = TempDir::new().expect("temporary directory should be available");
        let job_store = DownloadJobStore::new(test_directory.path().join("models"));
        job_store
            .create(&job_failed_by_the_missing_empty_file())
            .expect("the leftover failed job should persist");
        let staged_romeo =
            staged_file_path_for_relative_path(job_store.models_directory(), RELATIVE_PATH);
        fs::create_dir_all(
            staged_romeo
                .parent()
                .expect("staged file should have a parent"),
        )
        .expect("staging directory should exist");
        fs::write(&staged_romeo, ROMEO_AND_JULIET)
            .expect("the complete staged file should be written");
        let no_network = Arc::new(ScriptedPayloadTransport::new([]));
        let transfer = transfer_over(
            &test_directory,
            &job_store,
            no_network.clone(),
            DownloadTransferControl::new(),
        );

        let transfer_outcome = transfer
            .resume(200)
            .await
            .expect("the upgraded daemon should finish the leftover job");

        assert!(matches!(
            transfer_outcome,
            DownloadPayloadTransferOutcome::ReadyToPublish(_)
        ));
        assert!(
            no_network.requests().is_empty(),
            "every byte was already on disk"
        );
        let empty_file =
            staged_file_path_for_relative_path(job_store.models_directory(), EMPTY_FILE_PATH);
        assert_eq!(
            fs::metadata(empty_file)
                .expect("the empty file should now exist")
                .len(),
            0
        );
    })
    .await
    .expect("the upgrade journey should remain bounded");
}
