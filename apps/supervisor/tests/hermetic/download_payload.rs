//! Acceptance journey for ranged payload transfer, verification, and atomic publication.

use std::{fs, sync::Arc, time::Duration};

use astronomical_supervisor::{
    DownloadJobPublicErrorCode, DownloadJobState, DownloadJobStore, DownloadPayloadTransfer,
    DownloadPayloadTransferOutcome, DownloadPublication, DownloadTransferControl,
    HubPayloadResponse, HubTransportError,
};
use bytes::Bytes;
use futures_util::{StreamExt, stream};
use tempfile::TempDir;

mod support;

use support::{
    FailingRefresh, RecordingRefresh, ScriptedPayloadTransport, disabled_attribution, git_blob_job,
    git_blob_sha1_hex, parse_job_with_size, payload_response, sha256_hex, sha256_job,
    staged_file_path,
};

const REPOSITORY_ID: &str = "astronomical-test/example-qwen";
const REVISION: &str = "0123456789abcdef0123456789abcdef01234567";
const RELATIVE_PATH: &str = "weights/romeo-and-juliet.txt";
const ROMEO_AND_JULIET: &[u8] = b"Romeo and Juliet";

#[tokio::test]
async fn should_resume_verify_and_atomically_publish_the_complete_user_journey() {
    tokio::time::timeout(Duration::from_secs(5), async {
        let test_directory = TempDir::new().expect("temporary directory should be available");
        let models_directory = test_directory.path().join("models");
        let job_store = DownloadJobStore::new(models_directory.clone());
        job_store
            .create(&sha256_job("paused", 0, sha256_hex(ROMEO_AND_JULIET), None))
            .expect("exact manifest job should persist");
        let staged_file = staged_file_path(&models_directory);
        fs::create_dir_all(
            staged_file
                .parent()
                .expect("staged file should have a parent"),
        )
        .expect("staging directory should exist");
        fs::write(&staged_file, b"Romeo ").expect("resumable prefix should be written");
        let transport = Arc::new(ScriptedPayloadTransport::new([payload_response(
            206,
            Some("bytes 6-15/16"),
            [b"and ".as_slice(), b"Juliet".as_slice()],
        )]));
        let transfer = DownloadPayloadTransfer::new(
            job_store.clone(),
            transport.clone(),
            disabled_attribution(&test_directory),
            DownloadTransferControl::new(),
        );

        let transfer_outcome = transfer
            .resume(200)
            .await
            .expect("resumed payload should verify");

        assert!(matches!(
            transfer_outcome,
            DownloadPayloadTransferOutcome::ReadyToPublish(_)
        ));
        let requests = transport.requests();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].resume_offset_bytes(), 6);
        assert!(
            requests[0]
                .url()
                .ends_with(&format!("/resolve/{REVISION}/{RELATIVE_PATH}"))
        );
        assert_eq!(
            fs::read(&staged_file).expect("staged file should be readable"),
            ROMEO_AND_JULIET
        );

        let refresh = Arc::new(RecordingRefresh::default());
        let published_directory =
            DownloadPublication::new(job_store.clone(), disabled_attribution(&test_directory))
                .publish(refresh.clone())
                .await
                .expect("verified staging should publish atomically");

        assert_eq!(
            fs::read(published_directory.join(RELATIVE_PATH))
                .expect("published payload should be readable"),
            ROMEO_AND_JULIET
        );
        assert_eq!(refresh.refreshed_directories(), [published_directory]);
        assert!(
            job_store
                .load()
                .expect("completed job lookup should succeed")
                .is_none()
        );
        assert!(!staged_file.exists());
    })
    .await
    .expect("payload acceptance journey should remain bounded");
}

#[tokio::test]
async fn should_verify_git_blob_sha1_without_requesting_already_complete_bytes() {
    tokio::time::timeout(Duration::from_secs(5), async {
        let test_directory = TempDir::new().expect("temporary directory should be available");
        let models_directory = test_directory.path().join("models");
        let job_store = DownloadJobStore::new(models_directory.clone());
        job_store
            .create(&git_blob_job(git_blob_sha1_hex(ROMEO_AND_JULIET)))
            .expect("Git blob manifest job should persist");
        let staged_file = staged_file_path(&models_directory);
        fs::create_dir_all(
            staged_file
                .parent()
                .expect("staged file should have a parent"),
        )
        .expect("staging directory should exist");
        fs::write(&staged_file, ROMEO_AND_JULIET).expect("complete staged file should be written");
        let transport = Arc::new(ScriptedPayloadTransport::new([]));
        let transfer = DownloadPayloadTransfer::new(
            job_store,
            transport.clone(),
            disabled_attribution(&test_directory),
            DownloadTransferControl::new(),
        );

        assert!(matches!(
            transfer
                .resume(200)
                .await
                .expect("Git blob SHA-1 should verify"),
            DownloadPayloadTransferOutcome::ReadyToPublish(_)
        ));
        assert!(transport.requests().is_empty());
    })
    .await
    .expect("Git blob verification should remain bounded");
}

#[tokio::test]
async fn should_publish_live_progress_before_a_large_file_finishes() {
    tokio::time::timeout(Duration::from_secs(5), async {
        const CHECKPOINT_BYTES: usize = 1_000_000;
        let first_chunk = vec![b'R'; CHECKPOINT_BYTES];
        let remaining_chunk = b"omeo and Juliet";
        let mut complete_payload = first_chunk.clone();
        complete_payload.extend_from_slice(remaining_chunk);
        let test_directory = TempDir::new().expect("temporary directory should be available");
        let models_directory = test_directory.path().join("models");
        let job_store = DownloadJobStore::new(models_directory);
        let payload_bytes = complete_payload.len() as u64;
        job_store
            .create(&parse_job_with_size(
                "paused",
                0,
                payload_bytes,
                "sha256",
                &sha256_hex(&complete_payload),
                None,
            ))
            .expect("exact manifest job should persist");
        let observed_store = job_store.clone();
        let first_chunk_bytes = Bytes::from(first_chunk);
        let remaining_chunk_bytes = Bytes::from_static(remaining_chunk);
        let live_progress = tokio::spawn(async move {
            for _poll_attempt in 0..100 {
                if let Some(download_job) = observed_store.load().expect("live job should load") {
                    if download_job.bytes_completed() >= CHECKPOINT_BYTES as u64
                        && download_job.bytes_completed() < download_job.bytes_total()
                    {
                        return download_job.bytes_completed();
                    }
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
            panic!("live progress was not published before the file finished");
        });
        let delayed_stream = stream::iter([Ok(first_chunk_bytes)]).chain(stream::once(async {
            tokio::time::sleep(Duration::from_millis(40)).await;
            Ok(remaining_chunk_bytes)
        }));
        let transfer = DownloadPayloadTransfer::new(
            job_store,
            Arc::new(ScriptedPayloadTransport::new([HubPayloadResponse::new(
                200,
                None,
                Some(complete_payload.len() as u64),
                Box::pin(delayed_stream),
            )])),
            disabled_attribution(&test_directory),
            DownloadTransferControl::new(),
        );

        transfer
            .resume(200)
            .await
            .expect("large file should finish after publishing live progress");
        let published_bytes = live_progress
            .await
            .expect("live progress observer should finish");
        assert_eq!(published_bytes, CHECKPOINT_BYTES as u64);
    })
    .await
    .expect("live progress journey should remain bounded");
}

#[tokio::test]
async fn should_pause_after_synchronizing_received_bytes_and_resume_later() {
    tokio::time::timeout(Duration::from_secs(5), async {
        let test_directory = TempDir::new().expect("temporary directory should be available");
        let models_directory = test_directory.path().join("models");
        let job_store = DownloadJobStore::new(models_directory.clone());
        job_store
            .create(&sha256_job("paused", 0, sha256_hex(ROMEO_AND_JULIET), None))
            .expect("exact manifest job should persist");
        let transfer_control = DownloadTransferControl::new();
        let pause_control = transfer_control.clone();
        let pausing_stream = stream::once(async move {
            pause_control.request_pause();
            Ok(Bytes::from_static(b"Romeo "))
        });
        let transport = Arc::new(ScriptedPayloadTransport::new([HubPayloadResponse::new(
            200,
            None,
            Some(ROMEO_AND_JULIET.len() as u64),
            Box::pin(pausing_stream),
        )]));
        let transfer = DownloadPayloadTransfer::new(
            job_store.clone(),
            transport,
            disabled_attribution(&test_directory),
            transfer_control,
        );

        let paused_job = match transfer.resume(200).await.expect("pause should be durable") {
            DownloadPayloadTransferOutcome::Paused(paused_job) => paused_job,
            DownloadPayloadTransferOutcome::ReadyToPublish(_) => panic!("transfer should pause"),
        };

        assert_eq!(paused_job.state(), DownloadJobState::Paused);
        assert_eq!(paused_job.bytes_completed(), 6);
        assert_eq!(
            fs::read(staged_file_path(&models_directory))
                .expect("synchronized prefix should be readable"),
            b"Romeo "
        );
        assert_eq!(
            job_store
                .load()
                .expect("paused job should load")
                .expect("paused job should exist"),
            paused_job
        );
    })
    .await
    .expect("pause journey should remain bounded");
}

#[tokio::test]
async fn should_retain_staging_and_fail_closed_for_bad_range_or_checksum() {
    for (response, expected_error_code) in [
        (
            payload_response(403, None, []),
            DownloadJobPublicErrorCode::DownloadGated,
        ),
        (
            payload_response(200, None, [b"and Juliet".as_slice()]),
            DownloadJobPublicErrorCode::DownloadFailed,
        ),
        (
            payload_response(206, Some("bytes 6-15/16"), [b"and Juliex".as_slice()]),
            DownloadJobPublicErrorCode::ChecksumMismatch,
        ),
    ] {
        let test_directory = TempDir::new().expect("temporary directory should be available");
        let models_directory = test_directory.path().join("models");
        let job_store = DownloadJobStore::new(models_directory.clone());
        job_store
            .create(&sha256_job("paused", 0, sha256_hex(ROMEO_AND_JULIET), None))
            .expect("exact manifest job should persist");
        let staged_file = staged_file_path(&models_directory);
        fs::create_dir_all(
            staged_file
                .parent()
                .expect("staged file should have a parent"),
        )
        .expect("staging directory should exist");
        fs::write(&staged_file, b"Romeo ").expect("resumable prefix should be written");
        let transfer = DownloadPayloadTransfer::new(
            job_store.clone(),
            Arc::new(ScriptedPayloadTransport::new([response])),
            disabled_attribution(&test_directory),
            DownloadTransferControl::new(),
        );

        transfer
            .resume(200)
            .await
            .expect_err("invalid payload must fail");

        let failed_job = job_store
            .load()
            .expect("failed job should load")
            .expect("failed job should exist");
        assert_eq!(failed_job.state(), DownloadJobState::Failed);
        assert_eq!(failed_job.error_code(), Some(expected_error_code));
        assert!(staged_file.exists());
    }
}

#[tokio::test]
async fn should_synchronize_partial_progress_before_persisting_a_stream_failure() {
    tokio::time::timeout(Duration::from_secs(5), async {
        let test_directory = TempDir::new().expect("temporary directory should be available");
        let models_directory = test_directory.path().join("models");
        let job_store = DownloadJobStore::new(models_directory.clone());
        job_store
            .create(&sha256_job("paused", 0, sha256_hex(ROMEO_AND_JULIET), None))
            .expect("exact manifest job should persist");
        let response = HubPayloadResponse::new(
            200,
            None,
            Some(ROMEO_AND_JULIET.len() as u64),
            Box::pin(stream::iter([
                Ok(Bytes::from_static(b"Romeo ")),
                Err(HubTransportError::new("scripted interrupted stream")),
            ])),
        );
        let transfer = DownloadPayloadTransfer::new(
            job_store.clone(),
            Arc::new(ScriptedPayloadTransport::new([response])),
            disabled_attribution(&test_directory),
            DownloadTransferControl::new(),
        );

        transfer
            .resume(200)
            .await
            .expect_err("interrupted stream must fail");

        let failed_job = job_store
            .load()
            .expect("failed job should load")
            .expect("failed job should remain resumable");
        assert_eq!(failed_job.state(), DownloadJobState::Failed);
        assert_eq!(failed_job.bytes_completed(), 6);
        assert_eq!(
            fs::read(staged_file_path(&models_directory)).expect("staged prefix should remain"),
            b"Romeo "
        );
    })
    .await
    .expect("stream failure journey should remain bounded");
}

#[tokio::test]
async fn should_finish_publication_after_restart_when_refresh_failed_after_atomic_rename() {
    tokio::time::timeout(Duration::from_secs(5), async {
        let test_directory = TempDir::new().expect("temporary directory should be available");
        let models_directory = test_directory.path().join("models");
        let job_store = DownloadJobStore::new(models_directory.clone());
        job_store
            .create(&sha256_job(
                "publishing",
                ROMEO_AND_JULIET.len() as u64,
                sha256_hex(ROMEO_AND_JULIET),
                None,
            ))
            .expect("publishing job should persist");
        let staged_file = staged_file_path(&models_directory);
        fs::create_dir_all(
            staged_file
                .parent()
                .expect("staged file should have a parent"),
        )
        .expect("staging directory should exist");
        fs::write(&staged_file, ROMEO_AND_JULIET).expect("verified staging should be written");
        let first_publication =
            DownloadPublication::new(job_store.clone(), disabled_attribution(&test_directory));

        first_publication
            .publish(Arc::new(FailingRefresh))
            .await
            .expect_err("refresh failure should retain publication intent");

        assert_eq!(
            job_store
                .load()
                .expect("publishing job should load")
                .expect("publishing job should remain")
                .state(),
            DownloadJobState::Publishing
        );
        assert!(!staged_file.exists());
        let refresh = Arc::new(RecordingRefresh::default());
        let published_directory =
            DownloadPublication::new(job_store.clone(), disabled_attribution(&test_directory))
                .publish(refresh.clone())
                .await
                .expect("restart should refresh the already-renamed destination");

        assert_eq!(refresh.refreshed_directories(), [published_directory]);
        assert!(
            job_store
                .load()
                .expect("completed job lookup should succeed")
                .is_none()
        );
    })
    .await
    .expect("publication recovery should remain bounded");
}

#[tokio::test]
async fn should_reject_an_existing_destination_before_persisting_publication_intent() {
    tokio::time::timeout(Duration::from_secs(5), async {
        let test_directory = TempDir::new().expect("temporary directory should be available");
        let models_directory = test_directory.path().join("models");
        let job_store = DownloadJobStore::new(models_directory.clone());
        job_store
            .create(&sha256_job("paused", 0, sha256_hex(ROMEO_AND_JULIET), None))
            .expect("exact manifest job should persist");
        let staged_file = staged_file_path(&models_directory);
        fs::create_dir_all(
            staged_file
                .parent()
                .expect("staged file should have a parent"),
        )
        .expect("staging directory should exist");
        fs::write(&staged_file, ROMEO_AND_JULIET).expect("complete staging should be written");
        let existing_destination = models_directory.join(REPOSITORY_ID);
        fs::create_dir_all(&existing_destination).expect("existing destination should be created");
        fs::write(existing_destination.join("keep.txt"), b"Juliet")
            .expect("existing destination fixture should be written");
        let transfer = DownloadPayloadTransfer::new(
            job_store.clone(),
            Arc::new(ScriptedPayloadTransport::new([])),
            disabled_attribution(&test_directory),
            DownloadTransferControl::new(),
        );

        transfer
            .resume(200)
            .await
            .expect_err("existing destination must reject publication");

        let failed_job = job_store
            .load()
            .expect("failed job should load")
            .expect("failed job should remain");
        assert_eq!(failed_job.state(), DownloadJobState::Failed);
        assert_eq!(
            failed_job.error_code(),
            Some(DownloadJobPublicErrorCode::ModelAlreadyPresent)
        );
        assert_eq!(
            fs::read(existing_destination.join("keep.txt"))
                .expect("existing destination should remain readable"),
            b"Juliet"
        );
        assert!(staged_file.exists());
    })
    .await
    .expect("existing destination rejection should remain bounded");
}
