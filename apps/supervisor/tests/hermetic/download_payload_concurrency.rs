//! Acceptance journey for bounded concurrent payload transfer across manifest files.

use std::{fs, sync::Arc, time::Duration};

use astronomical_supervisor::{
    DownloadJobPublicErrorCode, DownloadJobState, DownloadJobStore, DownloadPayloadTransfer,
    DownloadPayloadTransferOutcome, DownloadTransferControl, HubPayloadRequest, HubPayloadResponse,
    MAXIMUM_CONCURRENT_PAYLOAD_TRANSFERS,
};
use bytes::Bytes;
use futures_util::stream;
use tempfile::TempDir;

use super::download_payload_support as support;

use support::{
    ScriptedResponseFactory, disabled_attribution, multi_file_paused_job,
    payload_relative_path_for_request, payload_response, resume_payload_response,
    staged_file_path_for_relative_path,
};

const ROMEO_AND_JULIET_PAYLOAD: &[u8] = b"Romeo and Juliet";
const HAMLET_PAYLOAD: &[u8] = b"To be, or not to be";

fn path_keyed_transport(file_payloads: &[(&str, &[u8])]) -> support::PathKeyedPayloadTransport {
    support::PathKeyedPayloadTransport::new(
        file_payloads
            .iter()
            .map(|(relative_path, payload_bytes)| {
                let owned_payload_bytes = payload_bytes.to_vec();
                (
                    (*relative_path).to_owned(),
                    Box::new(move |_request: &HubPayloadRequest| {
                        payload_response(200, None, [&owned_payload_bytes])
                    }) as ScriptedResponseFactory,
                )
            })
            .collect::<Vec<_>>(),
    )
}

fn resume_offset_factories(
    file_payloads: &[(&str, &[u8])],
    resume_offsets_by_path: &[(String, u64)],
) -> Vec<(String, ScriptedResponseFactory)> {
    file_payloads
        .iter()
        .map(|(relative_path, payload_bytes)| {
            let resume_offset_bytes = resume_offsets_by_path
                .iter()
                .find(|(file_relative_path, _)| file_relative_path.as_str() == *relative_path)
                .map(|(_, bytes_on_disk)| *bytes_on_disk)
                .expect("fixture progress should cover every manifest file");
            let owned_payload_bytes = payload_bytes.to_vec();
            (
                (*relative_path).to_owned(),
                Box::new(move |_request: &HubPayloadRequest| {
                    resume_payload_response(&owned_payload_bytes, resume_offset_bytes)
                }) as ScriptedResponseFactory,
            )
        })
        .collect()
}

fn recorded_bytes_on_disk<'a>(
    download_job: &'a astronomical_supervisor::DownloadJob,
    relative_path: &str,
) -> u64 {
    download_job
        .files()
        .iter()
        .find(|file| file.relative_path() == relative_path)
        .expect("durable job should retain every manifest file")
        .bytes_on_disk()
}

#[tokio::test]
async fn should_transfer_incomplete_files_concurrently_within_the_bounded_window() {
    tokio::time::timeout(Duration::from_secs(5), async {
        let tempest_payload = b"We are such stuff".as_slice();
        let file_payloads: [(&str, &[u8]); 3] = [
            ("weights/romeo-and-juliet.txt", ROMEO_AND_JULIET_PAYLOAD),
            ("weights/hamlet.txt", HAMLET_PAYLOAD),
            ("weights/tempest.txt", tempest_payload),
        ];
        let test_directory = TempDir::new().expect("temporary directory should be available");
        let models_directory = test_directory.path().join("models");
        let job_store = DownloadJobStore::new(models_directory.clone());
        job_store
            .create(&multi_file_paused_job(&file_payloads))
            .expect("multi-file manifest job should persist");
        let transport = Arc::new(path_keyed_transport(&file_payloads));
        let transfer = DownloadPayloadTransfer::new(
            job_store.clone(),
            transport.clone(),
            disabled_attribution(&test_directory),
            DownloadTransferControl::new(),
        );

        let transfer_outcome = transfer
            .resume(200)
            .await
            .expect("concurrent transfer should verify and publish");

        let requested_relative_paths = transport
            .requests()
            .iter()
            .map(|request| payload_relative_path_for_request(request.url()))
            .collect::<Vec<_>>();
        assert_eq!(
            requested_relative_paths.len(),
            file_payloads.len(),
            "each pending file must request exactly once: {requested_relative_paths:?}"
        );
        assert_eq!(
            transport.maximum_active_request_count(),
            file_payloads.len(),
            "all pending files must be in flight together within the bounded window"
        );
        for (relative_path, payload_bytes) in file_payloads {
            assert_eq!(
                fs::read(staged_file_path_for_relative_path(
                    &models_directory,
                    relative_path
                ))
                .expect("staged file should be readable"),
                payload_bytes,
                "{relative_path} should hold its complete payload"
            );
        }
        assert!(matches!(
            transfer_outcome,
            DownloadPayloadTransferOutcome::ReadyToPublish(_)
        ));
    })
    .await
    .expect("concurrent transfer journey should remain bounded");
}

#[tokio::test]
async fn should_cap_concurrent_payload_requests_at_the_bounded_window() {
    tokio::time::timeout(Duration::from_secs(5), async {
        const SCENE_FILE_PATHS: [&str; 12] = [
            "weights/scene-one.txt",
            "weights/scene-two.txt",
            "weights/scene-three.txt",
            "weights/scene-four.txt",
            "weights/scene-five.txt",
            "weights/scene-six.txt",
            "weights/scene-seven.txt",
            "weights/scene-eight.txt",
            "weights/scene-nine.txt",
            "weights/scene-ten.txt",
            "weights/scene-eleven.txt",
            "weights/scene-twelve.txt",
        ];
        let file_payloads: Vec<(&str, &[u8])> = SCENE_FILE_PATHS
            .iter()
            .map(|relative_path| (*relative_path, ROMEO_AND_JULIET_PAYLOAD))
            .collect();
        let test_directory = TempDir::new().expect("temporary directory should be available");
        let models_directory = test_directory.path().join("models");
        let job_store = DownloadJobStore::new(models_directory);
        job_store
            .create(&multi_file_paused_job(&file_payloads))
            .expect("multi-file manifest job should persist");
        let transport = Arc::new(path_keyed_transport(&file_payloads));
        let transfer = DownloadPayloadTransfer::new(
            job_store,
            transport.clone(),
            disabled_attribution(&test_directory),
            DownloadTransferControl::new(),
        );

        transfer
            .resume(200)
            .await
            .expect("bounded concurrent transfer should complete");

        assert_eq!(
            transport.requests().len(),
            SCENE_FILE_PATHS.len(),
            "every pending file must still be requested exactly once"
        );
        assert_eq!(
            transport.maximum_active_request_count(),
            MAXIMUM_CONCURRENT_PAYLOAD_TRANSFERS,
            "the in-flight width must never exceed the bounded window"
        );
    })
    .await
    .expect("bounded-window journey should remain bounded");
}

#[tokio::test]
async fn should_pause_across_every_in_flight_file_and_resume_without_progress_mismatch() {
    tokio::time::timeout(Duration::from_secs(5), async {
        let file_payloads: [(&str, &[u8]); 2] = [
            ("weights/romeo-and-juliet.txt", ROMEO_AND_JULIET_PAYLOAD),
            ("weights/hamlet.txt", HAMLET_PAYLOAD),
        ];
        let test_directory = TempDir::new().expect("temporary directory should be available");
        let models_directory = test_directory.path().join("models");
        let job_store = DownloadJobStore::new(models_directory.clone());
        job_store
            .create(&multi_file_paused_job(&file_payloads))
            .expect("multi-file manifest job should persist");
        let transfer_control = DownloadTransferControl::new();
        let pause_control = transfer_control.clone();
        let transport = Arc::new(support::PathKeyedPayloadTransport::new(
            file_payloads
                .iter()
                .map(|(relative_path, payload_bytes)| {
                    let owned_payload_bytes = payload_bytes.to_vec();
                    let payload_byte_count = owned_payload_bytes.len() as u64;
                    let pause_control = pause_control.clone();
                    (
                        (*relative_path).to_owned(),
                        Box::new(move |_request: &HubPayloadRequest| {
                            let owned_payload = owned_payload_bytes.clone();
                            let owned_pause_control = pause_control.clone();
                            let pausing_stream = stream::once(async move {
                                owned_pause_control.request_pause();
                                Ok(Bytes::copy_from_slice(&owned_payload[..4]))
                            });
                            HubPayloadResponse::new(
                                200,
                                None,
                                Some(payload_byte_count),
                                Box::pin(pausing_stream),
                            )
                        }) as ScriptedResponseFactory,
                    )
                })
                .collect::<Vec<_>>(),
        ));
        let transfer = DownloadPayloadTransfer::new(
            job_store.clone(),
            transport,
            disabled_attribution(&test_directory),
            transfer_control,
        );

        let paused_job = match transfer.resume(200).await.expect("pause should be durable") {
            DownloadPayloadTransferOutcome::Paused(paused_job) => paused_job,
            DownloadPayloadTransferOutcome::ReadyToPublish(_) => {
                panic!("transfer should pause while files are in flight")
            }
        };

        assert_eq!(paused_job.state(), DownloadJobState::Paused);
        for (relative_path, _) in file_payloads {
            let staged_path = staged_file_path_for_relative_path(&models_directory, relative_path);
            let staged_bytes = fs::read(&staged_path).unwrap_or_default();
            let recorded_bytes_on_disk = recorded_bytes_on_disk(&paused_job, relative_path);
            assert_eq!(
                staged_bytes.len() as u64,
                recorded_bytes_on_disk,
                "{relative_path} staged length must match durable progress"
            );
            assert!(
                recorded_bytes_on_disk > 0,
                "{relative_path} should have received its first bytes before pausing"
            );
        }

        // Resuming the paused job must complete without a staged-progress mismatch, so each
        // pending file resumes from its own recorded offset with a valid ranged response.
        let resume_offsets_by_path = paused_job
            .files()
            .iter()
            .map(|file| (file.relative_path().to_owned(), file.bytes_on_disk()))
            .collect::<Vec<_>>();
        let resumed_transport = Arc::new(support::PathKeyedPayloadTransport::new(
            resume_offset_factories(&file_payloads, &resume_offsets_by_path),
        ));
        let resumed_transfer = DownloadPayloadTransfer::new(
            job_store.clone(),
            resumed_transport,
            disabled_attribution(&test_directory),
            DownloadTransferControl::new(),
        );
        let resumed_outcome = resumed_transfer
            .resume(300)
            .await
            .expect("resumed concurrent transfer should complete");
        assert!(matches!(
            resumed_outcome,
            DownloadPayloadTransferOutcome::ReadyToPublish(_)
        ));
        for (relative_path, payload_bytes) in file_payloads {
            assert_eq!(
                fs::read(staged_file_path_for_relative_path(
                    &models_directory,
                    relative_path
                ))
                .expect("staged file should be readable after resume"),
                payload_bytes,
                "{relative_path} should hold its complete payload after resume"
            );
        }
    })
    .await
    .expect("concurrent pause and resume journey should remain bounded");
}

#[tokio::test]
async fn should_keep_sibling_progress_durable_when_one_concurrent_file_fails() {
    tokio::time::timeout(Duration::from_secs(5), async {
        // The manifest declares more bytes than this transport serves for the hamlet file, so
        // hamlet's response framing is invalid while the romeo sibling streams normally.
        let romeo_payload = ROMEO_AND_JULIET_PAYLOAD;
        let hamlet_payload = HAMLET_PAYLOAD;
        let short_payload = &hamlet_payload[..5];
        let file_payloads: [(&str, &[u8]); 2] = [
            ("weights/romeo-and-juliet.txt", romeo_payload),
            ("weights/hamlet.txt", hamlet_payload),
        ];
        let test_directory = TempDir::new().expect("temporary directory should be available");
        let models_directory = test_directory.path().join("models");
        let job_store = DownloadJobStore::new(models_directory.clone());
        job_store
            .create(&multi_file_paused_job(&file_payloads))
            .expect("multi-file manifest job should persist");
        let transport = Arc::new(support::PathKeyedPayloadTransport::new([
            (
                "weights/romeo-and-juliet.txt".to_owned(),
                Box::new(move |_request: &HubPayloadRequest| {
                    payload_response(200, None, [romeo_payload])
                }) as support::ScriptedResponseFactory,
            ),
            (
                "weights/hamlet.txt".to_owned(),
                Box::new(move |_request: &HubPayloadRequest| {
                    payload_response(200, None, [short_payload])
                }) as support::ScriptedResponseFactory,
            ),
        ]));
        let transfer = DownloadPayloadTransfer::new(
            job_store.clone(),
            transport,
            disabled_attribution(&test_directory),
            DownloadTransferControl::new(),
        );

        let transfer_error = transfer
            .resume(200)
            .await
            .expect_err("a short payload must fail the journey");
        assert!(
            matches!(
                transfer_error,
                astronomical_supervisor::DownloadPayloadTransferError::InvalidPayloadLength
            ),
            "unexpected error: {transfer_error:?}"
        );

        let failed_job = job_store
            .load()
            .expect("failed job should load")
            .expect("failed job should exist");
        assert_eq!(failed_job.state(), DownloadJobState::Failed);
        assert_eq!(
            failed_job.error_code(),
            Some(DownloadJobPublicErrorCode::DownloadFailed)
        );
        let resume_offsets_by_path = failed_job
            .files()
            .iter()
            .map(|file| (file.relative_path().to_owned(), file.bytes_on_disk()))
            .collect::<Vec<_>>();
        for (relative_path, _) in file_payloads {
            let staged_bytes = fs::read(staged_file_path_for_relative_path(
                &models_directory,
                relative_path,
            ))
            .unwrap_or_default();
            assert_eq!(
                staged_bytes.len() as u64,
                recorded_bytes_on_disk(&failed_job, relative_path),
                "{relative_path} staged length must match durable progress after the failure"
            );
        }

        // A later resume must fetch only each file's missing bytes and complete the journey.
        let resumed_transport = Arc::new(support::PathKeyedPayloadTransport::new(
            resume_offset_factories(&file_payloads, &resume_offsets_by_path),
        ));
        let resumed_transfer = DownloadPayloadTransfer::new(
            job_store.clone(),
            resumed_transport,
            disabled_attribution(&test_directory),
            DownloadTransferControl::new(),
        );
        let resumed_outcome = resumed_transfer
            .resume(300)
            .await
            .expect("resumed journey should complete after failure recovery");
        assert!(matches!(
            resumed_outcome,
            DownloadPayloadTransferOutcome::ReadyToPublish(_)
        ));
    })
    .await
    .expect("failure-and-recovery journey should remain bounded");
}
