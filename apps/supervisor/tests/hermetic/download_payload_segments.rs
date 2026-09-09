//! Acceptance journey for intra-file segmented payload transfer.

use std::{fs, sync::Arc, time::Duration};

use astronomical_supervisor::{
    DownloadJobStore, DownloadPayloadTransfer, DownloadPayloadTransferOutcome,
    DownloadTransferControl, HubPayloadRequest, HubPayloadResponse, PAYLOAD_TRANSFER_SEGMENT_BYTES,
};
use bytes::Bytes;
use futures_util::stream;
use tempfile::TempDir;

use super::download_payload_support as support;

use support::{
    ScriptedResponseFactory, disabled_attribution, payload_response,
    staged_file_path_for_relative_path,
};

/// Segments one large payload: every request restarts at its own offset and serves the whole
/// remaining tail, mirroring how the real delivery endpoint ignores an upper range bound.
fn tail_serving_factories(complete_payload: &[u8]) -> Vec<(String, ScriptedResponseFactory)> {
    let owned_payload_bytes = complete_payload.to_vec();
    let total_bytes = owned_payload_bytes.len() as u64;
    vec![(
        support::RELATIVE_PATH.to_owned(),
        Box::new(move |request: &HubPayloadRequest| {
            let resume_offset_bytes = request.resume_offset_bytes();
            let remaining_payload = &owned_payload_bytes[resume_offset_bytes as usize..];
            let content_range = if resume_offset_bytes == 0 {
                None
            } else {
                Some(format!(
                    "bytes {resume_offset_bytes}-{last_byte_index}/{total_bytes}",
                    last_byte_index = total_bytes - 1,
                ))
            };
            payload_response(
                if resume_offset_bytes == 0 { 200 } else { 206 },
                content_range.as_deref(),
                [remaining_payload],
            )
        }) as ScriptedResponseFactory,
    )]
}

/// Same tail-serving behavior, but the first segment's stream stalls so later segments complete
/// first and must be held by the assembler until the prefix reaches them.
fn delayed_first_segment_factories(
    complete_payload: &[u8],
) -> Vec<(String, ScriptedResponseFactory)> {
    let owned_payload_bytes = complete_payload.to_vec();
    let total_bytes = owned_payload_bytes.len() as u64;
    vec![(
        support::RELATIVE_PATH.to_owned(),
        Box::new(move |request: &HubPayloadRequest| {
            let resume_offset_bytes = request.resume_offset_bytes();
            let remaining_payload = owned_payload_bytes[resume_offset_bytes as usize..].to_vec();
            let total_byte_count = owned_payload_bytes.len() as u64;
            let content_range = if resume_offset_bytes == 0 {
                None
            } else {
                Some(format!(
                    "bytes {resume_offset_bytes}-{last_byte_index}/{total_bytes}",
                    last_byte_index = total_bytes - 1,
                ))
            };
            let status = if resume_offset_bytes == 0 { 200 } else { 206 };
            if resume_offset_bytes != 0 {
                return payload_response(
                    status,
                    content_range.as_deref(),
                    [remaining_payload.as_slice()],
                );
            }
            HubPayloadResponse::new(
                status,
                None,
                Some(total_byte_count),
                Box::pin(stream::once(async move {
                    tokio::time::sleep(Duration::from_millis(100)).await;
                    Ok(Bytes::copy_from_slice(&remaining_payload))
                })),
            )
        }) as ScriptedResponseFactory,
    )]
}

#[tokio::test]
async fn should_segment_a_large_file_into_parallel_ranged_requests_within_the_bounded_window() {
    tokio::time::timeout(Duration::from_secs(10), async {
        // Two full segments plus a short final segment: one large file must transfer as several
        // concurrent ranged requests instead of one serial connection, so no single shard can
        // straggle at the end of a download.
        let segment_bytes = PAYLOAD_TRANSFER_SEGMENT_BYTES as usize;
        let mut complete_payload = vec![b'R'; segment_bytes * 2 + 100];
        for (byte_index, payload_byte) in complete_payload.iter_mut().enumerate() {
            *payload_byte = (byte_index % 251) as u8;
        }
        let test_directory = TempDir::new().expect("temporary directory should be available");
        let models_directory = test_directory.path().join("models");
        let job_store = DownloadJobStore::new(models_directory.clone());
        job_store
            .create(&support::parse_job_with_size(
                "paused",
                0,
                complete_payload.len() as u64,
                "sha256",
                &support::sha256_hex(&complete_payload),
                None,
            ))
            .expect("segmented job should persist");
        let transport = Arc::new(support::PathKeyedPayloadTransport::new(
            tail_serving_factories(&complete_payload),
        ));
        let transfer = DownloadPayloadTransfer::new(
            job_store.clone(),
            transport.clone(),
            disabled_attribution(&test_directory),
            DownloadTransferControl::new(),
        );

        let transfer_outcome = transfer
            .resume(200)
            .await
            .expect("segmented transfer should verify and publish");

        let requested_resume_offsets = transport
            .requests()
            .iter()
            .map(HubPayloadRequest::resume_offset_bytes)
            .collect::<Vec<_>>();
        assert_eq!(
            requested_resume_offsets,
            vec![
                0,
                PAYLOAD_TRANSFER_SEGMENT_BYTES,
                PAYLOAD_TRANSFER_SEGMENT_BYTES * 2
            ],
            "each segment must request its own open-ended range: {requested_resume_offsets:?}"
        );
        assert_eq!(
            transport.maximum_active_request_count(),
            3,
            "segments of one file must transfer concurrently within the window"
        );
        assert_eq!(
            fs::read(staged_file_path_for_relative_path(
                &models_directory,
                support::RELATIVE_PATH,
            ))
            .expect("staged file should be readable"),
            complete_payload,
            "segmented assembly must reproduce the provider payload byte for byte"
        );
        assert!(matches!(
            transfer_outcome,
            DownloadPayloadTransferOutcome::ReadyToPublish(_)
        ));
    })
    .await
    .expect("segmented transfer journey should remain bounded");
}

#[tokio::test]
async fn should_assemble_out_of_order_segments_without_corruption() {
    tokio::time::timeout(Duration::from_secs(10), async {
        // The first segment's stream stalls, so later segments finish first and must wait inside
        // the assembler instead of overwriting the contiguous prefix out of order.
        let segment_bytes = PAYLOAD_TRANSFER_SEGMENT_BYTES as usize;
        let mut complete_payload = vec![b'J'; segment_bytes * 2 + 50];
        for (byte_index, payload_byte) in complete_payload.iter_mut().enumerate() {
            *payload_byte = (byte_index * 7 % 253) as u8;
        }
        let test_directory = TempDir::new().expect("temporary directory should be available");
        let models_directory = test_directory.path().join("models");
        let job_store = DownloadJobStore::new(models_directory.clone());
        job_store
            .create(&support::parse_job_with_size(
                "paused",
                0,
                complete_payload.len() as u64,
                "sha256",
                &support::sha256_hex(&complete_payload),
                None,
            ))
            .expect("reordered job should persist");
        let transport = Arc::new(support::PathKeyedPayloadTransport::new(
            delayed_first_segment_factories(&complete_payload),
        ));
        let transfer = DownloadPayloadTransfer::new(
            job_store,
            transport.clone(),
            disabled_attribution(&test_directory),
            DownloadTransferControl::new(),
        );

        let transfer_outcome = transfer
            .resume(200)
            .await
            .expect("reordered segments should verify and publish");

        assert_eq!(transport.requests().len(), 3);
        assert!(matches!(
            transfer_outcome,
            DownloadPayloadTransferOutcome::ReadyToPublish(_)
        ));
        assert_eq!(
            fs::read(staged_file_path_for_relative_path(
                &models_directory,
                support::RELATIVE_PATH,
            ))
            .expect("staged file should be readable"),
            complete_payload,
            "out-of-order segment arrival must not corrupt ordered assembly"
        );
    })
    .await
    .expect("reordered assembly journey should remain bounded");
}
