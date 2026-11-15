//! Acceptance journeys for one payload piece's requests and its recovery: a
//! piece asks for exactly its own bytes, and a piece whose connection drops or
//! whose response ends early is asked for again from the first missing byte.

use std::{
    fs,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

use astronomical_daemon::{
    DownloadJobStore, DownloadPayloadTransfer, DownloadPayloadTransferError,
    DownloadPayloadTransferOutcome, DownloadTransferControl, HubPayloadRequest, HubPayloadResponse,
    HubTransportError, PAYLOAD_TRANSFER_SEGMENT_BYTES,
};
use bytes::Bytes;
use futures_util::stream;
use tempfile::TempDir;

use super::download_payload_support as support;

use support::{
    PathKeyedPayloadTransport, ScriptedResponseFactory, disabled_attribution,
    staged_file_path_for_relative_path,
};

/// How a scripted response to one request misbehaves, if at all.
#[derive(Clone, Copy)]
enum ScriptedFault {
    /// The requested bytes, whole.
    None,
    /// This many bytes, then the connection fails.
    DropsAfter(usize),
    /// This many bytes, then the response simply ends.
    EndsAfter(usize),
    /// A range the request did not ask for.
    WrongRange,
}

/// A server that honors the exact range of each request, misbehaving on each
/// request as `fault_for_request_number` says (numbered from zero).
fn exact_range_factories(
    complete_payload: &[u8],
    fault_for_request_number: impl Fn(usize) -> ScriptedFault + Send + Sync + 'static,
) -> Vec<(String, ScriptedResponseFactory)> {
    let owned_payload_bytes = complete_payload.to_vec();
    let request_count = AtomicUsize::new(0);
    vec![(
        support::RELATIVE_PATH.to_owned(),
        Box::new(move |request: &HubPayloadRequest| {
            let request_number = request_count.fetch_add(1, Ordering::AcqRel);
            let total_bytes = owned_payload_bytes.len();
            let start_offset = request.resume_offset_bytes() as usize;
            let end_offset = request
                .end_offset_bytes()
                .map_or(total_bytes, |end_offset_bytes| end_offset_bytes as usize);
            let requested_bytes = owned_payload_bytes[start_offset..end_offset].to_vec();
            let exact_range = format!("bytes {start_offset}-{}/{total_bytes}", end_offset - 1);
            match fault_for_request_number(request_number) {
                ScriptedFault::None => ranged_response(&exact_range, requested_bytes, None),
                ScriptedFault::DropsAfter(delivered_byte_count) => ranged_response(
                    &exact_range,
                    requested_bytes[..delivered_byte_count].to_vec(),
                    Some(HubTransportError::new("scripted connection reset")),
                ),
                ScriptedFault::EndsAfter(delivered_byte_count) => ranged_response(
                    &exact_range,
                    requested_bytes[..delivered_byte_count].to_vec(),
                    None,
                ),
                ScriptedFault::WrongRange => ranged_response(
                    &format!("bytes 7-{}/{total_bytes}", end_offset - 1),
                    requested_bytes,
                    None,
                ),
            }
        }) as ScriptedResponseFactory,
    )]
}

/// A `206` that delivers `delivered_bytes` and then, when given, fails. The
/// length is left undeclared so an early end is seen only in the stream.
fn ranged_response(
    content_range: &str,
    delivered_bytes: Vec<u8>,
    closing_error: Option<HubTransportError>,
) -> HubPayloadResponse {
    let mut stream_items = vec![Ok(Bytes::from(delivered_bytes))];
    if let Some(closing_error) = closing_error {
        stream_items.push(Err(closing_error));
    }
    HubPayloadResponse::new(
        206,
        Some(content_range.to_owned()),
        None,
        Box::pin(stream::iter(stream_items)),
    )
}

fn patterned_payload(byte_count: usize) -> Vec<u8> {
    (0..byte_count)
        .map(|byte_index| (byte_index % 251) as u8)
        .collect()
}

struct RecoveryJourney {
    transfer_outcome: Result<DownloadPayloadTransferOutcome, DownloadPayloadTransferError>,
    requested_ranges: Vec<(u64, Option<u64>)>,
    staged_bytes: Vec<u8>,
}

/// Downloads `complete_payload` from a server with the given faults and
/// reports the outcome, the ranges that were asked for, and the staged bytes.
async fn run_recovery_journey(
    complete_payload: &[u8],
    fault_for_request_number: impl Fn(usize) -> ScriptedFault + Send + Sync + 'static,
) -> RecoveryJourney {
    let test_directory = TempDir::new().expect("temporary directory should be available");
    let models_directory = test_directory.path().join("models");
    let job_store = DownloadJobStore::new(models_directory.clone());
    job_store
        .create(&support::parse_job_with_size(
            "paused",
            0,
            complete_payload.len() as u64,
            "sha256",
            &support::sha256_hex(complete_payload),
            None,
        ))
        .expect("the job should persist");
    let transport = Arc::new(PathKeyedPayloadTransport::new(exact_range_factories(
        complete_payload,
        fault_for_request_number,
    )));
    let transfer = DownloadPayloadTransfer::new(
        job_store,
        transport.clone(),
        disabled_attribution(&test_directory),
        DownloadTransferControl::new(),
    );
    let transfer_outcome = tokio::time::timeout(Duration::from_secs(10), transfer.resume(200))
        .await
        .expect("the journey should remain bounded");
    RecoveryJourney {
        transfer_outcome,
        requested_ranges: transport
            .requests()
            .iter()
            .map(|request| (request.resume_offset_bytes(), request.end_offset_bytes()))
            .collect(),
        staged_bytes: fs::read(staged_file_path_for_relative_path(
            &models_directory,
            support::RELATIVE_PATH,
        ))
        .unwrap_or_default(),
    }
}

#[tokio::test]
async fn should_ask_for_exactly_the_bytes_of_each_piece() {
    let complete_payload = patterned_payload(PAYLOAD_TRANSFER_SEGMENT_BYTES as usize + 100);

    let journey = run_recovery_journey(&complete_payload, |_| ScriptedFault::None).await;

    let mut requested_ranges = journey.requested_ranges.clone();
    requested_ranges.sort_unstable();
    assert_eq!(
        requested_ranges,
        vec![
            (0, Some(PAYLOAD_TRANSFER_SEGMENT_BYTES)),
            (
                PAYLOAD_TRANSFER_SEGMENT_BYTES,
                Some(PAYLOAD_TRANSFER_SEGMENT_BYTES + 100)
            ),
        ]
    );
    assert!(matches!(
        journey.transfer_outcome,
        Ok(DownloadPayloadTransferOutcome::ReadyToPublish(_))
    ));
    assert_eq!(journey.staged_bytes, complete_payload);
}

#[tokio::test]
async fn should_ask_again_from_the_first_missing_byte_after_a_connection_drops() {
    let complete_payload = patterned_payload(1_000);

    let journey = run_recovery_journey(&complete_payload, |request_number| {
        if request_number == 0 {
            ScriptedFault::DropsAfter(400)
        } else {
            ScriptedFault::None
        }
    })
    .await;

    assert_eq!(
        journey.requested_ranges,
        vec![(0, Some(1_000)), (400, Some(1_000))]
    );
    assert!(matches!(
        journey.transfer_outcome,
        Ok(DownloadPayloadTransferOutcome::ReadyToPublish(_))
    ));
    assert_eq!(journey.staged_bytes, complete_payload);
}

#[tokio::test]
async fn should_ask_again_when_a_response_ends_before_its_piece_does() {
    let complete_payload = patterned_payload(1_000);

    let journey = run_recovery_journey(&complete_payload, |request_number| {
        if request_number == 0 {
            ScriptedFault::EndsAfter(300)
        } else {
            ScriptedFault::None
        }
    })
    .await;

    assert_eq!(
        journey.requested_ranges,
        vec![(0, Some(1_000)), (300, Some(1_000))]
    );
    assert_eq!(journey.staged_bytes, complete_payload);
}

#[tokio::test]
async fn should_give_up_after_three_retries_and_keep_the_bytes_that_arrived() {
    let complete_payload = patterned_payload(1_000);

    let journey = run_recovery_journey(&complete_payload, |_| ScriptedFault::DropsAfter(100)).await;

    assert_eq!(
        journey.requested_ranges,
        vec![
            (0, Some(1_000)),
            (100, Some(1_000)),
            (200, Some(1_000)),
            (300, Some(1_000))
        ],
        "one first attempt and three retries, each from the first missing byte"
    );
    assert!(matches!(
        journey.transfer_outcome,
        Err(DownloadPayloadTransferError::Transport(_))
    ));
    assert_eq!(journey.staged_bytes, complete_payload[..400]);
}

#[tokio::test]
async fn should_not_ask_again_after_an_answer_for_the_wrong_range() {
    let complete_payload = patterned_payload(1_000);

    let journey = run_recovery_journey(&complete_payload, |_| ScriptedFault::WrongRange).await;

    assert_eq!(journey.requested_ranges, vec![(0, Some(1_000))]);
    assert!(matches!(
        journey.transfer_outcome,
        Err(DownloadPayloadTransferError::InvalidRangeResponse)
    ));
}
