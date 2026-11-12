//! Unit contracts for correlating a worker's speech events with the one
//! active request: how audio is collected, what the caller receives, when
//! the worker slot is released, and which event orders are violations.

use std::sync::{Arc, RwLock};
use std::time::Duration;

use astronomical_ipc_protocol::{
    RequestId, SpeechResultMetadata, SpokenAudioSegment, WorkerEvent, WorkerModelCapabilities,
    WorkerSpeechCapabilities,
};
use astronomical_speech_contract::SpeechFailure;
use tokio::sync::{Semaphore, mpsc};

use super::handle_worker_speech_event;
use crate::worker::loop_types::{ActiveSpeechGeneration, ActiveWorkerRequest};
use crate::{SpeechExecutionError, SpeechOutput, WorkerHealthSnapshot};

const ACTIVE_REQUEST_ID: u64 = 9;

type SpeechResultReceiver = mpsc::Receiver<Result<SpeechOutput, SpeechExecutionError>>;

fn speech_health_snapshot() -> Arc<RwLock<WorkerHealthSnapshot>> {
    Arc::new(RwLock::new(WorkerHealthSnapshot::ready_with_model(
        String::from("Kokoro-82M-bf16"),
        WorkerModelCapabilities::speech(WorkerSpeechCapabilities {
            sample_rate_hertz: 24_000,
            voice_names: vec!["af_heart".to_owned()],
            default_voice_name: "af_heart".to_owned(),
            supports_speed: true,
        }),
    )))
}

fn active_speech_request() -> (Option<ActiveWorkerRequest>, SpeechResultReceiver) {
    let (speech_result_sender, speech_result_receiver) = mpsc::channel(1);
    let active_generation_permit = Arc::new(Semaphore::new(1))
        .try_acquire_owned()
        .expect("the test semaphore permit should be acquirable");
    let now = tokio::time::Instant::now();
    let active_speech = ActiveSpeechGeneration {
        _active_generation_permit: active_generation_permit,
        request_id: RequestId::new(ACTIVE_REQUEST_ID),
        execution_deadline: now + Duration::from_secs(600),
        progress_stall_deadline: now + Duration::from_secs(60),
        progress_stall_timeout: Duration::from_secs(60),
        pcm16_bytes: Vec::new(),
        maximum_audio_bytes: 6,
        has_exceeded_audio_limit: false,
        completed_segments: 0,
        terminal_received_at: None,
        speech_result_sender,
        terminal_outcome: None,
    };
    (
        Some(ActiveWorkerRequest::Speech(active_speech)),
        speech_result_receiver,
    )
}

fn segment_event(completed_segments: u32, pcm16_bytes: Vec<u8>) -> WorkerEvent {
    WorkerEvent::SpeechSegmentSpoken {
        request_id: RequestId::new(ACTIVE_REQUEST_ID),
        audio_segment: SpokenAudioSegment { pcm16_bytes },
        completed_segments,
        total_segments: 2,
        was_cut_short: false,
        elapsed_millis: 10,
        mlx_memory_snapshot: None,
    }
}

fn completed_event(sample_count: u64, segment_count: u32) -> WorkerEvent {
    WorkerEvent::SpeechCompleted {
        request_id: RequestId::new(ACTIVE_REQUEST_ID),
        result_metadata: SpeechResultMetadata {
            sample_rate_hertz: 24_000,
            sample_count,
            segment_count,
            cut_short_segment_count: 0,
            voice_name: "af_heart".to_owned(),
            seed: 42,
        },
    }
}

fn finalized_event() -> WorkerEvent {
    WorkerEvent::SpeechFinalized {
        request_id: RequestId::new(ACTIVE_REQUEST_ID),
        elapsed_millis: 20,
        mlx_memory_snapshot: None,
    }
}

// Positive: two segments are joined in order and delivered once finalized,
// and the worker slot is released.
#[tokio::test]
async fn should_join_segments_and_deliver_the_audio_once_finalized() {
    let health_snapshot = speech_health_snapshot();
    let (mut active_request, mut speech_result_receiver) = active_speech_request();
    for speech_event in [
        segment_event(1, vec![1, 0, 2, 0]),
        segment_event(2, vec![3, 0]),
        completed_event(3, 2),
        finalized_event(),
    ] {
        handle_worker_speech_event(speech_event, &health_snapshot, &mut active_request)
            .expect("a well-ordered speech event");
    }
    assert!(active_request.is_none());
    let speech_output = speech_result_receiver
        .try_recv()
        .expect("the caller receives the result")
        .expect("the speech succeeded");
    assert_eq!(speech_output.pcm16_bytes, [1, 0, 2, 0, 3, 0]);
    assert_eq!(speech_output.result_metadata.seed, 42);
}

// Negative: a failure is delivered as the worker's failure.
#[tokio::test]
async fn should_deliver_a_worker_failure_after_finalization() {
    let health_snapshot = speech_health_snapshot();
    let (mut active_request, mut speech_result_receiver) = active_speech_request();
    for speech_event in [
        WorkerEvent::SpeechFailed {
            request_id: RequestId::new(ACTIVE_REQUEST_ID),
            reason: SpeechFailure::invalid_request("there is no voice \"alloy\""),
        },
        finalized_event(),
    ] {
        handle_worker_speech_event(speech_event, &health_snapshot, &mut active_request)
            .expect("a well-ordered speech event");
    }
    assert!(matches!(
        speech_result_receiver.try_recv(),
        Ok(Err(SpeechExecutionError::WorkerFailure(
            SpeechFailure::InvalidRequest { .. }
        )))
    ));
}

// Negative: a skipped segment, half a sample, or a sample count that does
// not match what was collected is a protocol violation.
#[tokio::test]
async fn should_refuse_segments_out_of_order_or_a_mismatched_result() {
    let health_snapshot = speech_health_snapshot();
    let (mut skipped, _skipped_receiver) = active_speech_request();
    assert!(
        handle_worker_speech_event(segment_event(2, vec![1, 0]), &health_snapshot, &mut skipped)
            .is_err()
    );
    let (mut partial, _partial_receiver) = active_speech_request();
    assert!(
        handle_worker_speech_event(segment_event(1, vec![1]), &health_snapshot, &mut partial)
            .is_err()
    );
    let (mut mismatched, _mismatched_receiver) = active_speech_request();
    handle_worker_speech_event(
        segment_event(1, vec![1, 0]),
        &health_snapshot,
        &mut mismatched,
    )
    .expect("one whole sample");
    assert!(
        handle_worker_speech_event(completed_event(5, 1), &health_snapshot, &mut mismatched)
            .is_err()
    );
}

// Edge: finalization before any outcome, or an event for another request,
// is a protocol violation.
#[tokio::test]
async fn should_refuse_an_early_finalization_or_another_requests_event() {
    let health_snapshot = speech_health_snapshot();
    let (mut early, _early_receiver) = active_speech_request();
    assert!(handle_worker_speech_event(finalized_event(), &health_snapshot, &mut early).is_err());
    let (mut other, _other_receiver) = active_speech_request();
    let other_request_event = WorkerEvent::SpeechFailed {
        request_id: RequestId::new(ACTIVE_REQUEST_ID + 1),
        reason: SpeechFailure::Cancelled,
    };
    assert!(handle_worker_speech_event(other_request_event, &health_snapshot, &mut other).is_err());
}

// Edge: audio past the request's bound fails the request, not the worker:
// the stream stays valid and the caller is told the text was too long.
#[tokio::test]
async fn should_fail_only_the_request_when_its_audio_runs_past_the_bound() {
    let health_snapshot = speech_health_snapshot();
    let (mut active_request, mut speech_result_receiver) = active_speech_request();
    for speech_event in [
        segment_event(1, vec![1, 0, 2, 0]),
        segment_event(2, vec![3, 0, 4, 0]),
        completed_event(4, 2),
        finalized_event(),
    ] {
        handle_worker_speech_event(speech_event, &health_snapshot, &mut active_request)
            .expect("an over-long request is still a well-formed stream");
    }
    assert!(matches!(
        speech_result_receiver.try_recv(),
        Ok(Err(SpeechExecutionError::WorkerFailure(
            SpeechFailure::InvalidRequest { .. }
        )))
    ));
}
