//! Correlates speech events: collects each segment's audio, keeps the stall
//! bound moving while segments arrive, and releases the request only after
//! the worker confirms its cleanup.

use std::sync::{Arc, RwLock};

use astronomical_ipc_protocol::{
    RequestId, SpeechResultMetadata, SpokenAudioSegment, WorkerEvent, WorkerMlxMemorySnapshot,
};

use crate::worker::embeddings_event::publish_single_shot_finalized_memory_snapshot;
use crate::worker::event_handler::protocol_violation;
use crate::worker::health::{
    clear_active_request_progress, publish_activity, publish_latest_mlx_memory_snapshot,
};
use crate::worker::loop_types::{ActiveSpeechGeneration, ActiveWorkerRequest};
use astronomical_speech_contract::SpeechFailure;

use crate::{SpeechExecutionError, SpeechOutput, WorkerControlError, WorkerHealthSnapshot};

/// Why a request whose audio ran past its bound was refused.
pub(crate) const AUDIO_TOO_LONG_REASON: &str =
    "the audio is longer than one request may return; send the text in shorter parts";

pub(super) fn handle_worker_speech_event(
    worker_event: WorkerEvent,
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
    active_worker_request: &mut Option<ActiveWorkerRequest>,
) -> Result<(), WorkerControlError> {
    match worker_event {
        WorkerEvent::SpeechSegmentSpoken {
            request_id,
            audio_segment,
            completed_segments,
            total_segments,
            mlx_memory_snapshot,
            ..
        } => {
            let active_speech = matching_active_speech(active_worker_request, request_id)?;
            collect_spoken_segment(
                active_speech,
                &audio_segment,
                completed_segments,
                total_segments,
            )?;
            publish_activity(health_snapshot, crate::WorkerActivity::Speaking);
            publish_segment_memory_snapshot(health_snapshot, mlx_memory_snapshot);
            Ok(())
        }
        WorkerEvent::SpeechCompleted {
            request_id,
            result_metadata,
        } => {
            let active_speech = matching_active_speech(active_worker_request, request_id)?;
            record_completed_speech(active_speech, result_metadata)
        }
        WorkerEvent::SpeechFailed { request_id, reason } => {
            let active_speech = matching_active_speech(active_worker_request, request_id)?;
            if active_speech.terminal_outcome.is_some() {
                return Err(protocol_violation("duplicate speech terminal outcome"));
            }
            active_speech.terminal_outcome = Some(Err(reason));
            active_speech.terminal_received_at = Some(tokio::time::Instant::now());
            Ok(())
        }
        WorkerEvent::SpeechFinalized {
            request_id,
            mlx_memory_snapshot,
            ..
        } => finalize_speech(
            health_snapshot,
            active_worker_request,
            request_id,
            mlx_memory_snapshot,
        ),
        _ => Err(protocol_violation(
            "non-speech event passed to speech handler",
        )),
    }
}

/// Appends one segment's audio after checking it follows the last one. Audio
/// past the request's bound is a request that is too long, not a broken
/// worker: it is dropped, and the request fails when it completes.
fn collect_spoken_segment(
    active_speech: &mut ActiveSpeechGeneration,
    audio_segment: &SpokenAudioSegment,
    completed_segments: u32,
    total_segments: u32,
) -> Result<(), WorkerControlError> {
    if active_speech.terminal_outcome.is_some()
        || !audio_segment.has_whole_samples()
        || completed_segments != active_speech.completed_segments + 1
        || completed_segments > total_segments
    {
        return Err(protocol_violation("invalid spoken speech segment"));
    }
    let collected_bytes = active_speech.pcm16_bytes.len() + audio_segment.pcm16_bytes.len();
    if collected_bytes > active_speech.maximum_audio_bytes {
        active_speech.has_exceeded_audio_limit = true;
        active_speech.pcm16_bytes = Vec::new();
    }
    if !active_speech.has_exceeded_audio_limit {
        active_speech
            .pcm16_bytes
            .extend_from_slice(&audio_segment.pcm16_bytes);
    }
    active_speech.completed_segments = completed_segments;
    active_speech.progress_stall_deadline =
        tokio::time::Instant::now() + active_speech.progress_stall_timeout;
    Ok(())
}

/// Records the whole audio as the request's result once its last segment is
/// in; the sample count the worker reports must be what was collected.
fn record_completed_speech(
    active_speech: &mut ActiveSpeechGeneration,
    result_metadata: SpeechResultMetadata,
) -> Result<(), WorkerControlError> {
    if active_speech.terminal_outcome.is_some() {
        return Err(protocol_violation("duplicate speech terminal outcome"));
    }
    if active_speech.has_exceeded_audio_limit {
        active_speech.terminal_outcome =
            Some(Err(SpeechFailure::invalid_request(AUDIO_TOO_LONG_REASON)));
        active_speech.terminal_received_at = Some(tokio::time::Instant::now());
        return Ok(());
    }
    let collected_sample_count = (active_speech.pcm16_bytes.len() / 2) as u64;
    if result_metadata.sample_count != collected_sample_count
        || result_metadata.segment_count != active_speech.completed_segments
        || result_metadata.sample_rate_hertz == 0
    {
        return Err(protocol_violation("invalid completed speech result"));
    }
    active_speech.terminal_outcome = Some(Ok(SpeechOutput {
        pcm16_bytes: std::mem::take(&mut active_speech.pcm16_bytes),
        result_metadata,
    }));
    active_speech.terminal_received_at = Some(tokio::time::Instant::now());
    Ok(())
}

/// Delivers the recorded outcome to the caller and frees the worker.
fn finalize_speech(
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
    active_worker_request: &mut Option<ActiveWorkerRequest>,
    request_id: RequestId,
    mlx_memory_snapshot: Option<WorkerMlxMemorySnapshot>,
) -> Result<(), WorkerControlError> {
    let active_speech = matching_active_speech(active_worker_request, request_id)?;
    let Some(terminal_outcome) = active_speech.terminal_outcome.take() else {
        return Err(protocol_violation(
            "speech finalized before any terminal outcome",
        ));
    };
    let result_sender = active_speech.speech_result_sender.clone();
    *active_worker_request = None;
    let client_outcome = terminal_outcome.map_err(SpeechExecutionError::WorkerFailure);
    let _send_outcome = result_sender.try_send(client_outcome);
    publish_activity(health_snapshot, crate::WorkerActivity::Idle);
    clear_active_request_progress(health_snapshot);
    // A speech engine keeps no experts or context state between requests,
    // so its cleanup observation obeys the forward-pass rules exactly.
    publish_single_shot_finalized_memory_snapshot(health_snapshot, mlx_memory_snapshot)
}

/// Mid-request samples stay unvalidated like image step samples; only the
/// finalization snapshot proves cleanup, so a segment sample never blocks.
fn publish_segment_memory_snapshot(
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
    mlx_memory_snapshot: Option<WorkerMlxMemorySnapshot>,
) {
    if let Some(mlx_memory_snapshot) = mlx_memory_snapshot {
        publish_latest_mlx_memory_snapshot(health_snapshot, mlx_memory_snapshot);
    }
}

fn matching_active_speech(
    active_worker_request: &mut Option<ActiveWorkerRequest>,
    request_id: RequestId,
) -> Result<&mut ActiveSpeechGeneration, WorkerControlError> {
    let Some(active_request) = active_worker_request else {
        return Err(protocol_violation(
            "received a speech event without an active request",
        ));
    };
    let Some(active_speech) = active_request.speech_mut() else {
        return Err(protocol_violation(
            "received a speech event while another kind of request is active",
        ));
    };
    if active_speech.request_id != request_id {
        return Err(protocol_violation(
            "received a speech event for a different request",
        ));
    }
    Ok(active_speech)
}

#[cfg(test)]
#[path = "speech_event_tests.rs"]
mod speech_event_tests;
