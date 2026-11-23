use std::sync::Arc;
use std::time::Duration;

use ipc_protocol::RequestId;
use tokio::sync::{Semaphore, mpsc};
use tokio::time::Instant;

use super::{
    MAXIMUM_AUDIO_BYTES_AT_NATURAL_PACE, SpeechBounds, active_speech_generation,
    maximum_audio_bytes_for,
};
use crate::SpeechTimeouts;
use crate::worker::request_kind_admission::AdmittedRequest;

const EXECUTION_TIMEOUT: Duration = Duration::from_secs(600);
const PROGRESS_STALL_TIMEOUT: Duration = Duration::from_secs(90);

fn admitted_speech_at(
    execution_started_at: Instant,
) -> AdmittedRequest<Result<crate::SpeechOutput, crate::SpeechExecutionError>> {
    let (result_sender, _result_receiver) = mpsc::channel(1);
    AdmittedRequest {
        active_generation_permit: Arc::new(Semaphore::new(1))
            .try_acquire_owned()
            .expect("the test semaphore permit should be acquirable"),
        request_id: RequestId::new(41),
        execution_started_at,
        result_sender,
    }
}

fn speech_bounds_at_speed(speed: f32) -> SpeechBounds {
    SpeechBounds {
        speech_timeouts: SpeechTimeouts {
            execution_timeout: EXECUTION_TIMEOUT,
            progress_stall_timeout: PROGRESS_STALL_TIMEOUT,
        },
        maximum_audio_bytes: maximum_audio_bytes_for(speed),
    }
}

// Positive: a started speech request is bounded from its start by both
// timeouts, owns its request id, and starts with no audio and no segment.
#[test]
fn should_bound_a_started_speech_request_from_its_start() {
    let execution_started_at = Instant::now();

    let active_speech = active_speech_generation(
        admitted_speech_at(execution_started_at),
        speech_bounds_at_speed(1.0),
    );

    assert_eq!(active_speech.request_id, RequestId::new(41));
    assert_eq!(
        active_speech.execution_deadline,
        execution_started_at + EXECUTION_TIMEOUT
    );
    assert_eq!(
        active_speech.progress_stall_deadline,
        execution_started_at + PROGRESS_STALL_TIMEOUT
    );
    assert_eq!(active_speech.progress_stall_timeout, PROGRESS_STALL_TIMEOUT);
    assert_eq!(
        active_speech.maximum_audio_bytes,
        MAXIMUM_AUDIO_BYTES_AT_NATURAL_PACE
    );
    assert!(active_speech.pcm16_bytes.is_empty());
    assert_eq!(active_speech.completed_segments, 0);
    assert!(!active_speech.has_exceeded_audio_limit);
    assert!(active_speech.terminal_outcome.is_none());
}

// Edge: the slowest pace carries its larger audio bound into the request.
#[test]
fn should_carry_the_slowest_pace_audio_bound_into_the_started_request() {
    let active_speech = active_speech_generation(
        admitted_speech_at(Instant::now()),
        speech_bounds_at_speed(0.25),
    );

    assert_eq!(
        active_speech.maximum_audio_bytes,
        MAXIMUM_AUDIO_BYTES_AT_NATURAL_PACE * 4
    );
}

// Positive: at natural pace or faster, the bound is the natural one.
#[test]
fn should_keep_the_natural_bound_at_natural_pace_or_faster() {
    assert_eq!(
        maximum_audio_bytes_for(1.0),
        MAXIMUM_AUDIO_BYTES_AT_NATURAL_PACE
    );
    assert_eq!(
        maximum_audio_bytes_for(4.0),
        MAXIMUM_AUDIO_BYTES_AT_NATURAL_PACE
    );
}

// Edge: the slowest pace stretches the same text four times.
#[test]
fn should_grow_the_bound_for_the_slowest_pace() {
    assert_eq!(
        maximum_audio_bytes_for(0.25),
        MAXIMUM_AUDIO_BYTES_AT_NATURAL_PACE * 4
    );
}

// Negative: a pace the contract would refuse never raises the bound past
// the slowest pace's, and NaN keeps the natural one.
#[test]
fn should_cap_the_bound_for_a_pace_the_contract_refuses() {
    assert_eq!(
        maximum_audio_bytes_for(0.0),
        MAXIMUM_AUDIO_BYTES_AT_NATURAL_PACE * 4
    );
    assert_eq!(
        maximum_audio_bytes_for(f32::NAN),
        MAXIMUM_AUDIO_BYTES_AT_NATURAL_PACE
    );
}
