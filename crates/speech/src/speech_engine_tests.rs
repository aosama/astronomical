use astronomical_speech_contract::{SpeechFailure, SpeechRequest};

use super::{SpeechEngine, SpeechUnavailableEngine};

fn romeo_request() -> SpeechRequest {
    SpeechRequest {
        text: "But soft, what light through yonder window breaks?".to_owned(),
        voice_name: None,
        speed: 1.0,
        seed: 7,
    }
}

// Negative: a worker whose model does not speak refuses every speech call
// as the server's failure, never as the caller's.
#[test]
fn should_refuse_to_load_or_speak_without_a_speech_model() {
    let mut unavailable_engine = SpeechUnavailableEngine;
    assert!(matches!(
        unavailable_engine.load(),
        Err(SpeechFailure::FatalExecution { .. })
    ));
    assert!(matches!(
        unavailable_engine.start_speech(romeo_request()),
        Err(SpeechFailure::FatalExecution { .. })
    ));
    assert!(matches!(
        unavailable_engine.advance_speech(),
        Err(SpeechFailure::FatalExecution { .. })
    ));
}

// Positive and edge: a boxed engine forwards each call, and cancelling with
// nothing to cancel is harmless.
#[test]
fn should_forward_calls_through_a_boxed_engine() {
    let mut boxed_engine: Box<dyn SpeechEngine> = Box::new(SpeechUnavailableEngine);
    boxed_engine.cancel_speech();
    assert_eq!(boxed_engine.minimum_mlx_memory_ceiling_bytes(), 1);
    assert!(boxed_engine.collect_mlx_memory_telemetry().is_none());
    assert!(boxed_engine.take_post_cleanup_memory_telemetry().is_none());
    assert!(boxed_engine.start_speech(romeo_request()).is_err());
}
