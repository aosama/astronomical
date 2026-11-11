//! Kokoro-82M speaks a Shakespeare passage from the real checkpoint, through
//! the speech engine a worker uses.
//!
//! No reference recording is compared against, so the journey checks what
//! wrong arithmetic would break: a human pace, audio neither silent nor
//! clipped, whole 600-sample frames, and the same seed giving the same
//! samples. Comparing frame counts and pitch curves against Kokoro's PyTorch
//! reference is still to do. With `evidence_directory` set it writes the
//! audio as a WAV file for a person to listen to.
//!
//! Run it with `cargo test-kokoro-speech --configuration PATH`, where the
//! configuration's `model_directory` holds Kokoro's `config.json`,
//! `kokoro-v1_0.safetensors` and `voices/af_heart.safetensors`.

use std::time::Instant;

use astronomical_model_serving::{KokoroSpeechEngine, SpeechEngine};
use astronomical_speech_contract::SpeechRequest;
use astronomical_test_harness::test_run::TestRunContext;

use super::speech_audio_checks::{
    ROMEO_PASSAGE, assert_sounds_like_the_passage, save_evidence, speak_through_engine,
    test_memory_limits,
};

/// Kokoro's generator makes 600 samples for every alignment frame.
const SAMPLES_PER_FRAME: usize = 600;

fn romeo_request() -> SpeechRequest {
    SpeechRequest {
        text: ROMEO_PASSAGE.to_owned(),
        voice_name: Some("af_heart".to_owned()),
        speed: 1.0,
        seed: Some(7),
    }
}

#[test]
#[ignore = "requires the Kokoro-82M model_directory in the explicit test-run configuration"]
#[serial_test::serial]
fn should_speak_a_shakespeare_passage_at_a_human_pace() {
    let context = TestRunContext::current();
    let mut engine =
        KokoroSpeechEngine::new(context.required_model_directory(), test_memory_limits());
    let load_result = engine.load().expect("Kokoro loads");
    assert!(
        load_result.voice_names.contains(&"af_heart".to_owned()),
        "{:?}",
        load_result.voice_names
    );
    let speak_started = Instant::now();
    let spoken_passage = speak_through_engine(&mut engine, romeo_request());
    eprintln!(
        "kokoro: {} segments in {:.2}s",
        spoken_passage.segment_count,
        speak_started.elapsed().as_secs_f32()
    );
    save_evidence(
        context.configuration.evidence_directory.as_deref(),
        "kokoro_romeo_af_heart.wav",
        &spoken_passage.samples,
        load_result.sample_rate_hertz,
    );
    assert_sounds_like_the_passage(&spoken_passage.samples, load_result.sample_rate_hertz);
    assert_eq!(
        spoken_passage.samples.len() % SAMPLES_PER_FRAME,
        0,
        "whole frames only"
    );
    let repeated = speak_through_engine(&mut engine, romeo_request());
    assert_eq!(
        spoken_passage.samples, repeated.samples,
        "the same seed gives the same audio"
    );
}
