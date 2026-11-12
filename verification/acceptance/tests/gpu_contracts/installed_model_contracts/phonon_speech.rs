//! Phonon 31M speaks a Shakespeare passage from the real checkpoint, through
//! the speech engine a worker uses.
//!
//! This journey compares against no recording, so it checks what
//! wrong arithmetic would break: a human pace, audio neither silent nor
//! clipped, every chunk ended by the model's own end-of-speech signal rather
//! than its frame budget, and the same seed giving the same samples. Unlike
//! Kokoro, Phonon is not held against its reference runtime: its sampled
//! latents leave little to compare without one, and what this journey and
//! the served endpoint show was judged enough for this model. With
//! `evidence_directory` set it writes the audio as a WAV file for a person
//! to listen to.
//!
//! Run it with `cargo test-phonon-speech --configuration PATH`, where the
//! configuration's `model_directory` holds Phonon's `config.json`,
//! `model.safetensors`, `tokenizer.json` and `default-voice.safetensors`.

use std::time::Instant;

use astronomical_model_serving::{PhononSpeechEngine, SpeechEngine};
use astronomical_speech_contract::SpeechRequest;
use astronomical_test_harness::test_run::TestRunContext;

use super::speech_audio_checks::{
    ROMEO_PASSAGE, TEST_ALLOCATOR_CACHE_LIMIT_BYTES, TEST_MLX_MEMORY_CEILING_BYTES,
    assert_sounds_like_the_passage, save_evidence, speak_through_engine,
};

fn romeo_request() -> SpeechRequest {
    SpeechRequest {
        text: ROMEO_PASSAGE.to_owned(),
        voice_name: None,
        speed: 1.0,
        seed: 7,
    }
}

#[test]
#[ignore = "requires the Phonon 31M model_directory in the explicit test-run configuration"]
#[serial_test::serial]
fn should_speak_a_shakespeare_passage_at_a_human_pace() {
    let context = TestRunContext::current();
    let mut engine = PhononSpeechEngine::new(
        "phonon-31m-en",
        context.required_model_directory(),
        TEST_MLX_MEMORY_CEILING_BYTES,
        TEST_ALLOCATOR_CACHE_LIMIT_BYTES,
    );
    let load_result = engine.load().expect("Phonon loads");
    assert_eq!(load_result.default_voice_name, "default");
    let speak_started = Instant::now();
    let spoken_passage = speak_through_engine(&mut engine, romeo_request());
    eprintln!(
        "phonon: {} segments in {:.2}s",
        spoken_passage.segment_count,
        speak_started.elapsed().as_secs_f32()
    );
    save_evidence(
        context.configuration.evidence_directory.as_deref(),
        "phonon_romeo_default_voice.wav",
        &spoken_passage.samples,
        load_result.sample_rate_hertz,
    );
    assert_sounds_like_the_passage(&spoken_passage.samples, load_result.sample_rate_hertz);
    assert_eq!(
        spoken_passage.cut_short_segment_count, 0,
        "every chunk ends on the model's own signal"
    );
    let repeated = speak_through_engine(&mut engine, romeo_request());
    assert_eq!(
        spoken_passage.samples, repeated.samples,
        "the same seed gives the same audio"
    );
}
