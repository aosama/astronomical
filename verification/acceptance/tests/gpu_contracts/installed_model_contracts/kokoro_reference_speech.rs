//! Holds the MLX Kokoro-82M port against Kokoro's PyTorch reference on the
//! real checkpoint.
//!
//! The fixture (`tests/fixtures/kokoro_reference_speech.json`) holds what the
//! reference package computed in float32 on the CPU from the original
//! `hexgrad/Kokoro-82M` checkpoint, at the revision the fixture records, for
//! three Shakespeare lines in the `af_heart` voice: Misaki's phonemes, how
//! many frames the model held each token, its pitch and energy curves, and a
//! fingerprint of the audio (each 600-sample frame's level and a stretch of
//! raw samples from the middle). Both sides speak without the vocoder's
//! random excitation noise, which no two runtimes draw alike.
//!
//! This journey feeds the reference's phonemes to the port, so a phonemizer
//! difference cannot hide an arithmetic one, and checks that:
//!
//! * the same number of tokens is read and each is held for the same number
//!   of frames, which checks the text reading and the duration predictor
//!   (the fixture's token identifiers are kept for whoever regenerates it);
//! * the pitch and energy curves agree within stated tolerances, which checks
//!   the prosody predictor; and
//! * the audio has the same length and the same loudness frame by frame,
//!   which checks the decoder and the vocoder. How closely the middle
//!   stretch's samples line up is printed, not asserted (see
//!   `CaseMeasurement`).
//!
//! Run it with `cargo test-kokoro-reference-speech --configuration PATH`,
//! where the configuration's `model_directory` holds Kokoro's `config.json`,
//! `kokoro-v1_0.safetensors` and `voices/af_heart.safetensors`.

use astronomical_model_serving::{KokoroSpeechEngine, KokoroSpeechTrace, SpeechEngine};
use astronomical_test_harness::test_run::TestRunContext;
use serde_json::Value;

use super::speech_audio_checks::{TEST_ALLOCATOR_CACHE_LIMIT_BYTES, TEST_MLX_MEMORY_CEILING_BYTES};

const REFERENCE_SPEECH_JSON: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/kokoro_reference_speech.json"
));
const SAMPLES_PER_FRAME: usize = 600;
/// Frames quieter than this (about -60 dBFS) are left out of the loudness
/// comparison: in near silence a tiny absolute difference is a large ratio.
const AUDIBLE_FRAME_LEVEL: f64 = 1e-3;
// The tolerances leave room above what an Apple Silicon Mac measured when
// this journey was added: pitch within 1.38 Hz, energy within 0.044 and
// audible frames within 2.7 dB, the largest of each over the three cases.
// Both sides compute in float32 (the `Kokoro-82M-bf16` build stores float32
// tensors despite its name), so the gap is MLX on the GPU against PyTorch on
// the CPU, compounded through two recurrent layers and the pitch predictor.
/// The largest pitch difference accepted, in hertz.
const MAXIMUM_PITCH_DIFFERENCE_HERTZ: f64 = 3.0;
/// The largest energy difference accepted, in the predictor's own units.
const MAXIMUM_ENERGY_DIFFERENCE: f64 = 0.1;
/// The largest loudness difference accepted on an audible frame, in decibels.
const MAXIMUM_FRAME_LEVEL_DIFFERENCE_DECIBELS: f64 = 4.0;

/// How far the port strayed from the reference on one case.
struct CaseMeasurement {
    largest_pitch_difference_hertz: f64,
    largest_energy_difference: f64,
    largest_frame_level_difference_decibels: f64,
    /// Reported, not asserted: the excitation's phase is the running sum of
    /// the pitch, so a pitch difference well under a hertz drifts the phase
    /// further every second. The audio sounds the same and has the same
    /// loudness frame by frame, but its samples stop lining up one for one.
    excerpt_signal_to_error_decibels: f64,
}

#[test]
#[ignore = "requires the Kokoro-82M model_directory in the explicit test-run configuration"]
#[serial_test::serial]
fn should_speak_like_the_pytorch_reference_without_excitation_noise() {
    let reference_speech: Value =
        serde_json::from_str(REFERENCE_SPEECH_JSON).expect("the fixture is JSON");
    let voice_name = reference_speech["voice_name"]
        .as_str()
        .expect("a voice name");
    let speed = reference_speech["speed"].as_f64().expect("a speed") as f32;
    let context = TestRunContext::current();
    let mut engine = KokoroSpeechEngine::new(
        "Kokoro-82M-bf16",
        context.required_model_directory(),
        TEST_MLX_MEMORY_CEILING_BYTES,
        TEST_ALLOCATOR_CACHE_LIMIT_BYTES,
    );
    engine.load().expect("Kokoro loads");
    let reference_cases = reference_speech["cases"].as_array().expect("cases");
    assert!(!reference_cases.is_empty(), "the fixture has cases");
    for reference_case in reference_cases {
        let case_name = reference_case["name"].as_str().expect("a case name");
        let phonemes = reference_case["phonemes"].as_str().expect("phonemes");
        let speech_trace = engine
            .trace_phonemes(phonemes, voice_name, speed)
            .expect("the port speaks the reference phonemes");
        assert_eq!(
            speech_trace.frame_counts,
            integers(&reference_case["frame_counts"]),
            "case {case_name}: frames per token differ from the reference"
        );
        let measurement = measure_case(&speech_trace, reference_case);
        report_measurement(case_name, &measurement);
        assert_within_tolerances(case_name, &measurement);
    }
}

fn measure_case(speech_trace: &KokoroSpeechTrace, reference_case: &Value) -> CaseMeasurement {
    let reference_audio = &reference_case["audio"];
    assert_eq!(
        speech_trace.samples.len() as u64,
        reference_audio["sample_count"]
            .as_u64()
            .expect("a sample count"),
        "the audio length follows the frame counts"
    );
    let excerpt_start = reference_audio["excerpt_start"]
        .as_u64()
        .expect("an excerpt start") as usize;
    let reference_excerpt = numbers(&reference_audio["excerpt"]);
    let port_excerpt =
        &speech_trace.samples[excerpt_start..excerpt_start + reference_excerpt.len()];
    CaseMeasurement {
        largest_pitch_difference_hertz: largest_difference(
            &speech_trace.pitch_curve,
            &numbers(&reference_case["pitch_curve"]),
        ),
        largest_energy_difference: largest_difference(
            &speech_trace.energy_curve,
            &numbers(&reference_case["energy_curve"]),
        ),
        largest_frame_level_difference_decibels: largest_frame_level_difference_decibels(
            &frame_levels(&speech_trace.samples),
            &numbers(&reference_audio["frame_levels"]),
        ),
        excerpt_signal_to_error_decibels: signal_to_error_decibels(
            port_excerpt,
            &reference_excerpt,
        ),
    }
}

fn report_measurement(case_name: &str, measurement: &CaseMeasurement) {
    eprintln!(
        "kokoro reference {case_name}: pitch within {:.3} Hz, energy within {:.4}, \
         audible frames within {:.3} dB, middle stretch at {:.1} dB signal to error",
        measurement.largest_pitch_difference_hertz,
        measurement.largest_energy_difference,
        measurement.largest_frame_level_difference_decibels,
        measurement.excerpt_signal_to_error_decibels
    );
}

fn assert_within_tolerances(case_name: &str, measurement: &CaseMeasurement) {
    assert!(
        measurement.largest_pitch_difference_hertz <= MAXIMUM_PITCH_DIFFERENCE_HERTZ,
        "case {case_name}: pitch strays {:.3} Hz",
        measurement.largest_pitch_difference_hertz
    );
    assert!(
        measurement.largest_energy_difference <= MAXIMUM_ENERGY_DIFFERENCE,
        "case {case_name}: energy strays {:.4}",
        measurement.largest_energy_difference
    );
    assert!(
        measurement.largest_frame_level_difference_decibels
            <= MAXIMUM_FRAME_LEVEL_DIFFERENCE_DECIBELS,
        "case {case_name}: an audible frame's loudness strays {:.3} dB",
        measurement.largest_frame_level_difference_decibels
    );
}

fn largest_difference(port_values: &[f32], reference_values: &[f64]) -> f64 {
    assert_eq!(port_values.len(), reference_values.len(), "curve lengths");
    port_values
        .iter()
        .zip(reference_values)
        .map(|(port_value, reference_value)| (f64::from(*port_value) - reference_value).abs())
        .fold(0.0, f64::max)
}

/// The root-mean-square level of every whole 600-sample frame.
fn frame_levels(samples: &[f32]) -> Vec<f64> {
    samples
        .chunks_exact(SAMPLES_PER_FRAME)
        .map(|frame_samples| {
            let power = frame_samples
                .iter()
                .map(|sample| f64::from(*sample).powi(2))
                .sum::<f64>()
                / SAMPLES_PER_FRAME as f64;
            power.sqrt()
        })
        .collect()
}

fn largest_frame_level_difference_decibels(port_levels: &[f64], reference_levels: &[f64]) -> f64 {
    assert_eq!(port_levels.len(), reference_levels.len(), "frame counts");
    port_levels
        .iter()
        .zip(reference_levels)
        .filter(|(_, reference_level)| **reference_level >= AUDIBLE_FRAME_LEVEL)
        .map(|(port_level, reference_level)| {
            (20.0 * (port_level.max(f64::MIN_POSITIVE) / reference_level).log10()).abs()
        })
        .fold(0.0, f64::max)
}

fn signal_to_error_decibels(port_samples: &[f32], reference_samples: &[f64]) -> f64 {
    let mut reference_power = 0.0;
    let mut error_power = 0.0;
    for (port_sample, reference_sample) in port_samples.iter().zip(reference_samples) {
        reference_power += reference_sample.powi(2);
        error_power += (f64::from(*port_sample) - reference_sample).powi(2);
    }
    10.0 * (reference_power / error_power.max(f64::MIN_POSITIVE)).log10()
}

fn numbers(json_numbers: &Value) -> Vec<f64> {
    json_numbers
        .as_array()
        .expect("an array of numbers")
        .iter()
        .map(|number| number.as_f64().expect("a number"))
        .collect()
}

fn integers(json_integers: &Value) -> Vec<i32> {
    json_integers
        .as_array()
        .expect("an array of integers")
        .iter()
        .map(|integer| integer.as_i64().expect("an integer") as i32)
        .collect()
}
