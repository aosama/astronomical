//! What both speech contracts check about the audio a model speaks, and how
//! they save it for a person to listen to.

use std::path::Path;

use astronomical_model_serving::{SpeechEngine, SpeechEngineStep};
use astronomical_speech_contract::{SpeechRequest, wav_bytes};

pub(crate) const TEST_MLX_MEMORY_CEILING_BYTES: usize = 4 * 1024 * 1024 * 1024;
pub(crate) const TEST_ALLOCATOR_CACHE_LIMIT_BYTES: usize = 512 * 1024 * 1024;
pub(crate) const ROMEO_PASSAGE: &str = "But soft, what light through yonder window breaks? It is the east, and Juliet is the sun. \
Arise, fair sun, and kill the envious moon, who is already sick and pale with grief.";
/// The passage's 30 words take a person about 8 to 12 seconds to say.
const SHORTEST_PLAUSIBLE_SECONDS: f32 = 5.0;
const LONGEST_PLAUSIBLE_SECONDS: f32 = 16.0;

/// What one request's steps produced.
pub(crate) struct SpokenPassage {
    pub(crate) samples: Vec<f32>,
    pub(crate) segment_count: usize,
    pub(crate) cut_short_segment_count: usize,
}

/// Every step of one request, segments concatenated.
pub(crate) fn speak_through_engine(
    engine: &mut impl SpeechEngine,
    speech_request: SpeechRequest,
) -> SpokenPassage {
    engine
        .start_speech(speech_request)
        .expect("the request is accepted");
    let mut spoken_passage = SpokenPassage {
        samples: Vec::new(),
        segment_count: 0,
        cut_short_segment_count: 0,
    };
    loop {
        // Each segment's time is printed so the daemon's progress-stall
        // bound can be checked against what a real model takes.
        let segment_started = std::time::Instant::now();
        match engine.advance_speech().expect("the engine speaks") {
            SpeechEngineStep::SegmentSpoken {
                samples,
                was_cut_short,
                completed_segments,
                total_segments,
            } => {
                eprintln!(
                    "speech: segment {completed_segments}/{total_segments} spoken in {:.2}s",
                    segment_started.elapsed().as_secs_f32()
                );
                spoken_passage.samples.extend(samples);
                spoken_passage.segment_count += 1;
                spoken_passage.cut_short_segment_count += usize::from(was_cut_short);
            }
            SpeechEngineStep::Completed => return spoken_passage,
        }
    }
}

/// The passage at a human pace, neither silent nor clipped, every sample a number.
pub(crate) fn assert_sounds_like_the_passage(samples: &[f32], sample_rate_hertz: u32) {
    let audio_seconds = samples.len() as f32 / sample_rate_hertz as f32;
    let peak = samples
        .iter()
        .fold(0.0f32, |loudest, sample| loudest.max(sample.abs()));
    let root_mean_square = (samples.iter().map(|sample| sample * sample).sum::<f32>()
        / samples.len().max(1) as f32)
        .sqrt();
    eprintln!("speech: {audio_seconds:.2}s of audio, peak {peak:.3}, rms {root_mean_square:.4}");
    assert!(
        samples.iter().all(|sample| sample.is_finite()),
        "every sample is a number"
    );
    assert!(
        (SHORTEST_PLAUSIBLE_SECONDS..=LONGEST_PLAUSIBLE_SECONDS).contains(&audio_seconds),
        "{audio_seconds:.2}s is not a human pace for the passage"
    );
    // Encoding clamps to [-1, 1]; a peak far past it would mean broken arithmetic.
    assert!(
        peak > 0.05 && peak < 1.5,
        "peak {peak} is near silence or far past full scale"
    );
    assert!(
        root_mean_square > 0.01,
        "rms {root_mean_square} is near silence"
    );
}

/// Writes the audio as a WAV file when the run configuration names an evidence folder.
pub(crate) fn save_evidence(
    evidence_directory: Option<&Path>,
    file_name: &str,
    samples: &[f32],
    sample_rate_hertz: u32,
) {
    let Some(evidence_directory) = evidence_directory else {
        return;
    };
    std::fs::create_dir_all(evidence_directory).expect("the evidence directory exists");
    let wav = wav_bytes(samples, sample_rate_hertz).expect("the passage fits a WAV file");
    std::fs::write(evidence_directory.join(file_name), wav).expect("the evidence WAV is written");
    eprintln!("speech: evidence in {}", evidence_directory.display());
}
