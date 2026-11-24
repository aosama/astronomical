//! Kokoro as a speech family: text is phonemized and cut into passes of at
//! most 510 phonemes at sentence ends, and each step speaks one pass. The
//! engine around it is the shared [`SpeechEngineShell`].

use std::collections::HashMap;
use std::path::Path;

use english_phonemes::{english_phoneme_chunks, prepare_english_lexicon};
use runtime_integration::MlxRuntime;
use speech_contract::{SpeechFailure, SpeechRequest};

use super::artifact_files::{KOKORO_WEIGHTS_FILE_NAME, voice_file_path};
use super::reference_trace::KokoroSpeechTrace;
use super::speech_model::{KOKORO_SAMPLE_RATE_HERTZ, KokoroSpeechModel};
use super::utterance::{KOKORO_MAXIMUM_PHONEMES, KokoroUtterance};
use super::voice_pack::KokoroVoice;
use speech::speech_models::speech_engine_shell::{
    FamilyLoad, LoadedSpeech, PlannedSpeech, SpeechEngineShell, SpeechFamily,
};
use speech::speech_models::speech_request_slot::{SegmentSpeaker, SpokenSegment};

/// Kokoro behind the shared speech engine; each segment is one pass's
/// phonemes.
pub type KokoroSpeechEngine = SpeechEngineShell<KokoroSpeechFamily>;

pub struct KokoroSpeechFamily;

pub struct KokoroParts {
    model: KokoroSpeechModel,
    /// Voices are a few hundred kilobytes; each is read once, on first use.
    voices_by_name: HashMap<String, KokoroVoice>,
}

/// What every pass of one request is spoken with.
pub struct KokoroRequestContext {
    voice_name: String,
    speed: f32,
    noise_seed: u64,
}

impl SpeechFamily for KokoroSpeechFamily {
    type FamilyParts = KokoroParts;
    type Segment = String;
    type RequestContext = KokoroRequestContext;
    const FAMILY_NAME: &'static str = "Kokoro";
    const WEIGHTS_FILE_NAME: &'static str = KOKORO_WEIGHTS_FILE_NAME;

    fn load_family(
        runtime: &MlxRuntime,
        model_directory: &Path,
    ) -> Result<FamilyLoad<KokoroParts>, SpeechFailure> {
        let model = KokoroSpeechModel::load(runtime, model_directory)?;
        prepare_english_lexicon();
        Ok(FamilyLoad {
            family_parts: KokoroParts {
                model,
                voices_by_name: HashMap::new(),
            },
            speech_capabilities: config::kokoro_speech_capabilities(model_directory),
            model_sample_rate_hertz: KOKORO_SAMPLE_RATE_HERTZ,
        })
    }

    fn plan_speech(
        _loaded: &mut LoadedSpeech<KokoroParts>,
        speech_request: &SpeechRequest,
        voice_name: &str,
    ) -> Result<PlannedSpeech<String, KokoroRequestContext>, SpeechFailure> {
        Ok(PlannedSpeech {
            segments: english_phoneme_chunks(&speech_request.text, KOKORO_MAXIMUM_PHONEMES),
            request_context: KokoroRequestContext {
                voice_name: voice_name.to_owned(),
                speed: speech_request.speed,
                noise_seed: speech_request.seed,
            },
        })
    }
}

/// Speaks `phonemes` without excitation noise and returns the durations,
/// curves and samples, so a journey can hold them against Kokoro's PyTorch
/// reference. Serving never calls it. A trait because the engine is the
/// speech shell's type, which only the shell's crate may add methods to.
#[doc(hidden)]
pub trait KokoroSpeechTracing {
    fn trace_phonemes(
        &mut self,
        phonemes: &str,
        voice_name: &str,
        speed: f32,
    ) -> Result<KokoroSpeechTrace, SpeechFailure>;
}

impl KokoroSpeechTracing for KokoroSpeechEngine {
    fn trace_phonemes(
        &mut self,
        phonemes: &str,
        voice_name: &str,
        speed: f32,
    ) -> Result<KokoroSpeechTrace, SpeechFailure> {
        let loaded = self.loaded_mut()?;
        ensure_voice_loaded(loaded, voice_name)?;
        let utterance = KokoroUtterance {
            phonemes,
            voice: &loaded.family_parts.voices_by_name[voice_name],
            speed,
            noise_seed: None,
        };
        let speech_trace = loaded
            .family_parts
            .model
            .forward(&loaded.runtime, &utterance)
            .and_then(|forward_pass| KokoroSpeechTrace::read(&loaded.runtime, &forward_pass));
        // Cleaned up on failure too, so a failed trace leaves no arrays behind.
        loaded.clean_up_after_request();
        Ok(speech_trace?)
    }
}

/// Reads a voice the first time a request asks for it.
fn ensure_voice_loaded(
    loaded: &mut LoadedSpeech<KokoroParts>,
    voice_name: &str,
) -> Result<(), SpeechFailure> {
    if !loaded.family_parts.voices_by_name.contains_key(voice_name) {
        let voice = KokoroVoice::load(
            &loaded.runtime,
            &voice_file_path(&loaded.model_directory, voice_name)?,
        )?;
        loaded
            .family_parts
            .voices_by_name
            .insert(voice_name.to_owned(), voice);
    }
    Ok(())
}

impl SegmentSpeaker<String, KokoroRequestContext> for LoadedSpeech<KokoroParts> {
    /// Pass `n` of a request seeds its noise with `seed + n`.
    fn speak_segment(
        &mut self,
        phonemes: &String,
        segment_index: usize,
        request_context: &KokoroRequestContext,
    ) -> Result<SpokenSegment, SpeechFailure> {
        let voice_name = &request_context.voice_name;
        ensure_voice_loaded(self, voice_name)?;
        let utterance = KokoroUtterance {
            phonemes,
            voice: &self.family_parts.voices_by_name[voice_name],
            speed: request_context.speed,
            noise_seed: Some(
                request_context
                    .noise_seed
                    .wrapping_add(segment_index as u64),
            ),
        };
        let samples = self.family_parts.model.speak(&self.runtime, &utterance)?;
        Ok(SpokenSegment {
            samples,
            was_cut_short: false,
        })
    }
}
