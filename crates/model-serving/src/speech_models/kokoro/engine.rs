//! Kokoro as a [`SpeechEngine`]: text is phonemized and cut into passes of
//! at most 510 phonemes at sentence ends, and each step speaks one pass.

use std::collections::HashMap;
use std::path::PathBuf;

use astronomical_english_phonemes::{english_phoneme_chunks, prepare_english_lexicon};
use astronomical_runtime_integration::{MlxMemoryLimits, MlxRuntime};
use astronomical_speech_contract::{SpeechFailure, SpeechRequest};

use super::artifact_files::{voice_file_path, voice_names};
use super::speech_model::{KOKORO_SAMPLE_RATE_HERTZ, KokoroSpeechModel};
use super::utterance::{KOKORO_MAXIMUM_PHONEMES, KokoroUtterance};
use super::voice_pack::KokoroVoice;
use crate::speech_engine::{SpeechEngine, SpeechEngineLoadResult, SpeechEngineStep};
use crate::speech_models::speech_request_slot::{
    SegmentSpeaker, SpeechRequestSlot, SpokenSegment, default_voice_name, voice_for_checked_request,
};

/// The voice listeners rated highest among Kokoro's, the default when the
/// artifact has it.
const PREFERRED_DEFAULT_VOICE: &str = "af_heart";
/// The noise seed when a request names none, so the same text sounds the same.
const DEFAULT_NOISE_SEED: u64 = 0;

pub struct KokoroSpeechEngine {
    model_directory: PathBuf,
    memory_limits: MlxMemoryLimits,
    loaded: Option<LoadedKokoro>,
    /// Each segment is one pass's phonemes.
    request_slot: SpeechRequestSlot<String, KokoroRequestContext>,
}

struct LoadedKokoro {
    model_directory: PathBuf,
    runtime: MlxRuntime,
    model: KokoroSpeechModel,
    load_result: SpeechEngineLoadResult,
    /// Voices are a few hundred kilobytes; each is read once, on first use.
    voices_by_name: HashMap<String, KokoroVoice>,
}

/// What every pass of one request is spoken with.
struct KokoroRequestContext {
    voice_name: String,
    speed: f32,
    noise_seed: u64,
}

impl KokoroSpeechEngine {
    pub fn new(model_directory: PathBuf, memory_limits: MlxMemoryLimits) -> Self {
        Self {
            model_directory,
            memory_limits,
            loaded: None,
            request_slot: SpeechRequestSlot::empty(),
        }
    }

    fn loaded_mut(&mut self) -> Result<&mut LoadedKokoro, SpeechFailure> {
        self.loaded
            .as_mut()
            .ok_or_else(|| SpeechFailure::fatal_execution("the Kokoro engine is not loaded"))
    }
}

impl SpeechEngine for KokoroSpeechEngine {
    fn load(&mut self) -> Result<SpeechEngineLoadResult, SpeechFailure> {
        let runtime = MlxRuntime::initialize(self.memory_limits)
            .map_err(|runtime_error| SpeechFailure::fatal_execution(runtime_error.to_string()))?;
        let model = KokoroSpeechModel::load(&runtime, &self.model_directory)?;
        prepare_english_lexicon();
        let voice_names = voice_names(&self.model_directory);
        let load_result = SpeechEngineLoadResult {
            sample_rate_hertz: KOKORO_SAMPLE_RATE_HERTZ,
            default_voice_name: default_voice_name(&voice_names, PREFERRED_DEFAULT_VOICE)?,
            voice_names,
            supports_speed: true,
        };
        self.loaded = Some(LoadedKokoro {
            model_directory: self.model_directory.clone(),
            runtime,
            model,
            load_result: load_result.clone(),
            voices_by_name: HashMap::new(),
        });
        Ok(load_result)
    }

    fn start_speech(&mut self, speech_request: SpeechRequest) -> Result<(), SpeechFailure> {
        self.request_slot.ensure_idle()?;
        let voice_name =
            voice_for_checked_request(&speech_request, &self.loaded_mut()?.load_result)?;
        let phoneme_chunks = english_phoneme_chunks(&speech_request.text, KOKORO_MAXIMUM_PHONEMES);
        let request_context = KokoroRequestContext {
            voice_name,
            speed: speech_request.speed,
            noise_seed: speech_request.seed.unwrap_or(DEFAULT_NOISE_SEED),
        };
        self.request_slot.start(phoneme_chunks, request_context)
    }

    fn advance_speech(&mut self) -> Result<SpeechEngineStep, SpeechFailure> {
        let Some(loaded) = self.loaded.as_mut() else {
            return Err(SpeechFailure::fatal_execution(
                "the Kokoro engine is not loaded",
            ));
        };
        self.request_slot.advance(loaded)
    }

    fn cancel_speech(&mut self) {
        self.request_slot.cancel();
    }
}

impl SegmentSpeaker<String, KokoroRequestContext> for LoadedKokoro {
    /// Pass `n` of a request seeds its noise with `seed + n`.
    fn speak_segment(
        &mut self,
        phonemes: &String,
        segment_index: usize,
        request_context: &KokoroRequestContext,
    ) -> Result<SpokenSegment, SpeechFailure> {
        let voice_name = &request_context.voice_name;
        if !self.voices_by_name.contains_key(voice_name) {
            let voice = KokoroVoice::load(
                &self.runtime,
                &voice_file_path(&self.model_directory, voice_name)?,
            )?;
            self.voices_by_name.insert(voice_name.clone(), voice);
        }
        let utterance = KokoroUtterance {
            phonemes,
            voice: &self.voices_by_name[voice_name],
            speed: request_context.speed,
            noise_seed: request_context
                .noise_seed
                .wrapping_add(segment_index as u64),
        };
        let samples = self.model.speak(&self.runtime, &utterance)?;
        Ok(SpokenSegment {
            samples,
            was_cut_short: false,
        })
    }
}
