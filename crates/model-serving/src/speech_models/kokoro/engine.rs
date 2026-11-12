//! Kokoro as a [`SpeechEngine`]: text is phonemized and cut into passes of
//! at most 510 phonemes at sentence ends, and each step speaks one pass.

use std::collections::HashMap;
use std::path::PathBuf;

use astronomical_english_phonemes::{english_phoneme_chunks, prepare_english_lexicon};
use astronomical_runtime_integration::MlxRuntime;
use astronomical_speech_contract::{SpeechFailure, SpeechRequest};

use super::artifact_files::{KOKORO_WEIGHTS_FILE_NAME, voice_file_path};
use super::speech_model::{KOKORO_SAMPLE_RATE_HERTZ, KokoroSpeechModel};
use super::utterance::{KOKORO_MAXIMUM_PHONEMES, KokoroUtterance};
use super::voice_pack::KokoroVoice;
use crate::memory::telemetry::MlxMemoryTelemetry;
use crate::speech_engine::{SpeechEngine, SpeechEngineLoadResult, SpeechEngineStep, StartedSpeech};
use crate::speech_models::checkpoint_tensors::weights_file_bytes;
use crate::speech_models::engine_settings::SpeechEngineSettings;
use crate::speech_models::speech_memory::SpeechMemoryAccount;
use crate::speech_models::speech_request_slot::{
    SegmentSpeaker, SpeechRequestSlot, SpokenSegment, ends_request, voice_for_checked_request,
};

pub struct KokoroSpeechEngine {
    settings: SpeechEngineSettings,
    loaded: Option<LoadedKokoro>,
    /// Each segment is one pass's phonemes.
    request_slot: SpeechRequestSlot<String, KokoroRequestContext>,
}

struct LoadedKokoro {
    model_directory: PathBuf,
    runtime: MlxRuntime,
    model: KokoroSpeechModel,
    load_result: SpeechEngineLoadResult,
    memory_account: SpeechMemoryAccount,
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
    pub fn new(
        model_id: impl Into<String>,
        model_directory: PathBuf,
        effective_mlx_memory_ceiling_bytes: usize,
        allocator_cache_memory_limit_bytes: usize,
    ) -> Self {
        Self {
            settings: SpeechEngineSettings {
                model_id: model_id.into(),
                model_directory,
                effective_mlx_memory_ceiling_bytes,
                allocator_cache_memory_limit_bytes,
            },
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
        let runtime = self.settings.initialize_runtime()?;
        let model_directory = self.settings.model_directory.clone();
        let model = KokoroSpeechModel::load(&runtime, &model_directory)?;
        prepare_english_lexicon();
        let load_result = self.settings.load_result(
            astronomical_config::kokoro_speech_capabilities(&model_directory),
            KOKORO_SAMPLE_RATE_HERTZ,
        )?;
        let memory_account = self.settings.memory_account(weights_file_bytes(
            &model_directory.join(KOKORO_WEIGHTS_FILE_NAME),
        ));
        self.loaded = Some(LoadedKokoro {
            model_directory,
            runtime,
            model,
            load_result: load_result.clone(),
            memory_account,
            voices_by_name: HashMap::new(),
        });
        Ok(load_result)
    }

    fn start_speech(
        &mut self,
        speech_request: SpeechRequest,
    ) -> Result<StartedSpeech, SpeechFailure> {
        self.request_slot.ensure_idle()?;
        let voice_name =
            voice_for_checked_request(&speech_request, &self.loaded_mut()?.load_result)?;
        let phoneme_chunks = english_phoneme_chunks(&speech_request.text, KOKORO_MAXIMUM_PHONEMES);
        let started_speech = StartedSpeech {
            voice_name: voice_name.clone(),
            sample_rate_hertz: KOKORO_SAMPLE_RATE_HERTZ,
            total_segments: phoneme_chunks.len(),
        };
        let request_context = KokoroRequestContext {
            voice_name,
            speed: speech_request.speed,
            noise_seed: speech_request.seed,
        };
        self.request_slot.start(phoneme_chunks, request_context)?;
        Ok(started_speech)
    }

    fn advance_speech(&mut self) -> Result<SpeechEngineStep, SpeechFailure> {
        let Some(loaded) = self.loaded.as_mut() else {
            return Err(SpeechFailure::fatal_execution(
                "the Kokoro engine is not loaded",
            ));
        };
        let speech_step = self.request_slot.advance(loaded);
        if ends_request(&speech_step) {
            loaded
                .memory_account
                .clean_up_after_request(&loaded.runtime);
        }
        speech_step
    }

    fn cancel_speech(&mut self) {
        self.request_slot.cancel();
        if let Some(loaded) = self.loaded.as_mut() {
            loaded
                .memory_account
                .clean_up_after_request(&loaded.runtime);
        }
    }

    fn take_post_cleanup_memory_telemetry(&mut self) -> Option<MlxMemoryTelemetry> {
        self.loaded
            .as_mut()?
            .memory_account
            .take_post_cleanup_memory_telemetry()
    }

    fn collect_mlx_memory_telemetry(&self) -> Option<MlxMemoryTelemetry> {
        let loaded = self.loaded.as_ref()?;
        loaded.memory_account.observe(&loaded.runtime)
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
