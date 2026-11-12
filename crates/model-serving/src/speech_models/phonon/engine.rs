//! Phonon as a [`SpeechEngine`]: text is cut into chunks of at most 50
//! tokens at sentence ends, and each step speaks one chunk from the flow
//! model's state after reading the voice.

use std::collections::HashMap;
use std::path::PathBuf;

use astronomical_runtime_integration::MlxRuntime;
use astronomical_speech_contract::{SpeechFailure, SpeechRequest};

use super::artifact_files::{PHONON_WEIGHTS_FILE_NAME, voice_file_path};
use super::attention_history::AttentionHistory;
use super::speech_model::{PHONON_DEFAULT_TEMPERATURE, PhononSampling, PhononSpeechModel};
use super::text_chunks::TextChunk;
use super::voice::PhononVoice;
use crate::memory::telemetry::MlxMemoryTelemetry;
use crate::speech_engine::{SpeechEngine, SpeechEngineLoadResult, SpeechEngineStep, StartedSpeech};
use crate::speech_models::checkpoint_tensors::weights_file_bytes;
use crate::speech_models::engine_settings::SpeechEngineSettings;
use crate::speech_models::speech_memory::SpeechMemoryAccount;
use crate::speech_models::speech_request_slot::{
    SegmentSpeaker, SpeechRequestSlot, SpokenSegment, ends_request, voice_for_checked_request,
};

pub struct PhononSpeechEngine {
    settings: SpeechEngineSettings,
    loaded: Option<LoadedPhonon>,
    request_slot: SpeechRequestSlot<TextChunk, PhononRequestContext>,
}

struct LoadedPhonon {
    model_directory: PathBuf,
    runtime: MlxRuntime,
    model: PhononSpeechModel,
    load_result: SpeechEngineLoadResult,
    memory_account: SpeechMemoryAccount,
    /// The flow model's state after reading each voice used so far. Reading
    /// a voice runs the backbone over about 125 positions; a cached state is
    /// branched per request instead, which copies only array handles.
    voice_histories_by_name: HashMap<String, AttentionHistory>,
}

/// What every chunk of one request is spoken from.
struct PhononRequestContext {
    voice_history: AttentionHistory,
    sampling: PhononSampling,
}

impl PhononSpeechEngine {
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

    fn loaded_mut(&mut self) -> Result<&mut LoadedPhonon, SpeechFailure> {
        self.loaded
            .as_mut()
            .ok_or_else(|| SpeechFailure::fatal_execution("the Phonon engine is not loaded"))
    }
}

impl SpeechEngine for PhononSpeechEngine {
    fn load(&mut self) -> Result<SpeechEngineLoadResult, SpeechFailure> {
        let runtime = self.settings.initialize_runtime()?;
        let model_directory = self.settings.model_directory.clone();
        let model = PhononSpeechModel::load(&runtime, &model_directory)?;
        let load_result = self.settings.load_result(
            astronomical_config::phonon_speech_capabilities(&model_directory),
            model.sample_rate_hertz(),
        )?;
        let memory_account = self.settings.memory_account(weights_file_bytes(
            &model_directory.join(PHONON_WEIGHTS_FILE_NAME),
        ));
        self.loaded = Some(LoadedPhonon {
            model_directory,
            runtime,
            model,
            load_result: load_result.clone(),
            memory_account,
            voice_histories_by_name: HashMap::new(),
        });
        Ok(load_result)
    }

    fn start_speech(
        &mut self,
        speech_request: SpeechRequest,
    ) -> Result<StartedSpeech, SpeechFailure> {
        self.request_slot.ensure_idle()?;
        let loaded = self.loaded_mut()?;
        let voice_name = voice_for_checked_request(&speech_request, &loaded.load_result)?;
        let chunks = loaded.model.prompt_chunks(&speech_request.text)?;
        let request_context = PhononRequestContext {
            voice_history: loaded.voice_history(&voice_name)?,
            sampling: PhononSampling {
                temperature: PHONON_DEFAULT_TEMPERATURE,
                seed: speech_request.seed,
            },
        };
        let started_speech = StartedSpeech {
            voice_name,
            sample_rate_hertz: loaded.load_result.sample_rate_hertz,
            total_segments: chunks.len(),
        };
        self.request_slot.start(chunks, request_context)?;
        Ok(started_speech)
    }

    fn advance_speech(&mut self) -> Result<SpeechEngineStep, SpeechFailure> {
        let Some(loaded) = self.loaded.as_mut() else {
            return Err(SpeechFailure::fatal_execution(
                "the Phonon engine is not loaded",
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

impl LoadedPhonon {
    /// A branch of the state after reading the voice, read once per voice.
    fn voice_history(&mut self, voice_name: &str) -> Result<AttentionHistory, SpeechFailure> {
        if !self.voice_histories_by_name.contains_key(voice_name) {
            let voice = PhononVoice::load(
                &self.runtime,
                &voice_file_path(&self.model_directory, voice_name)?,
            )?;
            let voice_history = self.model.voice_history(&self.runtime, &voice)?;
            self.voice_histories_by_name
                .insert(voice_name.to_owned(), voice_history);
        }
        Ok(self.voice_histories_by_name[voice_name].branch()?)
    }
}

impl SegmentSpeaker<TextChunk, PhononRequestContext> for LoadedPhonon {
    fn speak_segment(
        &mut self,
        chunk: &TextChunk,
        segment_index: usize,
        request_context: &PhononRequestContext,
    ) -> Result<SpokenSegment, SpeechFailure> {
        let (samples, chunk_report) = self.model.speak_chunk(
            &self.runtime,
            &request_context.voice_history,
            chunk,
            segment_index,
            request_context.sampling,
        )?;
        Ok(SpokenSegment {
            samples,
            was_cut_short: !chunk_report.was_ended_by_end_of_speech,
        })
    }
}
