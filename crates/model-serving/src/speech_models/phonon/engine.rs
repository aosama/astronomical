//! Phonon as a [`SpeechEngine`]: text is cut into chunks of at most 50
//! tokens at sentence ends, and each step speaks one chunk from the flow
//! model's state after reading the voice.

use std::collections::HashMap;
use std::path::PathBuf;

use astronomical_runtime_integration::{MlxMemoryLimits, MlxRuntime};
use astronomical_speech_contract::{SpeechFailure, SpeechRequest};

use super::artifact_files::{PHONON_DEFAULT_VOICE_NAME, voice_file_path, voice_names};
use super::attention_history::AttentionHistory;
use super::speech_model::{PHONON_DEFAULT_TEMPERATURE, PhononSampling, PhononSpeechModel};
use super::text_chunks::TextChunk;
use super::voice::PhononVoice;
use crate::speech_engine::{SpeechEngine, SpeechEngineLoadResult, SpeechEngineStep};
use crate::speech_models::speech_request_slot::{
    SegmentSpeaker, SpeechRequestSlot, SpokenSegment, default_voice_name, voice_for_checked_request,
};

/// The reference runtime's default seed, used when a request names none.
const DEFAULT_NOISE_SEED: u64 = 4_242_424_242_424_242;

pub struct PhononSpeechEngine {
    model_directory: PathBuf,
    memory_limits: MlxMemoryLimits,
    loaded: Option<LoadedPhonon>,
    request_slot: SpeechRequestSlot<TextChunk, PhononRequestContext>,
}

struct LoadedPhonon {
    model_directory: PathBuf,
    runtime: MlxRuntime,
    model: PhononSpeechModel,
    load_result: SpeechEngineLoadResult,
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
    pub fn new(model_directory: PathBuf, memory_limits: MlxMemoryLimits) -> Self {
        Self {
            model_directory,
            memory_limits,
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
        let runtime = MlxRuntime::initialize(self.memory_limits)
            .map_err(|runtime_error| SpeechFailure::fatal_execution(runtime_error.to_string()))?;
        let model = PhononSpeechModel::load(&runtime, &self.model_directory)?;
        let voice_names = voice_names(&self.model_directory);
        let load_result = SpeechEngineLoadResult {
            sample_rate_hertz: model.sample_rate_hertz(),
            default_voice_name: default_voice_name(&voice_names, PHONON_DEFAULT_VOICE_NAME)?,
            voice_names,
            // Phonon has a duration conditioner, but its effect on pace is
            // untested here, so only the natural pace is offered.
            supports_speed: false,
        };
        self.loaded = Some(LoadedPhonon {
            model_directory: self.model_directory.clone(),
            runtime,
            model,
            load_result: load_result.clone(),
            voice_histories_by_name: HashMap::new(),
        });
        Ok(load_result)
    }

    fn start_speech(&mut self, speech_request: SpeechRequest) -> Result<(), SpeechFailure> {
        self.request_slot.ensure_idle()?;
        let loaded = self.loaded_mut()?;
        let voice_name = voice_for_checked_request(&speech_request, &loaded.load_result)?;
        let chunks = loaded.model.prompt_chunks(&speech_request.text)?;
        let request_context = PhononRequestContext {
            voice_history: loaded.voice_history(&voice_name)?,
            sampling: PhononSampling {
                temperature: PHONON_DEFAULT_TEMPERATURE,
                seed: speech_request.seed.unwrap_or(DEFAULT_NOISE_SEED),
            },
        };
        self.request_slot.start(chunks, request_context)
    }

    fn advance_speech(&mut self) -> Result<SpeechEngineStep, SpeechFailure> {
        let Some(loaded) = self.loaded.as_mut() else {
            return Err(SpeechFailure::fatal_execution(
                "the Phonon engine is not loaded",
            ));
        };
        self.request_slot.advance(loaded)
    }

    fn cancel_speech(&mut self) {
        self.request_slot.cancel();
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
