//! Phonon as a speech family: text is cut into chunks of at most 50 tokens
//! at sentence ends, and each step speaks one chunk from the flow model's
//! state after reading the voice. The engine around it is the shared
//! [`SpeechEngineShell`].

use std::collections::HashMap;
use std::path::Path;

use runtime_integration::MlxRuntime;
use speech_contract::{SpeechFailure, SpeechRequest};

use super::artifact_files::{PHONON_WEIGHTS_FILE_NAME, voice_file_path};
use super::attention_history::AttentionHistory;
use super::speech_model::{PHONON_DEFAULT_TEMPERATURE, PhononSampling, PhononSpeechModel};
use super::text_chunks::TextChunk;
use super::voice::PhononVoice;
use speech::speech_models::speech_engine_shell::{
    FamilyLoad, LoadedSpeech, PlannedSpeech, SpeechEngineShell, SpeechFamily,
};
use speech::speech_models::speech_request_slot::{SegmentSpeaker, SpokenSegment};

/// Phonon behind the shared speech engine.
pub type PhononSpeechEngine = SpeechEngineShell<PhononSpeechFamily>;

pub struct PhononSpeechFamily;

pub struct PhononParts {
    model: PhononSpeechModel,
    /// The flow model's state after reading each voice used so far. Reading
    /// a voice runs the backbone over about 125 positions; a memoized state is
    /// branched per request instead, which copies only array handles.
    voice_histories_by_name: HashMap<String, AttentionHistory>,
}

/// What every chunk of one request is spoken from.
pub struct PhononRequestContext {
    voice_history: AttentionHistory,
    sampling: PhononSampling,
}

impl SpeechFamily for PhononSpeechFamily {
    type FamilyParts = PhononParts;
    type Segment = TextChunk;
    type RequestContext = PhononRequestContext;
    const FAMILY_NAME: &'static str = "Phonon";
    const WEIGHTS_FILE_NAME: &'static str = PHONON_WEIGHTS_FILE_NAME;

    fn load_family(
        runtime: &MlxRuntime,
        model_directory: &Path,
    ) -> Result<FamilyLoad<PhononParts>, SpeechFailure> {
        let model = PhononSpeechModel::load(runtime, model_directory)?;
        let model_sample_rate_hertz = model.sample_rate_hertz();
        Ok(FamilyLoad {
            family_parts: PhononParts {
                model,
                voice_histories_by_name: HashMap::new(),
            },
            speech_capabilities: config::phonon_speech_capabilities(model_directory),
            model_sample_rate_hertz,
        })
    }

    fn plan_speech(
        loaded: &mut LoadedSpeech<PhononParts>,
        speech_request: &SpeechRequest,
        voice_name: &str,
    ) -> Result<PlannedSpeech<TextChunk, PhononRequestContext>, SpeechFailure> {
        let chunks = loaded
            .family_parts
            .model
            .prompt_chunks(&speech_request.text)?;
        Ok(PlannedSpeech {
            segments: chunks,
            request_context: PhononRequestContext {
                voice_history: voice_history_branch(loaded, voice_name)?,
                sampling: PhononSampling {
                    temperature: PHONON_DEFAULT_TEMPERATURE,
                    seed: speech_request.seed,
                },
            },
        })
    }
}

/// A branch of the state after reading the voice, read once per voice.
fn voice_history_branch(
    loaded: &mut LoadedSpeech<PhononParts>,
    voice_name: &str,
) -> Result<AttentionHistory, SpeechFailure> {
    if !loaded
        .family_parts
        .voice_histories_by_name
        .contains_key(voice_name)
    {
        let voice = PhononVoice::load(
            &loaded.runtime,
            &voice_file_path(&loaded.model_directory, voice_name)?,
        )?;
        let voice_history = loaded
            .family_parts
            .model
            .voice_history(&loaded.runtime, &voice)?;
        loaded
            .family_parts
            .voice_histories_by_name
            .insert(voice_name.to_owned(), voice_history);
    }
    Ok(loaded.family_parts.voice_histories_by_name[voice_name].branch()?)
}

impl SegmentSpeaker<TextChunk, PhononRequestContext> for LoadedSpeech<PhononParts> {
    fn speak_segment(
        &mut self,
        chunk: &TextChunk,
        segment_index: usize,
        request_context: &PhononRequestContext,
    ) -> Result<SpokenSegment, SpeechFailure> {
        let (samples, chunk_report) = self.family_parts.model.speak_chunk(
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
