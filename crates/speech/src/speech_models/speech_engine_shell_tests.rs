//! The shell's behaviour before a model is loaded, through a family that
//! never loads one: every call that needs the model refuses with the family's
//! own wording, and the calls that only tidy up stay quiet. The loaded path is
//! covered by the Kokoro and Phonon journeys, which need real model folders.

use std::path::{Path, PathBuf};

use runtime_integration::MlxRuntime;
use speech_contract::{SpeechFailure, SpeechRequest};

use super::{
    FamilyLoad, LoadedSpeech, PlannedSpeech, SpeechEngineShell, SpeechFamily, not_loaded_failure,
};
use crate::speech_engine::SpeechEngine;
use crate::speech_models::speech_request_slot::{SegmentSpeaker, SpokenSegment};

struct UnloadableFamily;

impl SpeechFamily for UnloadableFamily {
    type FamilyParts = ();
    type Segment = String;
    type RequestContext = ();
    const FAMILY_NAME: &'static str = "Globe";
    const WEIGHTS_FILE_NAME: &'static str = "globe.safetensors";

    fn load_family(
        _runtime: &MlxRuntime,
        _model_directory: &Path,
    ) -> Result<FamilyLoad<()>, SpeechFailure> {
        Err(SpeechFailure::fatal_execution("the Globe has no folder"))
    }

    fn plan_speech(
        _loaded: &mut LoadedSpeech<()>,
        speech_request: &SpeechRequest,
        _voice_name: &str,
    ) -> Result<PlannedSpeech<String, ()>, SpeechFailure> {
        Ok(PlannedSpeech {
            segments: vec![speech_request.text.clone()],
            request_context: (),
        })
    }
}

impl SegmentSpeaker<String, ()> for LoadedSpeech<()> {
    fn speak_segment(
        &mut self,
        _segment: &String,
        _segment_index: usize,
        _request_context: &(),
    ) -> Result<SpokenSegment, SpeechFailure> {
        Ok(SpokenSegment {
            samples: Vec::new(),
            was_cut_short: false,
        })
    }
}

fn unloaded_engine() -> SpeechEngineShell<UnloadableFamily> {
    SpeechEngineShell::new("Globe-1", PathBuf::from("globe"), 1 << 30, 1 << 20)
}

fn speech_request() -> SpeechRequest {
    SpeechRequest {
        text: "All the world's a stage.".to_owned(),
        voice_name: None,
        speed: 1.0,
        seed: 7,
    }
}

#[test]
fn should_refuse_to_start_speech_before_the_model_is_loaded() {
    let mut engine = unloaded_engine();
    assert_eq!(
        engine.start_speech(speech_request()).err(),
        Some(not_loaded_failure("Globe"))
    );
}

#[test]
fn should_refuse_to_advance_speech_before_the_model_is_loaded() {
    let mut engine = unloaded_engine();
    assert_eq!(
        engine.advance_speech().err(),
        Some(not_loaded_failure("Globe"))
    );
}

#[test]
fn should_cancel_and_report_no_memory_before_the_model_is_loaded() {
    let mut engine = unloaded_engine();
    engine.cancel_speech();
    assert!(engine.take_post_cleanup_memory_telemetry().is_none());
    assert!(engine.collect_mlx_memory_telemetry().is_none());
}

#[test]
fn should_word_the_not_loaded_failure_with_the_family_name() {
    assert_eq!(
        not_loaded_failure("Kokoro"),
        SpeechFailure::fatal_execution("the Kokoro engine is not loaded")
    );
}
