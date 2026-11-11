use astronomical_speech_contract::{SpeechFailure, SpeechRequest};

use super::{
    SegmentSpeaker, SpeechRequestSlot, SpokenSegment, default_voice_name, voice_for_checked_request,
};
use crate::speech_engine::{SpeechEngineLoadResult, SpeechEngineStep};

fn load_result(supports_speed: bool) -> SpeechEngineLoadResult {
    SpeechEngineLoadResult {
        sample_rate_hertz: 24_000,
        voice_names: vec!["af_heart".to_owned(), "bf_emma".to_owned()],
        default_voice_name: "af_heart".to_owned(),
        supports_speed,
    }
}

fn request(voice_name: Option<&str>, speed: f32) -> SpeechRequest {
    SpeechRequest {
        text: "Parting is such sweet sorrow.".to_owned(),
        voice_name: voice_name.map(str::to_owned),
        speed,
        seed: None,
    }
}

#[test]
fn should_use_the_default_voice_when_none_is_asked_for() {
    assert_eq!(
        voice_for_checked_request(&request(None, 1.0), &load_result(true)),
        Ok("af_heart".to_owned())
    );
    assert_eq!(
        voice_for_checked_request(&request(Some("bf_emma"), 1.2), &load_result(true)),
        Ok("bf_emma".to_owned())
    );
}

// Negative: a voice the model does not have.
#[test]
fn should_refuse_an_unknown_voice_and_name_the_known_ones() {
    let Err(SpeechFailure::InvalidRequest { reason }) =
        voice_for_checked_request(&request(Some("romeo"), 1.0), &load_result(true))
    else {
        panic!("an unknown voice is the caller's to fix");
    };
    assert!(reason.contains("af_heart, bf_emma"), "{reason}");
}

// Boundary: a model without pace control takes exactly 1.0.
#[test]
fn should_refuse_a_pace_change_the_model_cannot_make() {
    assert!(voice_for_checked_request(&request(None, 1.0), &load_result(false)).is_ok());
    assert!(voice_for_checked_request(&request(None, 1.25), &load_result(false)).is_err());
}

#[test]
fn should_apply_the_contract_checks_first() {
    let blank_request = SpeechRequest {
        text: " ".to_owned(),
        ..request(None, 1.0)
    };
    assert!(voice_for_checked_request(&blank_request, &load_result(true)).is_err());
}

/// Speaks each segment as its length in samples; fails on the segment named "fail".
struct CountingSpeaker;

impl SegmentSpeaker<&'static str, ()> for CountingSpeaker {
    fn speak_segment(
        &mut self,
        segment: &&'static str,
        _segment_index: usize,
        _request_context: &(),
    ) -> Result<SpokenSegment, SpeechFailure> {
        if *segment == "fail" {
            return Err(SpeechFailure::fatal_execution("the segment failed"));
        }
        Ok(SpokenSegment {
            samples: vec![0.0; segment.len()],
            was_cut_short: false,
        })
    }
}

#[test]
fn should_speak_each_segment_then_complete_and_free_the_slot() {
    let mut slot = SpeechRequestSlot::empty();
    slot.start(vec!["But soft", "what light"], ())
        .expect("starts");
    let first = slot.advance(&mut CountingSpeaker).expect("first segment");
    assert!(matches!(
        first,
        SpeechEngineStep::SegmentSpoken {
            completed_segments: 1,
            total_segments: 2,
            ..
        }
    ));
    slot.advance(&mut CountingSpeaker).expect("second segment");
    assert_eq!(
        slot.advance(&mut CountingSpeaker),
        Ok(SpeechEngineStep::Completed)
    );
    assert!(
        slot.ensure_idle().is_ok(),
        "a completed request frees the engine"
    );
}

#[test]
fn should_refuse_a_second_request_while_one_is_in_progress() {
    let mut slot = SpeechRequestSlot::empty();
    slot.start(vec!["Romeo"], ()).expect("starts");
    assert_eq!(
        slot.start(vec!["Juliet"], ()),
        Err(SpeechFailure::EngineBusy)
    );
}

// Negative: advancing with nothing started is the worker's sequencing error.
#[test]
fn should_report_an_advance_with_no_request_as_the_servers_error() {
    let mut slot: SpeechRequestSlot<&'static str, ()> = SpeechRequestSlot::empty();
    assert!(matches!(
        slot.advance(&mut CountingSpeaker),
        Err(SpeechFailure::FatalExecution { .. })
    ));
}

// Edge: a failed segment ends its request and frees the engine.
#[test]
fn should_end_the_request_when_a_segment_fails() {
    let mut slot = SpeechRequestSlot::empty();
    slot.start(vec!["fail", "never spoken"], ())
        .expect("starts");
    assert!(slot.advance(&mut CountingSpeaker).is_err());
    assert!(slot.ensure_idle().is_ok());
}

#[test]
fn should_free_the_slot_on_cancel_and_refuse_nothing_to_speak() {
    let mut slot = SpeechRequestSlot::empty();
    slot.start(vec!["Arise, fair sun"], ()).expect("starts");
    slot.cancel();
    assert!(slot.ensure_idle().is_ok());
    assert!(slot.start(Vec::new(), ()).is_err());
}

#[test]
fn should_prefer_the_named_default_voice_and_fall_back_to_the_first() {
    let voices = vec!["am_adam".to_owned(), "af_heart".to_owned()];
    assert_eq!(
        default_voice_name(&voices, "af_heart"),
        Ok("af_heart".to_owned())
    );
    assert_eq!(
        default_voice_name(&voices, "missing"),
        Ok("am_adam".to_owned())
    );
    assert!(default_voice_name(&[], "af_heart").is_err());
}
