//! What both speech families share behind [`SpeechEngine`]: the request
//! slot that steps a request segment by segment, and the checks every
//! request passes before a family prepares its segments.
//!
//! It lives with the families rather than beside the trait, because only
//! engines built on the MLX runtime use it.
//!
//! [`SpeechEngine`]: crate::speech_engine::SpeechEngine

use astronomical_speech_contract::{
    NATURAL_PACE_ONLY_REASON, SpeechFailure, SpeechRequest, unknown_voice_reason,
};

use crate::speech_engine::{SpeechEngineLoadResult, SpeechEngineStep};

/// One spoken segment, before it is counted.
pub(crate) struct SpokenSegment {
    pub(crate) samples: Vec<f32>,
    pub(crate) was_cut_short: bool,
}

/// How a family speaks one of its segments.
pub(crate) trait SegmentSpeaker<Segment, RequestContext> {
    fn speak_segment(
        &mut self,
        segment: &Segment,
        segment_index: usize,
        request_context: &RequestContext,
    ) -> Result<SpokenSegment, SpeechFailure>;
}

/// The request an engine is speaking, if any: its segments, how many are
/// spoken, and what every segment needs (voice, pace, seed, ...). This is
/// the state machine both families share.
pub(crate) struct SpeechRequestSlot<Segment, RequestContext> {
    in_progress: Option<SpeechInProgress<Segment, RequestContext>>,
}

struct SpeechInProgress<Segment, RequestContext> {
    segments: Vec<Segment>,
    completed_segments: usize,
    request_context: RequestContext,
}

impl<Segment, RequestContext> SpeechRequestSlot<Segment, RequestContext> {
    pub(crate) fn empty() -> Self {
        Self { in_progress: None }
    }

    /// Refuses a new request while one is in progress; checked before a
    /// request's segments are prepared, which may be costly.
    pub(crate) fn ensure_idle(&self) -> Result<(), SpeechFailure> {
        if self.in_progress.is_some() {
            return Err(SpeechFailure::EngineBusy);
        }
        Ok(())
    }

    pub(crate) fn start(
        &mut self,
        segments: Vec<Segment>,
        request_context: RequestContext,
    ) -> Result<(), SpeechFailure> {
        self.ensure_idle()?;
        if segments.is_empty() {
            return Err(SpeechFailure::invalid_request(
                "the text has nothing that can be spoken",
            ));
        }
        self.in_progress = Some(SpeechInProgress {
            segments,
            completed_segments: 0,
            request_context,
        });
        Ok(())
    }

    /// Speaks the next segment through `speaker`. After the last segment the
    /// request ends with [`SpeechEngineStep::Completed`]; a failed segment
    /// ends it with the error.
    pub(crate) fn advance(
        &mut self,
        speaker: &mut impl SegmentSpeaker<Segment, RequestContext>,
    ) -> Result<SpeechEngineStep, SpeechFailure> {
        let Some(mut in_progress) = self.in_progress.take() else {
            return Err(SpeechFailure::fatal_execution(
                "advance_speech was called with no speech request in progress",
            ));
        };
        let segment_index = in_progress.completed_segments;
        let total_segments = in_progress.segments.len();
        if segment_index == total_segments {
            return Ok(SpeechEngineStep::Completed);
        }
        let spoken_segment = speaker.speak_segment(
            &in_progress.segments[segment_index],
            segment_index,
            &in_progress.request_context,
        )?;
        in_progress.completed_segments += 1;
        let completed_segments = in_progress.completed_segments;
        self.in_progress = Some(in_progress);
        Ok(SpeechEngineStep::SegmentSpoken {
            samples: spoken_segment.samples,
            completed_segments,
            total_segments,
            was_cut_short: spoken_segment.was_cut_short,
        })
    }

    pub(crate) fn cancel(&mut self) {
        self.in_progress = None;
    }
}

/// Whether `speech_step` was the last step of its request: the request was
/// spoken to its end, or a segment failed and ended it.
pub(crate) fn ends_request(speech_step: &Result<SpeechEngineStep, SpeechFailure>) -> bool {
    !matches!(speech_step, Ok(SpeechEngineStep::SegmentSpoken { .. }))
}

/// The model-independent part of starting a request: the contract's checks,
/// then the pace and the voice against what the engine loaded. Returns the
/// voice to speak in.
pub(crate) fn voice_for_checked_request(
    speech_request: &SpeechRequest,
    load_result: &SpeechEngineLoadResult,
) -> Result<String, SpeechFailure> {
    speech_request.validate()?;
    if !load_result.supports_speed && speech_request.speed != 1.0 {
        return Err(SpeechFailure::invalid_request(NATURAL_PACE_ONLY_REASON));
    }
    let voice_name = speech_request
        .voice_name
        .clone()
        .unwrap_or_else(|| load_result.default_voice_name.clone());
    if !load_result.voice_names.contains(&voice_name) {
        return Err(SpeechFailure::invalid_request(unknown_voice_reason(
            &voice_name,
            &load_result.voice_names,
        )));
    }
    Ok(voice_name)
}

#[cfg(test)]
#[path = "speech_request_slot_tests.rs"]
mod tests;
