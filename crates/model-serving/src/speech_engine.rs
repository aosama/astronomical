//! The worker-facing seam of a text-to-speech model family.
//!
//! Speech is stepped like image generation rather than answered in one call
//! like a decision: a request is split into segments (Kokoro reads at most
//! 510 phonemes at a time, Phonon at most 50 tokens), and each step speaks
//! one. A long request can then be streamed segment by segment and
//! cancelled between segments. An engine speaks one request at a time; the
//! state machine both families use for that is
//! `speech_models::speech_request_slot`.
//!
//! The worker integration that comes next adds what the image and decision
//! seams already have: request ids on these calls, a placeholder engine for
//! workers without a speech model, and an implementation for
//! `Box<dyn SpeechEngine>`.

use astronomical_speech_contract::{SpeechFailure, SpeechRequest};

/// What a loaded speech engine offers.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SpeechEngineLoadResult {
    pub sample_rate_hertz: u32,
    pub voice_names: Vec<String>,
    /// The voice a request without one is spoken in.
    pub default_voice_name: String,
    /// Whether the model can change its pace; when not, a request must ask
    /// for speed 1.0.
    pub supports_speed: bool,
}

/// One step of a speech request.
#[derive(Clone, Debug, PartialEq)]
pub enum SpeechEngineStep {
    /// One more segment is spoken; its samples follow the previous ones.
    SegmentSpoken {
        samples: Vec<f32>,
        completed_segments: usize,
        total_segments: usize,
        /// Whether the model stopped before saying the segment to its end
        /// (Phonon's frame budget ran out); the audio is kept, but a caller
        /// may want to know.
        was_cut_short: bool,
    },
    /// Every segment is spoken; the request has ended.
    Completed,
}

pub trait SpeechEngine: 'static {
    /// Loads the model; called once before any request.
    fn load(&mut self) -> Result<SpeechEngineLoadResult, SpeechFailure>;

    /// Checks `speech_request` against the loaded model and prepares its
    /// segments. Fails with [`SpeechFailure::EngineBusy`] while another
    /// request is in progress.
    fn start_speech(&mut self, speech_request: SpeechRequest) -> Result<(), SpeechFailure>;

    /// Speaks the next segment, or reports that none is left. A segment that
    /// fails ends its request: the error is returned and the engine is free
    /// for the next request.
    fn advance_speech(&mut self) -> Result<SpeechEngineStep, SpeechFailure>;

    /// Drops the request in progress, if any.
    fn cancel_speech(&mut self);
}
