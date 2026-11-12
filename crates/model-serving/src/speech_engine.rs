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
//! The worker holds its speech engine as a type parameter, as it holds its
//! image and decision engines: the real engines keep MLX handles, which are
//! not `Send`, so a boxed engine in the worker's own type would make every
//! worker future unsendable, the scripted ones its tests spawn included.
//! Workers without a speech model use [`SpeechUnavailableEngine`]; the
//! inference worker uses `Box<dyn SpeechEngine>`. There are no request ids
//! on these calls: the worker owns the id of the one request an engine
//! speaks.

use astronomical_speech_contract::{SpeechFailure, SpeechRequest};

use crate::memory::telemetry::MlxMemoryTelemetry;

/// What a loaded speech engine offers.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SpeechEngineLoadResult {
    pub model_id: String,
    pub sample_rate_hertz: u32,
    pub voice_names: Vec<String>,
    /// The voice a request without one is spoken in.
    pub default_voice_name: String,
    /// Whether the model can change its pace; when not, a request must ask
    /// for speed 1.0.
    pub supports_speed: bool,
}

/// A request the engine accepted and how it will be spoken.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StartedSpeech {
    /// The voice it is spoken in: the one asked for, or the default.
    pub voice_name: String,
    pub sample_rate_hertz: u32,
    /// How many steps of [`SpeechEngine::advance_speech`] speak it.
    pub total_segments: usize,
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
    fn start_speech(
        &mut self,
        speech_request: SpeechRequest,
    ) -> Result<StartedSpeech, SpeechFailure>;

    /// Speaks the next segment, or reports that none is left. A segment that
    /// fails ends its request: the error is returned and the engine is free
    /// for the next request.
    fn advance_speech(&mut self) -> Result<SpeechEngineStep, SpeechFailure>;

    /// Drops the request in progress, if any.
    fn cancel_speech(&mut self);

    /// The smallest MLX memory ceiling this model can be served under.
    fn minimum_mlx_memory_ceiling_bytes(&self) -> u64 {
        1
    }

    /// The MLX observation taken after the last request ended and its
    /// memory was returned, once; the worker's finalization event carries it.
    fn take_post_cleanup_memory_telemetry(&mut self) -> Option<MlxMemoryTelemetry> {
        None
    }

    /// An MLX observation taken now, for idle polls and per-segment events.
    fn collect_mlx_memory_telemetry(&self) -> Option<MlxMemoryTelemetry> {
        None
    }
}

/// The speech slot of a worker whose model does not speak.
#[doc(hidden)]
pub struct SpeechUnavailableEngine;

impl SpeechEngine for SpeechUnavailableEngine {
    fn load(&mut self) -> Result<SpeechEngineLoadResult, SpeechFailure> {
        Err(SpeechFailure::fatal_execution(
            "the selected model does not speak",
        ))
    }

    fn start_speech(
        &mut self,
        _speech_request: SpeechRequest,
    ) -> Result<StartedSpeech, SpeechFailure> {
        Err(SpeechFailure::fatal_execution(
            "the selected model does not speak",
        ))
    }

    fn advance_speech(&mut self) -> Result<SpeechEngineStep, SpeechFailure> {
        Err(SpeechFailure::fatal_execution(
            "the selected model does not speak",
        ))
    }

    fn cancel_speech(&mut self) {}
}

/// A boxed engine is an engine, so the inference worker can hold whichever
/// speech family its factory built without naming each family in its type.
impl SpeechEngine for Box<dyn SpeechEngine> {
    fn load(&mut self) -> Result<SpeechEngineLoadResult, SpeechFailure> {
        self.as_mut().load()
    }

    fn start_speech(
        &mut self,
        speech_request: SpeechRequest,
    ) -> Result<StartedSpeech, SpeechFailure> {
        self.as_mut().start_speech(speech_request)
    }

    fn advance_speech(&mut self) -> Result<SpeechEngineStep, SpeechFailure> {
        self.as_mut().advance_speech()
    }

    fn cancel_speech(&mut self) {
        self.as_mut().cancel_speech();
    }

    fn minimum_mlx_memory_ceiling_bytes(&self) -> u64 {
        self.as_ref().minimum_mlx_memory_ceiling_bytes()
    }

    fn take_post_cleanup_memory_telemetry(&mut self) -> Option<MlxMemoryTelemetry> {
        self.as_mut().take_post_cleanup_memory_telemetry()
    }

    fn collect_mlx_memory_telemetry(&self) -> Option<MlxMemoryTelemetry> {
        self.as_ref().collect_mlx_memory_telemetry()
    }
}

#[cfg(test)]
#[path = "speech_engine_tests.rs"]
mod tests;
