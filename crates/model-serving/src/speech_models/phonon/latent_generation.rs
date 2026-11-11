//! The frame-by-frame loop that speaks one text chunk, and the rules that
//! end it: the model's end-of-speech signal plus one to three closing
//! frames, or the chunk's frame budget.

/// Roughly three tokens per second of speech plus two seconds of slack.
const TOKENS_PER_SECOND_OF_SPEECH: f64 = 3.0;
const SLACK_SECONDS: f64 = 2.0;

/// The most frames a chunk of `token_count` tokens may take, as the
/// reference runtime budgets it.
pub(crate) fn frame_budget(token_count: usize, frame_rate_hertz: f64) -> usize {
    ((token_count as f64 / TOKENS_PER_SECOND_OF_SPEECH + SLACK_SECONDS) * frame_rate_hertz).ceil()
        as usize
}

/// The reference runtime's `EosPolicy`: once the model signals the end of
/// speech, `frames_after_end` more frames close the utterance. It is asked
/// after each frame is kept, so the signalling frame itself is spoken.
#[derive(Clone, Copy, Debug)]
pub(crate) struct EndOfSpeechCountdown {
    frames_after_end: usize,
    frames_left: Option<usize>,
}

impl EndOfSpeechCountdown {
    pub(crate) fn new(frames_after_end: usize) -> Self {
        Self {
            frames_after_end,
            frames_left: None,
        }
    }

    /// Records whether the frame just kept signalled the end of speech, and
    /// says whether generation stops now.
    pub(crate) fn should_stop(&mut self, is_end_of_speech: bool) -> bool {
        if is_end_of_speech && self.frames_left.is_none() {
            self.frames_left = Some(self.frames_after_end);
        }
        match self.frames_left.as_mut() {
            None => false,
            Some(0) => true,
            Some(frames_left) => {
                *frames_left -= 1;
                false
            }
        }
    }

    pub(crate) fn has_seen_end_of_speech(&self) -> bool {
        self.frames_left.is_some()
    }
}

/// How one chunk's generation went, so a caller can tell a chunk that ended
/// on its own from one cut off by its budget.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ChunkGenerationReport {
    pub(crate) frame_count: usize,
    pub(crate) was_ended_by_end_of_speech: bool,
}

#[cfg(test)]
#[path = "latent_generation_tests.rs"]
mod tests;
