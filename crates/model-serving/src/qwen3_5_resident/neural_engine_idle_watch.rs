//! How long the GPU and Neural Engine have sat idle, for a reading that needs them to.
//!
//! The request paths say so directly. A layer read and a finished program are the
//! preparation's own work, and a gap between its steps catches anything else.

use std::time::{Duration, Instant};

/// MLX's own wired memory falls by 2 GB or more after its last work, and the driver
/// holds programs for 10 to 20 seconds after their last run. Measured sooner, a reading
/// came to as little as 0.37 of the true cost; after this, it agrees within about 1
/// percent (issue #161).
const IDLE_BEFORE_MEASURING: Duration = Duration::from_secs(30);
/// Steps come every 100 ms while the worker is idle, so a longer gap means something ran
/// that [`IdleWatch::note_use`] did not hear about.
const STEP_GAP_THAT_MEANS_A_REQUEST_RAN: Duration = Duration::from_secs(1);

#[derive(Clone, Copy, Debug)]
pub(crate) struct IdleWatch {
    last_use_at: Instant,
    last_step_at: Instant,
}

impl IdleWatch {
    pub(crate) const fn new(started_at: Instant) -> Self {
        Self {
            last_use_at: started_at,
            last_step_at: started_at,
        }
    }

    pub(crate) const fn note_use(&mut self, used_at: Instant) {
        self.last_use_at = used_at;
    }

    /// Takes one step and says whether everything has been idle long enough to measure.
    pub(crate) fn step(&mut self, now: Instant, was_used_this_step: bool) -> bool {
        if was_used_this_step
            || now.saturating_duration_since(self.last_step_at) > STEP_GAP_THAT_MEANS_A_REQUEST_RAN
        {
            self.last_use_at = now;
        }
        self.last_step_at = now;
        now.saturating_duration_since(self.last_use_at) >= IDLE_BEFORE_MEASURING
    }

    /// How much of the wait is left, for the status.
    pub(crate) fn remaining(&self, now: Instant) -> Duration {
        IDLE_BEFORE_MEASURING.saturating_sub(now.saturating_duration_since(self.last_use_at))
    }
}

#[cfg(test)]
#[path = "neural_engine_idle_watch_tests.rs"]
mod tests;
