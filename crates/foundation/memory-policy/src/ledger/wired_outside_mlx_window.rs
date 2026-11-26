//! The memory wired outside MLX that the reserve for macOS's wired line plans
//! around, read over a window longer than macOS's release lag.
//!
//! "Wired outside MLX" is the system's wired total less what MLX holds. macOS
//! un-wires memory MLX releases only some time after MLX stops counting it, so
//! for that long the figure counts MLX's own released memory as someone
//! else's. Measured with the MLX performance profiler's wire release probe on
//! an Apple silicon Mac: a median of 167 ms and at most 258 ms over ten 1 GiB
//! releases. Read per forward, that made single readings at prompt boundaries
//! jump by gigabytes, and each jump dropped a whole `astro_expert_cache` table
//! that was read back a moment later.
//!
//! So the plan takes the lowest reading of the last
//! `RELEASE_LAG_WINDOW_MILLISECONDS`, together with the last reading before
//! that window. A release's inflated readings end within the lag, so a true
//! reading is always still in the window; memory another application really
//! wires stays wired, so once it has lasted the whole window it is the lowest
//! reading and counts. A fall counts at once.

use std::collections::VecDeque;

/// Longer than the largest release lag measured (258 ms), with room to
/// spare for a busier Mac.
pub const RELEASE_LAG_WINDOW_MILLISECONDS: u64 = 300;

/// The recent readings of the memory wired outside MLX.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WiredOutsideMlxWindow {
    /// `(reading_milliseconds, wired_outside_mlx_bytes)`, oldest first: every
    /// reading in the window and the newest one before it.
    readings: VecDeque<(u64, u64)>,
    window_milliseconds: u64,
}

impl Default for WiredOutsideMlxWindow {
    fn default() -> Self {
        Self::with_window_milliseconds(RELEASE_LAG_WINDOW_MILLISECONDS)
    }
}

impl WiredOutsideMlxWindow {
    /// A window of another length, for the replay lab to weigh against the
    /// engine's; zero takes every reading as read.
    #[must_use]
    pub const fn with_window_milliseconds(window_milliseconds: u64) -> Self {
        Self {
            readings: VecDeque::new(),
            window_milliseconds,
        }
    }

    /// Takes a reading made at `reading_milliseconds` on a monotonic clock
    /// and returns the figure to plan around now.
    pub fn observe(&mut self, reading_milliseconds: u64, wired_outside_mlx_bytes: u64) -> u64 {
        self.readings
            .push_back((reading_milliseconds, wired_outside_mlx_bytes));
        let window_start_milliseconds =
            reading_milliseconds.saturating_sub(self.window_milliseconds);
        while let Some(&(second_oldest_milliseconds, _)) = self.readings.get(1)
            && second_oldest_milliseconds <= window_start_milliseconds
        {
            self.readings.pop_front();
        }
        if self.window_milliseconds == 0 {
            return wired_outside_mlx_bytes;
        }
        let mut lowest_bytes = wired_outside_mlx_bytes;
        for &(_, recent_bytes) in &self.readings {
            lowest_bytes = lowest_bytes.min(recent_bytes);
        }
        lowest_bytes
    }

    /// The last reading as read; `None` before any.
    #[must_use]
    pub fn latest_wired_outside_mlx_bytes(&self) -> Option<u64> {
        self.readings.back().map(|&(_, latest_bytes)| latest_bytes)
    }

    /// When the last reading was made; `None` before any.
    #[must_use]
    pub fn latest_reading_milliseconds(&self) -> Option<u64> {
        self.readings
            .back()
            .map(|&(latest_milliseconds, _)| latest_milliseconds)
    }
}

#[cfg(test)]
#[path = "wired_outside_mlx_window_tests.rs"]
mod tests;
