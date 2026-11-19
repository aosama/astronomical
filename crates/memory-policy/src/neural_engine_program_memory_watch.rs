//! Deciding when a reading of what Neural Engine programs hold can be trusted.
//!
//! The driver wires a running program's weights with a delay and in steps, so a single
//! reading says little. This watches successive readings until they stop rising.

use std::time::Duration;

/// How long the reading must stay flat before it counts as settled. The longest flat
/// stretch seen before a further rise was under half a second (issue #161).
const SETTLED_WIRED_GROWTH_DURATION: Duration = Duration::from_millis(1500);
/// A rise smaller than this share of the weights is the rest of the Mac, not the programs.
const SMALLEST_MEANINGFUL_RISE_DIVISOR: u64 = 64;
/// Ordinary wired churn reaches tens of megabytes, which is most of a sixty-fourth of a
/// small model's weights and would keep such a reading from ever settling.
const SMALLEST_MEANINGFUL_RISE_BYTES: u64 = 64 * 1024 * 1024;

/// Watches wired memory while a set of programs cycles, until what they hold is known.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProgramMemoryWatch {
    weight_bytes: u64,
    highest_growth_bytes: u64,
    last_rise_at: Duration,
}

impl ProgramMemoryWatch {
    #[must_use]
    pub const fn new(weight_bytes: u64) -> Self {
        Self {
            weight_bytes,
            highest_growth_bytes: 0,
            last_rise_at: Duration::ZERO,
        }
    }

    const fn smallest_meaningful_rise_bytes(&self) -> u64 {
        let share_of_weights = self.weight_bytes / SMALLEST_MEANINGFUL_RISE_DIVISOR;
        if share_of_weights > SMALLEST_MEANINGFUL_RISE_BYTES {
            share_of_weights
        } else {
            SMALLEST_MEANINGFUL_RISE_BYTES
        }
    }

    /// Takes one reading and says whether the programs have settled. Below the weights
    /// never counts as settled, because the driver has not finished wiring them yet.
    pub fn observe(&mut self, wired_growth_bytes: u64, elapsed: Duration) -> bool {
        let rise_needed = self
            .highest_growth_bytes
            .saturating_add(self.smallest_meaningful_rise_bytes());
        if wired_growth_bytes > rise_needed {
            self.highest_growth_bytes = wired_growth_bytes;
            self.last_rise_at = elapsed;
        }
        self.highest_growth_bytes >= self.weight_bytes
            && elapsed.saturating_sub(self.last_rise_at) >= SETTLED_WIRED_GROWTH_DURATION
    }

    /// What the programs wired. Their weights are the floor, because the reading is of
    /// the whole system: another process returning memory can hide growth, and programs
    /// cannot hold less than their weights.
    #[must_use]
    pub const fn wired_bytes(&self) -> u64 {
        if self.highest_growth_bytes > self.weight_bytes {
            self.highest_growth_bytes
        } else {
            self.weight_bytes
        }
    }
}

#[cfg(test)]
#[path = "neural_engine_program_memory_watch_tests.rs"]
mod tests;
