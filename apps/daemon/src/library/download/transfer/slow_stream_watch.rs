//! Tells a payload stream that has slowed to a trickle from a slow network.
//!
//! A download keeps several streams in flight, each on its own connection. A
//! connection can degrade on its own (a congested path, a struggling edge
//! server) while the others stay fast, and the read timeout never fires as
//! long as a few bytes keep arriving, so one such stream used to hold its
//! slot for as long as it liked. A stream is therefore judged against what
//! the download's other streams have recently achieved, never against a fixed
//! speed: on a slow network every stream is equally slow and none is restarted.

use std::{
    sync::atomic::{AtomicU64, Ordering},
    time::Duration,
};

/// How long a stream is watched before each judgement. Long enough to span
/// TCP slow start on a fresh connection, short enough that a stalled piece
/// costs seconds, not the 30-second read timeout.
pub(super) const SLOW_STREAM_WINDOW: Duration = Duration::from_secs(10);

/// A stream is slow when it runs below this fraction of the reference rate.
/// One eighth leaves ordinary variation between connections alone.
const SLOW_STREAM_RATE_DIVISOR: u64 = 8;

/// Below this reference rate no stream is judged: the whole link is slow, and
/// restarting connections would only add handshakes to it.
const MINIMUM_REFERENCE_BYTES_PER_SECOND: u64 = 256 * 1024;

/// Each report lets the reference fall by a tenth, so when the network itself
/// slows down the reference follows within a few windows instead of calling
/// every stream slow from then on.
const REFERENCE_DECAY_DIVISOR: u64 = 10;

/// What one stream of a download has recently been able to transfer per
/// second, shared by every stream of that download.
#[derive(Debug, Default)]
pub(super) struct StreamRateReference {
    bytes_per_second: AtomicU64,
}

impl StreamRateReference {
    /// Folds one stream's measured rate into the reference: a faster stream
    /// raises it at once, a slower one lets it decay by a tenth.
    pub(super) fn report(&self, stream_bytes_per_second: u64) {
        let previous_reference = self.bytes_per_second.load(Ordering::Acquire);
        let decayed_reference = previous_reference - previous_reference / REFERENCE_DECAY_DIVISOR;
        // Two streams reporting at once can lose one update; the reference is
        // an estimate, and the next report corrects it.
        self.bytes_per_second.store(
            stream_bytes_per_second.max(decayed_reference),
            Ordering::Release,
        );
    }

    /// Whether a stream that moved `window_bytes` in `window_duration` has
    /// fallen far behind what this download's streams achieve.
    pub(super) fn is_slow(&self, window_bytes: u64, window_duration: Duration) -> bool {
        let reference_bytes_per_second = self.bytes_per_second.load(Ordering::Acquire);
        if reference_bytes_per_second < MINIMUM_REFERENCE_BYTES_PER_SECOND {
            return false;
        }
        bytes_per_second(window_bytes, window_duration)
            < reference_bytes_per_second / SLOW_STREAM_RATE_DIVISOR
    }
}

/// The rate of `transferred_bytes` over `duration`, or zero for a duration
/// too short to measure.
pub(super) fn bytes_per_second(transferred_bytes: u64, duration: Duration) -> u64 {
    let duration_millis = u64::try_from(duration.as_millis()).unwrap_or(u64::MAX);
    if duration_millis == 0 {
        return 0;
    }
    transferred_bytes.saturating_mul(1_000) / duration_millis
}

#[cfg(test)]
#[path = "slow_stream_watch_tests.rs"]
mod slow_stream_watch_tests;
