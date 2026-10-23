//! Lock-free aggregate of positional file reads performed by lazy MLX arrays.
//!
//! # Why these counters exist
//!
//! The custom readers serve bytes into MLX lazily, so "how much did loading
//! this model actually read, and did the reads parallelize?" is invisible
//! without instrumentation. The snapshot feeds:
//!
//! - the performance-attribution counters in `model-serving` (expert paging
//!   input/output visibility),
//! - the streaming-latency acceptance journeys, and
//! - ad-hoc diagnosis of paged prefill wall time.
//!
//! # Design choices
//!
//! - **Lock-free atomics with `Relaxed` ordering** — every reader thread
//!   increments these from inside C callbacks where blocking is unacceptable,
//!   and the counters are monotone aggregates: a torn multi-counter read at
//!   most mixes two adjacent samples, which no consumer is sensitive to.
//! - **Measurement without synchronization** — a snapshot deliberately does
//!   not drain the graphics-processor stream; it samples what has completed,
//!   keeping telemetry cheap enough for per-request attribution.
//! - **Documented limits** — these count *logical* reads issued by the
//!   vtable; macOS may satisfy them from the file cache, so they are not
//!   proof of equal physical SSD (Solid-State Drive) traffic.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// Lock-free aggregate of positional file reads performed by lazy MLX arrays.
///
/// Share one instance across readers by wrapping it in an `Arc`; every
/// method is thread-safe.
#[derive(Debug, Default)]
pub struct PositionalFileReadMetrics {
    active_read_count: AtomicU64,
    maximum_concurrent_read_count: AtomicU64,
    read_call_count: AtomicU64,
    read_byte_count: AtomicU64,
    total_read_elapsed_nanoseconds: AtomicU64,
    maximum_read_elapsed_nanoseconds: AtomicU64,
    read_failure_count: AtomicU64,
}

/// Immutable positional file-read counters captured without synchronizing MLX execution.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PositionalFileReadMetricsSnapshot {
    pub maximum_concurrent_read_count: u64,
    pub read_call_count: u64,
    pub read_byte_count: u64,
    pub total_read_elapsed_nanoseconds: u64,
    pub maximum_read_elapsed_nanoseconds: u64,
    pub read_failure_count: u64,
}

impl PositionalFileReadMetrics {
    /// Captures the current aggregate counters.
    ///
    /// Consistent per-field but not cross-field atomic: fields are read in
    /// sequence, so two fields can straddle an in-flight read. Consumers
    /// aggregate over whole requests, where this skew is noise.
    #[must_use]
    pub fn snapshot(&self) -> PositionalFileReadMetricsSnapshot {
        PositionalFileReadMetricsSnapshot {
            maximum_concurrent_read_count: self
                .maximum_concurrent_read_count
                .load(Ordering::Relaxed),
            read_call_count: self.read_call_count.load(Ordering::Relaxed),
            read_byte_count: self.read_byte_count.load(Ordering::Relaxed),
            total_read_elapsed_nanoseconds: self
                .total_read_elapsed_nanoseconds
                .load(Ordering::Relaxed),
            maximum_read_elapsed_nanoseconds: self
                .maximum_read_elapsed_nanoseconds
                .load(Ordering::Relaxed),
            read_failure_count: self.read_failure_count.load(Ordering::Relaxed),
        }
    }

    /// Measures one read operation, updating every counter around it.
    ///
    /// Used only by the readers in this module family: the wrapper records
    /// concurrency (current and peak), then call count, byte count on
    /// success, total and maximum elapsed time, and failures. The boolean
    /// result of the wrapped operation passes through untouched so callers
    /// keep their own failure semantics.
    pub(crate) fn measure_read(
        &self,
        byte_count: usize,
        read_operation: impl FnOnce() -> bool,
    ) -> bool {
        let concurrent_read_count = self.active_read_count.fetch_add(1, Ordering::Relaxed) + 1;
        self.maximum_concurrent_read_count
            .fetch_max(concurrent_read_count, Ordering::Relaxed);
        let read_started_at = Instant::now();
        let read_succeeded = read_operation();
        self.active_read_count.fetch_sub(1, Ordering::Relaxed);

        let elapsed_nanoseconds =
            u64::try_from(read_started_at.elapsed().as_nanos()).unwrap_or(u64::MAX);
        self.read_call_count.fetch_add(1, Ordering::Relaxed);
        if read_succeeded {
            self.read_byte_count.fetch_add(
                u64::try_from(byte_count).unwrap_or(u64::MAX),
                Ordering::Relaxed,
            );
        }
        self.total_read_elapsed_nanoseconds
            .fetch_add(elapsed_nanoseconds, Ordering::Relaxed);
        self.maximum_read_elapsed_nanoseconds
            .fetch_max(elapsed_nanoseconds, Ordering::Relaxed);
        if !read_succeeded {
            self.read_failure_count.fetch_add(1, Ordering::Relaxed);
        }
        read_succeeded
    }
}
