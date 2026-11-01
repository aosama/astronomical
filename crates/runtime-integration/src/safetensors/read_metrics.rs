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
use std::sync::{Arc, OnceLock};
use std::time::Instant;

/// Lock-free aggregate of positional file reads performed by lazy MLX arrays.
///
/// Share one instance across readers by wrapping it in an `Arc`; every
/// method is thread-safe. An instance may carry one `peer_read_metrics`
/// target installed once (see [`Self::install_peer_read_metrics`]); every
/// counter then mirrors into the peer, so a request-scoped aggregate and a
/// session-scoped aggregate can both stay truthful from the same reads.
#[derive(Debug, Default)]
pub struct PositionalFileReadMetrics {
    active_read_count: AtomicU64,
    maximum_concurrent_read_count: AtomicU64,
    read_call_count: AtomicU64,
    read_byte_count: AtomicU64,
    total_read_elapsed_nanoseconds: AtomicU64,
    maximum_read_elapsed_nanoseconds: AtomicU64,
    read_failure_count: AtomicU64,
    peer_read_metrics: OnceLock<Arc<PositionalFileReadMetrics>>,
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

    /// Installs the session-scoped mirror target for this aggregate.
    ///
    /// Called once by the pager's streaming path when diagnostics
    /// attribution owns a request-scoped aggregate: the readers keep feeding
    /// the request aggregate (so per-request report deltas stay exact) and
    /// every counter mirrors into the session aggregate so the observatory
    /// expert page stays truthful even while diagnostics run. A second
    /// install is ignored; one peer per aggregate is the whole model.
    pub fn install_peer_read_metrics(&self, peer_read_metrics: Arc<Self>) {
        let _already_installed = self.peer_read_metrics.set(peer_read_metrics);
    }

    /// Measures one read operation, updating every counter around it.
    ///
    /// Used only by the readers in this module family: the wrapper records
    /// concurrency (current and peak), then call count, byte count on
    /// success, total and maximum elapsed time, and failures. The boolean
    /// result of the wrapped operation passes through untouched so callers
    /// keep their own failure semantics. Every cumulative counter mirrors
    /// into the installed peer so request-scoped and session-scoped
    /// aggregates observe the same reads; the peer needs no live
    /// active-read tracking because snapshots only expose the concurrency
    /// peak, which is mirrored directly.
    pub(crate) fn measure_read(
        &self,
        byte_count: usize,
        read_operation: impl FnOnce() -> bool,
    ) -> bool {
        let peer_read_metrics = self.peer_read_metrics.get();
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
        Self::mirror_completed_read_into_peer(
            peer_read_metrics,
            concurrent_read_count,
            byte_count,
            elapsed_nanoseconds,
            read_succeeded,
        );
        read_succeeded
    }

    /// Mirrors one completed read into the session aggregate when this request
    /// aggregate has a peer. The peer skips live active-read tracking because
    /// snapshots only expose the concurrency peak, which is mirrored directly.
    fn mirror_completed_read_into_peer(
        peer_read_metrics: Option<&Arc<Self>>,
        concurrent_read_count: u64,
        byte_count: usize,
        elapsed_nanoseconds: u64,
        read_succeeded: bool,
    ) {
        if let Some(peer_read_metrics) = peer_read_metrics {
            peer_read_metrics
                .maximum_concurrent_read_count
                .fetch_max(concurrent_read_count, Ordering::Relaxed);
            peer_read_metrics
                .read_call_count
                .fetch_add(1, Ordering::Relaxed);
            if read_succeeded {
                peer_read_metrics.read_byte_count.fetch_add(
                    u64::try_from(byte_count).unwrap_or(u64::MAX),
                    Ordering::Relaxed,
                );
            }
            peer_read_metrics
                .total_read_elapsed_nanoseconds
                .fetch_add(elapsed_nanoseconds, Ordering::Relaxed);
            peer_read_metrics
                .maximum_read_elapsed_nanoseconds
                .fetch_max(elapsed_nanoseconds, Ordering::Relaxed);
            if !read_succeeded {
                peer_read_metrics
                    .read_failure_count
                    .fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}
