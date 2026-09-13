//! Background speculative page warmer (issue #593).
//!
//! The cross-layer gate technique predicts which experts the next decoder
//! layer needs while the current layer still computes. Those predictions are
//! only worth anything if their storage reads happen *during* the graphics
//! processor's busy windows instead of inside the stall the prediction is
//! trying to remove.
//!
//! This warmer owns the overlap. The decode thread hands it file ranges and
//! keeps moving; a background thread issues positioned reads of exactly those
//! ranges and discards the bytes. The operating system keeps the pages in its
//! file cache, so when the decode thread's real page load issues the same
//! positioned reads moments later, they resolve from RAM instead of from the
//! solid-state drive.
//!
//! Why discard instead of handing bytes over: the expert page loader fuses its
//! reads into MLX array creation behind a callback interface, so raw bytes
//! cannot be handed to the owner thread without new C surface. Warming the
//! file cache needs no new interface, no MLX access from a second thread, and
//! no buffer lifetime management — the loader path is untouched.
//!
//! Fail-open by contract: a full queue drops the request and the next layer's
//! real page load simply reads from storage as it does today.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{SyncSender, TrySendError, sync_channel};
use std::thread::{self, JoinHandle};

/// Bounded queue of pending warm requests. Each request carries the byte
/// ranges of one layer's candidate set, so four queued layers bound the
/// warmer's lookahead without unbounded memory or stale work.
const WARMER_QUEUE_CAPACITY: usize = 4;

/// One layer's speculative warming work: byte ranges grouped by source file,
/// ready for positioned reads.
pub(crate) struct SpeculativeWarmRequest {
    pub file_ranges: Vec<(PathBuf, Vec<(u64, usize)>)>,
}

/// Background warmer owning the file-cache prefetch for speculative routes.
#[derive(Debug)]
pub(crate) struct SpeculativePageWarmer {
    request_sender: Option<SyncSender<SpeculativeWarmRequest>>,
    warmer_thread: Option<JoinHandle<()>>,
    dropped_request_count: Arc<AtomicU64>,
    completed_request_count: Arc<AtomicU64>,
}

impl SpeculativePageWarmer {
    /// Spawns the warmer thread. Spawn failure degrades to no warming: the
    /// caller keeps the synchronous page-load path.
    pub(crate) fn try_spawn() -> Option<Self> {
        let dropped_request_count = Arc::new(AtomicU64::new(0));
        let completed_request_count = Arc::new(AtomicU64::new(0));
        let (request_sender, request_receiver) =
            sync_channel::<SpeculativeWarmRequest>(WARMER_QUEUE_CAPACITY);
        let completed_request_count_for_thread = Arc::clone(&completed_request_count);
        let warmer_thread = thread::Builder::new()
            .name("speculative-page-warmer".to_owned())
            .spawn(move || {
                // One scratch buffer grows to the largest range seen and is
                // reused for every later read, so warming allocates nothing
                // steady-state.
                let mut scratch_bytes: Vec<u8> = Vec::new();
                while let Ok(request) = request_receiver.recv() {
                    for (source_file_path, ranges) in &request.file_ranges {
                        let Ok(mut source_file) = std::fs::File::open(source_file_path) else {
                            continue;
                        };
                        for (source_offset, source_byte_count) in ranges {
                            if scratch_bytes.len() < *source_byte_count {
                                scratch_bytes.resize(*source_byte_count, 0);
                            }
                            let warmed_bytes = &mut scratch_bytes[..*source_byte_count];
                            // A failed warming read is harmless: the real page
                            // load reads storage exactly as it would have.
                            let _ = std::os::unix::fs::FileExt::read_exact_at(
                                &mut source_file,
                                warmed_bytes,
                                *source_offset,
                            );
                        }
                    }
                    completed_request_count_for_thread.fetch_add(1, Ordering::Relaxed);
                }
            })
            .ok()?;
        Some(Self {
            request_sender: Some(request_sender),
            warmer_thread: Some(warmer_thread),
            dropped_request_count,
            completed_request_count,
        })
    }

    /// Offers one layer's ranges for background warming. A full queue drops
    /// the request and counts the drop; decode never waits on the warmer.
    pub(crate) fn try_warm(&self, request: SpeculativeWarmRequest) {
        let Some(request_sender) = self.request_sender.as_ref() else {
            return;
        };
        match request_sender.try_send(request) {
            Ok(()) => {}
            Err(TrySendError::Full(_) | TrySendError::Disconnected(_)) => {
                self.dropped_request_count.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// Reads the drop counter without consuming it. Tests poll this while
    /// waiting for the warmer to account for every dispatched request.
    #[cfg(test)]
    #[must_use]
    pub(crate) fn dropped_request_count(&self) -> u64 {
        self.dropped_request_count.load(Ordering::Relaxed)
    }

    /// Consumes the drop counter for attribution recording.
    #[must_use]
    pub(crate) fn take_dropped_request_count(&self) -> u64 {
        self.dropped_request_count.swap(0, Ordering::Relaxed)
    }

    /// Requests fully warmed so far. The decode thread reads this to record
    /// whether background warming kept pace or lagged behind the forward.
    #[must_use]
    pub(crate) fn completed_request_count(&self) -> u64 {
        self.completed_request_count.load(Ordering::Relaxed)
    }
}

impl Drop for SpeculativePageWarmer {
    fn drop(&mut self) {
        // Disconnect the queue so the warmer's recv returns. Do not join: a
        // warmer mid-read must not stall model teardown.
        self.request_sender.take();
        let _ = self.warmer_thread.take();
    }
}

#[cfg(test)]
mod warmer_tests {
    use super::*;

    fn write_source_file(directory: &std::path::Path, byte_count: usize) -> PathBuf {
        let source_path = directory.join("warmer-source.bin");
        let source_bytes: Vec<u8> = (0..byte_count).map(|index| index as u8).collect();
        std::fs::write(&source_path, source_bytes)
            .expect("the warmer test source file should be writable");
        source_path
    }

    #[test]
    fn warmer_reads_dispatched_ranges_and_counts_queue_drops() {
        let temporary_directory =
            tempfile::tempdir().expect("the warmer test directory should be created");
        let source_path = write_source_file(temporary_directory.path(), 64);
        let warmer =
            SpeculativePageWarmer::try_spawn().expect("the warmer thread should spawn in tests");
        // Six requests against a four-slot queue: every request must end up
        // either completed by the warmer or counted as a dropped offer. No
        // request may vanish, and the caller must never block.
        let dispatched_request_count = 6_u64;
        for _ in 0..dispatched_request_count {
            warmer.try_warm(SpeculativeWarmRequest {
                file_ranges: vec![(source_path.clone(), vec![(8, 16), (40, 8)])],
            });
        }
        let wait_started_at = std::time::Instant::now();
        while warmer.completed_request_count() + warmer.dropped_request_count()
            < dispatched_request_count
            && wait_started_at.elapsed() < std::time::Duration::from_secs(2)
        {
            std::thread::yield_now();
        }
        assert_eq!(
            warmer.completed_request_count() + warmer.dropped_request_count(),
            dispatched_request_count,
            "every warmed or dropped request must be accounted for"
        );
    }
}
