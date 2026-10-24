//! Smaller resident prompt-prefill chunk after a transient-peak-only rejection.
//!
//! The resident engine holds every expert in RAM, so a smaller forward re-reads
//! nothing from storage: shrinking only trades a little fixed per-forward
//! overhead for a smaller transient workspace. That is why the resident engine
//! may shrink while the streaming engine keeps its terminal capacity contract
//! (each streaming forward re-pays complete-layer storage reads).
//!
//! Shrinking is admission-only. It applies when the adaptive RAM growth guard
//! accepts the stable footprint but rejects the projected transient peak of the
//! configured chunk. A stable shortfall cannot be fixed by a smaller forward and
//! never reaches this module.

/// Below 256 tokens the fixed per-forward overhead dominates the transient
/// workspace, so a still-rejected 256-token chunk reports a real capacity
/// mismatch instead of collapsing into many tiny forwards.
pub(super) const MINIMUM_SHRUNK_RESIDENT_PREFILL_CHUNK_TOKEN_COUNT: usize = 256;

/// Halves the rejected chunk, never below the minimum. Returns `None` once the
/// rejected chunk is already at or below the minimum.
pub(super) fn next_shrunk_resident_prefill_chunk_token_count(
    rejected_prefill_chunk_token_count: usize,
) -> Option<usize> {
    if rejected_prefill_chunk_token_count <= MINIMUM_SHRUNK_RESIDENT_PREFILL_CHUNK_TOKEN_COUNT {
        return None;
    }
    Some(
        (rejected_prefill_chunk_token_count / 2)
            .max(MINIMUM_SHRUNK_RESIDENT_PREFILL_CHUNK_TOKEN_COUNT),
    )
}

#[cfg(test)]
#[path = "prefill_chunk_shrink_tests.rs"]
mod prefill_chunk_shrink_tests;
