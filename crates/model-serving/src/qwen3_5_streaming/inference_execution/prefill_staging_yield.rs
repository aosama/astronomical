//! How many warm-cache bytes one prefill chunk borrows for its staging tape.
//!
//! The chunk tape needs a fixed staging reserve of expert page slots while the
//! graph builds. Admission has already proven the forward's projected peak
//! fits, so any ceiling the admitted peak leaves unclaimed funds staging
//! first. Only the remainder is borrowed from the warm expert cache.
//!
//! Borrowing unconditionally ratcheted the cache down: the warm cache evicts
//! whole layer tables, so every chunk released one 453 MB table to cover a
//! 214 MB loan, and released layers were not re-seated during prefill. The
//! 2026-10-24 Development flight at a 21 GB ceiling drained 34 seated layers
//! to 17 across two requests while 11 GB of the paid ceiling sat unused.

/// Warm-cache bytes the chunk must borrow so its staging reserve fits.
///
/// Never borrows more than half the warm cache, so a nearly empty cache is not
/// drained to zero.
#[must_use]
pub(super) fn staging_yield_bytes(
    staging_reserve_bytes: u64,
    unreserved_headroom_bytes: u64,
    resident_cache_bytes: u64,
) -> u64 {
    staging_reserve_bytes
        .saturating_sub(unreserved_headroom_bytes)
        .min(resident_cache_bytes / 2)
}

#[cfg(test)]
#[path = "prefill_staging_yield_tests.rs"]
mod tests;
