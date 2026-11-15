//! Short-lived needs outside the normal pattern.
//!
//! Most memory follows the same shape every request: weights, context, one
//! layer's MLX working buffers. A few jobs do not: saving or restoring prompt
//! cache blocks, the vision tower for one image, a draft model. Each asks
//! the ledger for a lease naming its purpose and size, and gives it back
//! when done. That is how the code that allocates declares its cost.

use std::collections::BTreeMap;

/// Identifies one granted lease until it is given back.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct LeaseId(u64);

/// Why memory outside the normal pattern is needed.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum LeasePurpose {
    /// Snapshots held until astro-ssd-prompt-cache blocks are written to storage.
    AstroSsdPromptCachePublication,
    /// Blocks read back from the prompt cache before they join the context.
    AstroSsdPromptCacheRestore,
    /// The vision tower, loaded for the images of one request.
    VisionTower,
    /// A draft model serving beside the main one.
    DraftModel,
    /// Extra room for redoing one layer step that hit the limit, held only
    /// until the redone step has been measured.
    RedoHeadroom,
}

/// The leases granted and not yet given back.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct LeaseBook {
    next_lease_number: u64,
    lease_bytes_by_id: BTreeMap<LeaseId, (LeasePurpose, u64)>,
}

impl LeaseBook {
    /// Grants a lease. The bytes count against the limit until it is given
    /// back.
    pub fn grant(&mut self, purpose: LeasePurpose, lease_bytes: u64) -> LeaseId {
        let lease_id = LeaseId(self.next_lease_number);
        self.next_lease_number += 1;
        self.lease_bytes_by_id
            .insert(lease_id, (purpose, lease_bytes));
        lease_id
    }

    /// Gives a lease back. Returns its bytes, or `None` when no such lease is
    /// held: given back twice, or never granted.
    pub fn give_back(&mut self, lease_id: LeaseId) -> Option<u64> {
        let (_, lease_bytes) = self.lease_bytes_by_id.remove(&lease_id)?;
        Some(lease_bytes)
    }

    /// Bytes of every lease currently held.
    #[must_use]
    pub fn held_bytes(&self) -> u64 {
        let mut held_bytes = 0_u64;
        for (_, lease_bytes) in self.lease_bytes_by_id.values() {
            held_bytes = held_bytes.saturating_add(*lease_bytes);
        }
        held_bytes
    }

    /// Bytes of the held leases that stand for memory really in use: every
    /// lease except a redo's headroom, which is room and not an allocation.
    #[must_use]
    pub fn allocated_bytes(&self) -> u64 {
        self.held_bytes()
            .saturating_sub(self.held_bytes_for(LeasePurpose::RedoHeadroom))
    }

    /// Bytes of the leases currently held for one purpose.
    #[must_use]
    pub fn held_bytes_for(&self, purpose: LeasePurpose) -> u64 {
        let mut held_bytes = 0_u64;
        for (lease_purpose, lease_bytes) in self.lease_bytes_by_id.values() {
            if *lease_purpose == purpose {
                held_bytes = held_bytes.saturating_add(*lease_bytes);
            }
        }
        held_bytes
    }
}

#[cfg(test)]
#[path = "lease_tests.rs"]
mod tests;
