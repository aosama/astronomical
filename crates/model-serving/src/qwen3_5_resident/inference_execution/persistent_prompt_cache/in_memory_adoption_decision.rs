//! In-memory prefix-adoption decision for one persistent prompt-cache restore.

/// Outcome of comparing the retained live prefix against the disk store's
/// longest restorable prefix: the pure half of the issue #1167 adoption gate.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum InMemoryPrefixAdoptionDecision {
    /// The retained state restores at least as much as the disk store; the
    /// live arrays win and the disk store is not read for state.
    AdoptRetainedState,
    /// The retained state restores nothing for this prompt's chain.
    ChainMismatch,
    /// The retained state restores something but strictly less than disk.
    RetainedStateShorter,
}

/// Decides adoption from the two restored-token counts. A tie adopts the
/// retained state: live arrays need no file reads at all.
pub(super) fn decide_in_memory_prefix_adoption(
    retained_restored_token_count: usize,
    disk_restored_token_count: usize,
) -> InMemoryPrefixAdoptionDecision {
    if retained_restored_token_count == 0 {
        return InMemoryPrefixAdoptionDecision::ChainMismatch;
    }
    if retained_restored_token_count >= disk_restored_token_count {
        return InMemoryPrefixAdoptionDecision::AdoptRetainedState;
    }
    InMemoryPrefixAdoptionDecision::RetainedStateShorter
}

#[cfg(test)]
#[path = "in_memory_adoption_decision_tests.rs"]
mod in_memory_adoption_decision_tests;
