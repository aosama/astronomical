use super::{InMemoryPrefixAdoptionDecision, decide_in_memory_prefix_adoption};

#[test]
fn should_adopt_the_retained_state_when_it_restores_at_least_as_much_as_disk() {
    assert_eq!(
        decide_in_memory_prefix_adoption(1_024, 1_024),
        InMemoryPrefixAdoptionDecision::AdoptRetainedState
    );
    assert_eq!(
        decide_in_memory_prefix_adoption(2_048, 1_024),
        InMemoryPrefixAdoptionDecision::AdoptRetainedState
    );
}

#[test]
fn should_report_a_chain_mismatch_when_the_retained_state_restores_nothing() {
    assert_eq!(
        decide_in_memory_prefix_adoption(0, 1_024),
        InMemoryPrefixAdoptionDecision::ChainMismatch
    );
    // A cold store does not turn a dead chain into an adoption.
    assert_eq!(
        decide_in_memory_prefix_adoption(0, 0),
        InMemoryPrefixAdoptionDecision::ChainMismatch
    );
}

#[test]
fn should_report_a_shorter_retained_state_when_disk_restores_strictly_more() {
    assert_eq!(
        decide_in_memory_prefix_adoption(1_024, 2_048),
        InMemoryPrefixAdoptionDecision::RetainedStateShorter
    );
}
