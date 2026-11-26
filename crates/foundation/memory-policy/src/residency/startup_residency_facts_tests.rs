//! Unit contracts for the startup complete-residency decision.

use super::startup_residency_facts::StartupResidencyFacts;
use crate::admission::complete_residency::CompleteResidencyDecision;

/// Core 2.6 GB + complete experts 24.8 GB + startup headroom 0.9 GB: the
/// large sparse mixture-of-experts shape the expert-paging journeys load.
const LARGE_SPARSE_MOE_FACTS: StartupResidencyFacts = StartupResidencyFacts {
    model_core_payload_bytes: 2_600_000_000,
    complete_expert_payload_bytes: 24_800_000_000,
    startup_complete_residency_headroom_bytes: 900_000_000,
};

const DENSE_TEXT_WEIGHT_PAYLOAD_BYTES: u64 = 7_276_225_536;

fn admits_complete_residency(
    startup_facts: StartupResidencyFacts,
    active_memory_ceiling_bytes: u64,
) -> bool {
    matches!(
        startup_facts.startup_complete_residency_decision(active_memory_ceiling_bytes),
        CompleteResidencyDecision::Admit { .. }
    )
}

#[test]
fn should_admit_complete_residency_when_the_ceiling_covers_payload_and_headroom() {
    // 38 GB (2^30 convention) comfortably covers 28.3 GB of projected
    // payload plus headroom.
    assert!(admits_complete_residency(
        LARGE_SPARSE_MOE_FACTS,
        38 * 1_073_741_824
    ));
}

#[test]
fn should_refuse_complete_residency_one_byte_below_payload_and_headroom() {
    let payload_with_headroom_bytes = LARGE_SPARSE_MOE_FACTS.model_core_payload_bytes
        + LARGE_SPARSE_MOE_FACTS.complete_expert_payload_bytes
        + LARGE_SPARSE_MOE_FACTS.startup_complete_residency_headroom_bytes;
    assert!(admits_complete_residency(
        LARGE_SPARSE_MOE_FACTS,
        payload_with_headroom_bytes
    ));
    assert!(!admits_complete_residency(
        LARGE_SPARSE_MOE_FACTS,
        payload_with_headroom_bytes - 1
    ));
}

#[test]
fn should_refuse_complete_residency_for_a_ceiling_that_only_fits_the_payload() {
    // 27.4 GB covers the projected payload but not the activation headroom,
    // which is the exact squeeze the startup decision must reject.
    assert!(!admits_complete_residency(
        LARGE_SPARSE_MOE_FACTS,
        27_400_000_000
    ));
}

#[test]
fn should_admit_a_dense_model_at_a_ceiling_equal_to_its_text_weights() {
    let dense_startup_facts =
        StartupResidencyFacts::for_dense_text_weights(DENSE_TEXT_WEIGHT_PAYLOAD_BYTES);

    assert!(admits_complete_residency(
        dense_startup_facts,
        36_500_000_000
    ));
    assert!(admits_complete_residency(
        dense_startup_facts,
        DENSE_TEXT_WEIGHT_PAYLOAD_BYTES
    ));
}

#[test]
fn should_refuse_a_dense_model_at_a_ceiling_one_byte_short_of_its_text_weights() {
    let dense_startup_facts =
        StartupResidencyFacts::for_dense_text_weights(DENSE_TEXT_WEIGHT_PAYLOAD_BYTES);

    assert!(!admits_complete_residency(
        dense_startup_facts,
        DENSE_TEXT_WEIGHT_PAYLOAD_BYTES - 1
    ));
}

#[test]
fn should_give_dense_facts_no_experts_and_no_headroom() {
    let dense_startup_facts =
        StartupResidencyFacts::for_dense_text_weights(DENSE_TEXT_WEIGHT_PAYLOAD_BYTES);

    assert_eq!(
        dense_startup_facts.model_core_payload_bytes,
        DENSE_TEXT_WEIGHT_PAYLOAD_BYTES
    );
    assert_eq!(dense_startup_facts.complete_expert_payload_bytes, 0);
    assert_eq!(
        dense_startup_facts.startup_complete_residency_headroom_bytes,
        0
    );
}
