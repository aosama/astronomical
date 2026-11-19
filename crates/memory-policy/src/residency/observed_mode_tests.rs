use astronomical_ipc_protocol::ExpertMemoryMode;

use crate::residency::observed_mode::observed_expert_memory_mode;

#[test]
fn should_label_resident_when_no_expert_storage_bytes_were_read() {
    for engine_ownership_classification in [
        ExpertMemoryMode::Resident,
        ExpertMemoryMode::Hybrid,
        ExpertMemoryMode::Paged,
    ] {
        assert_eq!(
            observed_expert_memory_mode(engine_ownership_classification, 0),
            ExpertMemoryMode::Resident
        );
    }
}

#[test]
fn should_label_hybrid_when_hybrid_ownership_read_expert_storage_bytes() {
    assert_eq!(
        observed_expert_memory_mode(ExpertMemoryMode::Hybrid, 1),
        ExpertMemoryMode::Hybrid
    );
}

#[test]
fn should_label_paged_when_paged_ownership_read_expert_storage_bytes() {
    assert_eq!(
        observed_expert_memory_mode(ExpertMemoryMode::Paged, 4096),
        ExpertMemoryMode::Paged
    );
}

#[test]
fn should_defensively_label_hybrid_when_resident_claim_read_expert_storage_bytes() {
    // A resident ownership claim that still read expert storage contradicts
    // itself; the honest label for that request is hybrid, never resident.
    assert_eq!(
        observed_expert_memory_mode(ExpertMemoryMode::Resident, 4096),
        ExpertMemoryMode::Hybrid
    );
}

#[test]
fn should_label_resident_for_a_single_byte_boundary_read() {
    // The evidence threshold is any observed byte at all, not a page size:
    // one byte of expert storage traffic is already streaming evidence.
    assert_eq!(
        observed_expert_memory_mode(ExpertMemoryMode::Paged, 1),
        ExpertMemoryMode::Paged
    );
}
