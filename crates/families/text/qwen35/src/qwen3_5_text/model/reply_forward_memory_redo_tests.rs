use super::TextExecutionError;
use super::refused_page_shortfall_bytes;
use crate::MlxAllocationAdmissionError;
use engine_core::memory::MemoryBoundary;

fn refusal_short_by(shortfall_bytes: u64) -> TextExecutionError {
    TextExecutionError::MemoryAdmissionRejected(MlxAllocationAdmissionError::Rejected {
        stage: "rust_streamed_expert_layer_39 (26 experts, 4 routed tokens, Decode)".to_owned(),
        boundary: MemoryBoundary::AllocationProjection,
        shortfall_bytes,
        active_memory_bytes: 8_514_019_578,
        pending_allocation_bytes: 86_900_736,
        active_memory_ceiling_bytes: 8_589_934_592,
    })
}

#[test]
fn should_read_the_shortfall_of_a_refused_page() {
    assert_eq!(
        refused_page_shortfall_bytes(&refusal_short_by(10_985_722)),
        Some(10_985_722)
    );
}

#[test]
fn should_read_a_refusal_short_by_nothing_as_a_refusal() {
    assert_eq!(refused_page_shortfall_bytes(&refusal_short_by(0)), Some(0));
}

#[test]
fn should_not_treat_any_other_error_as_a_refused_page() {
    let invalid_input = TextExecutionError::InvalidInput {
        description: "an exact forward reported experts missing from memory",
    };
    assert_eq!(refused_page_shortfall_bytes(&invalid_input), None);
}
