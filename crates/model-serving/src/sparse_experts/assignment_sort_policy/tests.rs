//! Unit contracts for the sorted-expert-reduction capability seam and the
//! assignment order both engines declare to the gathered projection.
//!
//! Production must take the sorted custom-kernel route only when both the
//! assignment geometry and the GPU capability allow it. A demoted kernel
//! silently falls back to the unsorted MLX route for every forward, which the
//! existing direct-MLX contracts prove numerically equal.

use super::{expert_assignment_order, should_use_sorted_expert_reduction};
use crate::sparse_experts::ExpertAssignmentOrder;

/// The Qwen 3.5 35B-A3B layer: 256 experts, eight routed per token.
const QWEN3_5_35B_A3B_LAYER_EXPERT_COUNT: usize = 256;

#[test]
fn should_sort_large_assignment_sets_when_the_sorted_kernel_is_supported() {
    assert!(should_use_sorted_expert_reduction(64, true));
    assert!(should_use_sorted_expert_reduction(1_000, true));
}

#[test]
fn should_fall_back_to_the_unsorted_route_when_the_sorted_kernel_is_demoted() {
    assert!(
        !should_use_sorted_expert_reduction(64, false),
        "a capability-demoted kernel must never route to the sorted path regardless of assignment count"
    );
    assert!(!should_use_sorted_expert_reduction(10_000, false));
}

#[test]
fn should_keep_small_assignment_sets_unsorted_even_with_a_supported_kernel() {
    assert!(!should_use_sorted_expert_reduction(0, true));
    assert!(!should_use_sorted_expert_reduction(63, true));
}

#[test]
fn should_claim_sorted_ids_to_the_quantized_gather_from_four_assignments_per_layer_expert() {
    for assignment_count in [1_024, 8_192] {
        let assignment_order =
            expert_assignment_order(true, assignment_count, QWEN3_5_35B_A3B_LAYER_EXPERT_COUNT);
        assert_eq!(assignment_order, ExpertAssignmentOrder::SortedByExpert);
        assert!(assignment_order.uses_sorted_indices_for_quantized_gather());
        assert!(assignment_order.uses_sorted_indices());
    }
}

#[test]
fn should_withhold_the_sorted_claim_from_the_quantized_gather_below_the_floor() {
    // 384 assignments is a 48-token chunk at eight experts per token. Against
    // a 256-expert layer, MLX would take the tiled kernel for a small streamed
    // page but the vector kernel for the complete table, so claiming sorted
    // IDs here would let the page decide the arithmetic.
    for assignment_count in [64, 384, 1_023] {
        let assignment_order =
            expert_assignment_order(true, assignment_count, QWEN3_5_35B_A3B_LAYER_EXPERT_COUNT);
        assert_eq!(
            assignment_order,
            ExpertAssignmentOrder::SortedByExpertBelowTiledFloor
        );
        assert!(!assignment_order.uses_sorted_indices_for_quantized_gather());
        assert!(
            assignment_order.uses_sorted_indices(),
            "the dense gather picks its kernel from the flag alone and keeps the claim"
        );
    }
}

#[test]
fn should_declare_router_order_for_unsorted_ids_at_any_size() {
    for assignment_count in [0, 384, 8_192] {
        let assignment_order =
            expert_assignment_order(false, assignment_count, QWEN3_5_35B_A3B_LAYER_EXPERT_COUNT);
        assert_eq!(assignment_order, ExpertAssignmentOrder::Original);
        assert!(!assignment_order.uses_sorted_indices());
        assert!(!assignment_order.uses_sorted_indices_for_quantized_gather());
    }
}

#[test]
fn should_claim_sorted_ids_for_a_layer_without_experts() {
    // No experts means no page can vary, so the floor of zero is harmless.
    assert_eq!(
        expert_assignment_order(true, 0, 0),
        ExpertAssignmentOrder::SortedByExpert
    );
}

#[test]
fn should_saturate_instead_of_overflowing_on_an_absurd_expert_count() {
    assert_eq!(
        expert_assignment_order(true, usize::MAX - 1, usize::MAX),
        ExpertAssignmentOrder::SortedByExpertBelowTiledFloor
    );
}
