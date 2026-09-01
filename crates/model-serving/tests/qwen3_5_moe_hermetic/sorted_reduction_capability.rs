//! Hermetic contracts for the sorted-expert-reduction capability seam.
//!
//! Production must take the sorted custom-kernel route only when both the
//! assignment geometry and the GPU capability allow it. A demoted kernel
//! silently falls back to the unsorted MLX route for every forward, which the
//! existing direct-MLX contracts prove numerically equal.

use astronomical_model_serving::should_use_sorted_expert_reduction;

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
