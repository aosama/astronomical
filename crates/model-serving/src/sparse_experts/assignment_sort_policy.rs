//! The sorted-assignment route decision, shared by every model family.
//!
//! Sorting assignments for gathered expert projections pays off only above
//! the 64-assignment geometry floor, and the custom sorted-reduction kernel
//! runs only when this GPU's capability probe retained it. Both conditions
//! live here so a capability demotion cannot accidentally route to the
//! custom kernel.

/// Sorting gathered expert reads pays off only once the assignment set is
/// large enough to group reads by the leading expert axis.
pub const MINIMUM_SORTED_EXPERT_ASSIGNMENTS: usize = 64;

/// Returns whether production may take the sorted custom-kernel reduction
/// route. A capability-demoted kernel always takes the unsorted MLX route,
/// which the direct-MLX contracts prove numerically equal.
#[must_use]
pub fn should_use_sorted_expert_reduction(
    assignment_count: usize,
    is_sorted_reduction_kernel_available: bool,
) -> bool {
    assignment_count >= MINIMUM_SORTED_EXPERT_ASSIGNMENTS && is_sorted_reduction_kernel_available
}
