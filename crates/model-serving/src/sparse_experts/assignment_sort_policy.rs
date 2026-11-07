//! The sorted-assignment route decision, shared by every model family.
//!
//! Sorting assignments for gathered expert projections pays off only above
//! the 64-assignment geometry floor, and the custom sorted-reduction kernel
//! runs only when this GPU's capability probe retained it. Both conditions
//! live here so a capability demotion cannot accidentally route to the
//! custom kernel.
//!
//! Sorted rows must also not let the multiplied page pick the arithmetic.
//! MLX `GatherQMM` takes its tiled sorted kernel only when the assignment
//! count is at least four times the expert count of the weight tensor it
//! receives, and its per-row vector kernel otherwise; the two round
//! differently. That tensor is whatever page serves the forward: all of a
//! resident layer, a streamed routed union, a warm-table subset, or a complete
//! warm table. Claiming sorted indices to the quantized gather only at four
//! assignments per expert of the whole layer meets MLX's rule for every page
//! the layer can produce, so the kernel depends on the chunk alone. Before
//! this, a short prefill chunk changed kernels as warm tables filled, so the
//! same seeded request gave different answers and paged serving drifted from
//! resident serving (issue #1679).

#[cfg(all(test, feature = "direct-mlx"))]
mod tests;

use super::gathered_projection::ExpertAssignmentOrder;

/// Sorting gathered expert reads pays off only once the assignment set is
/// large enough to group reads by the leading expert axis. Everything in this
/// file executes only in the direct-MLX engine world, where the resident and
/// streaming engines route gathered expert projections.
pub const MINIMUM_SORTED_EXPERT_ASSIGNMENTS: usize = 64;

/// MLX `GatherQMM` runs its tiled sorted kernel only when assignments per
/// expert of the multiplied weight tensor reach this ratio (MLX 0.32.4
/// `backend/metal/quantized.cpp`, `B / E >= 4`).
pub const MLX_TILED_GATHER_MINIMUM_ASSIGNMENTS_PER_EXPERT: usize = 4;

/// Returns whether production may take the sorted custom-kernel reduction
/// route. A capability-demoted kernel always takes the unsorted MLX route,
/// which the direct-MLX contracts prove numerically equal.
#[must_use]
pub(crate) fn should_use_sorted_expert_reduction(
    assignment_count: usize,
    is_sorted_reduction_kernel_available: bool,
) -> bool {
    assignment_count >= MINIMUM_SORTED_EXPERT_ASSIGNMENTS && is_sorted_reduction_kernel_available
}

/// The order both engines declare to the gathered projection.
///
/// `are_expert_indices_sorted` says whether the caller already permuted the
/// IDs and their activation rows by expert; this never sorts anything.
/// `layer_expert_count` is the layer's full expert count, never the size of
/// the page being multiplied: a page never holds more experts than its layer,
/// so meeting MLX's ratio against the layer meets it against every page.
/// Below that floor the rows stay sorted, which keeps the per-row kernel's
/// weight reads grouped, but the quantized gather is not told so. A layer
/// with no experts has no page to vary, so its floor of zero is harmless.
#[must_use]
pub fn expert_assignment_order(
    are_expert_indices_sorted: bool,
    assignment_count: usize,
    layer_expert_count: usize,
) -> ExpertAssignmentOrder {
    if !are_expert_indices_sorted {
        return ExpertAssignmentOrder::Original;
    }
    sorted_expert_assignment_order(assignment_count, layer_expert_count)
}

/// Picks the sorted variant whose quantized-gather claim keeps the kernel
/// independent of the page; see [`expert_assignment_order`].
fn sorted_expert_assignment_order(
    assignment_count: usize,
    layer_expert_count: usize,
) -> ExpertAssignmentOrder {
    let page_independent_tiled_floor =
        layer_expert_count.saturating_mul(MLX_TILED_GATHER_MINIMUM_ASSIGNMENTS_PER_EXPERT);
    if assignment_count >= page_independent_tiled_floor {
        ExpertAssignmentOrder::SortedByExpert
    } else {
        ExpertAssignmentOrder::SortedByExpertBelowTiledFloor
    }
}
