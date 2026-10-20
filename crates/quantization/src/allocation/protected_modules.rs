//! The set of modules the allocator always upgrades first.

use std::collections::BTreeSet;

/// The KL multiplier that pushes protected modules to the front of the
/// greedy heap without exempting them from the budget.
pub const PROTECTED_MODULE_KL_MULTIPLIER: f64 = 100.0;

/// Union of OptiQ's three protected sets: output layers (embeddings, heads,
/// anything outside `layers.N`), mixture-of-experts routers and shared
/// experts, and, when enabled, the first and last `boundary_block_count`
/// blocks.
///
/// Work to port: classify every name through
/// [`crate::module_taxonomy::classify_module`] and keep the roles
/// [`crate::module_taxonomy::is_always_protected`] accepts; then add
/// [`crate::module_taxonomy::boundary_block_modules`] when
/// `protect_boundary_blocks`.
#[must_use]
pub fn protected_module_names(
    module_names: &[&str],
    protect_boundary_blocks: bool,
    boundary_block_count: usize,
) -> BTreeSet<String> {
    let _ = (module_names, protect_boundary_blocks, boundary_block_count);
    todo!("union of the taxonomy's protected roles and the boundary blocks")
}
