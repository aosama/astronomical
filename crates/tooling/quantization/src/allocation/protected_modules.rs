//! The set of modules the allocator always upgrades first.

use std::collections::BTreeSet;

use crate::module_taxonomy::{boundary_block_modules, classify_module, is_always_protected};

/// The KL multiplier that pushes protected modules to the front of the
/// greedy heap without exempting them from the budget.
pub const PROTECTED_MODULE_KL_MULTIPLIER: f64 = 100.0;

/// Union of OptiQ's three protected sets: output layers (embeddings, heads,
/// anything outside `layers.N`), mixture-of-experts routers and shared
/// experts, and, when enabled, the first and last `boundary_block_count`
/// blocks.
///
/// The taxonomy owns the name rules, so this is only a classification pass
/// plus the boundary edge set: every name is classified and kept when
/// [`crate::module_taxonomy::is_always_protected`] accepts its role, then the
/// [`crate::module_taxonomy::boundary_block_modules`] edge set is added when
/// `protect_boundary_blocks`. Classifying through the shared taxonomy (rather
/// than re-deriving the keywords here) is what keeps the allocator's protected
/// set identical to the static scorer's and the shard policy's.
#[must_use]
pub fn protected_module_names(
    module_names: &[&str],
    protect_boundary_blocks: bool,
    boundary_block_count: usize,
) -> BTreeSet<String> {
    let mut protected: BTreeSet<String> = BTreeSet::new();
    for &name in module_names {
        if is_always_protected(classify_module(name)) {
            protected.insert(name.to_owned());
        }
    }
    if protect_boundary_blocks {
        for name in boundary_block_modules(module_names, boundary_block_count) {
            protected.insert(name.to_owned());
        }
    }
    protected
}
