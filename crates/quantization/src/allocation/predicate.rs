//! The allocation as a lookup: module path in, bits and group size (or
//! "leave at bfloat16") out.
//!
//! OptiQ hands mlx-lm a `quant_predicate(path, module)` closure. The port
//! keeps the same matching rules so an allocation computed against one
//! naming convention still lands on the right tensors when the source shards
//! spell paths slightly differently.

use std::collections::BTreeMap;

use crate::bit_width::{BitWidth, GroupSize};

use super::{AllocationResult, ModuleQuantizationDecision};

/// What to do with one module.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QuantizationDecisionForModule {
    /// Encode at these parameters.
    Quantize {
        bit_width: BitWidth,
        group_size: GroupSize,
    },
    /// Leave the tensor at its source dtype (the keep-bfloat16 tier).
    KeepBfloat16,
}

/// The allocation indexed for matching.
#[derive(Clone, Debug)]
pub struct QuantizationPredicate {
    decisions_by_module_name: BTreeMap<String, ModuleQuantizationDecision>,
    /// Bits for modules the sensitivity analysis never saw (tied
    /// embeddings, an output head outside the sweep). Defaults to the
    /// allocation's highest width: unmatched modules are typically critical.
    default_bit_width: BitWidth,
    /// The most common group size in the allocation.
    default_group_size: GroupSize,
}

impl QuantizationPredicate {
    /// Builds the predicate from an allocation. `default_bit_width` of
    /// `None` selects the allocation's highest width (4 bits for an empty
    /// allocation, OptiQ's fallback).
    #[must_use]
    pub fn from_allocation(
        allocation: &AllocationResult,
        default_bit_width: Option<BitWidth>,
    ) -> Self {
        let _ = (allocation, default_bit_width);
        todo!("index decisions by name, derive the default width and the modal group size")
    }

    /// Looks a module path up.
    ///
    /// Work to port (OptiQ's `make_quant_predicate`):
    /// 1. Exact match on the canonical name.
    /// 2. Suffix match: compare with `model.` stripped from both sides, and
    ///    accept `path.ends_with(name)` or `name.ends_with(path)`, because
    ///    mlx-lm paths can differ from PyTorch paths by a prefix.
    /// 3. Unmatched: the default width and group size; a default of 16 means
    ///    keep bfloat16.
    /// A matched decision at the keep-bfloat16 tier returns `KeepBfloat16`.
    #[must_use]
    pub fn decision_for(&self, module_path: &str) -> QuantizationDecisionForModule {
        let _ = module_path;
        todo!("exact, then suffix, then default as documented above")
    }

    #[must_use]
    pub const fn default_bit_width(&self) -> BitWidth {
        self.default_bit_width
    }

    #[must_use]
    pub const fn default_group_size(&self) -> GroupSize {
        self.default_group_size
    }

    #[must_use]
    pub fn decision_count(&self) -> usize {
        self.decisions_by_module_name.len()
    }
}
