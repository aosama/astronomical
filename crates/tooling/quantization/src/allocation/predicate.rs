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
        // Index the decisions by canonical name for the exact-match lookup.
        let decisions_by_module_name: BTreeMap<String, ModuleQuantizationDecision> = allocation
            .decisions
            .iter()
            .map(|decision| (decision.module_name.clone(), decision.clone()))
            .collect();

        // Default width: the caller's override, else the allocation's highest
        // assigned width (OptiQ falls back to 4 bits for an empty allocation).
        // Unmatched modules are typically critical (embeddings, an output head
        // outside the sweep), so the safest available width is the default.
        let default_bit_width = default_bit_width.unwrap_or_else(|| {
            allocation
                .decisions
                .iter()
                .map(|decision| decision.bit_width)
                .max()
                .unwrap_or_else(|| BitWidth::new(4).expect("4 is a valid width"))
        });

        // Default group size: the most common one in the allocation, ties
        // broken to the first seen (matching OptiQ's `Counter.most_common`),
        // else 64 for an empty allocation.
        let mut group_counts: BTreeMap<GroupSize, (usize, usize)> = BTreeMap::new();
        for (index, decision) in allocation.decisions.iter().enumerate() {
            let entry = group_counts
                .entry(decision.group_size)
                .or_insert((0, index));
            entry.0 += 1;
        }
        let default_group_size = group_counts
            .iter()
            // Highest count wins; on a tie the smaller first-seen index wins
            // (`b.1.1.cmp(&a.1.1)` reverses the order so the earlier one is max).
            .max_by(|a, b| a.1.0.cmp(&b.1.0).then_with(|| b.1.1.cmp(&a.1.1)))
            .map(|(group_size, _)| *group_size)
            .unwrap_or_else(|| GroupSize::new(64).expect("64 is a valid group size"));

        QuantizationPredicate {
            decisions_by_module_name,
            default_bit_width,
            default_group_size,
        }
    }

    /// Looks a module path up.
    ///
    /// Ported from OptiQ's `make_quant_predicate`:
    /// 1. Exact match on the canonical name.
    /// 2. Suffix match: compare with `model.` stripped from both sides, and
    ///    accept `path.ends_with(name)` or `name.ends_with(path)`, because
    ///    mlx-lm paths can differ from PyTorch paths by a prefix.
    /// 3. Unmatched: the default width and group size; a default of 16 means
    ///    keep bfloat16.
    ///
    /// A matched decision at the keep-bfloat16 tier returns `KeepBfloat16`.
    #[must_use]
    pub fn decision_for(&self, module_path: &str) -> QuantizationDecisionForModule {
        // 1. Exact match on the canonical name.
        if let Some(decision) = self.decisions_by_module_name.get(module_path) {
            return emit(decision);
        }
        // 2. Suffix match. mlx-lm paths can differ from the sweep's PyTorch
        //    paths by a prefix, so compare with every "model." stripped from
        //    both sides and accept an either-side suffix relationship.
        let clean_path = module_path.replace("model.", "");
        for (name, decision) in &self.decisions_by_module_name {
            let clean_name = name.replace("model.", "");
            if clean_path == clean_name
                || module_path.ends_with(name.as_str())
                || name.ends_with(module_path)
            {
                return emit(decision);
            }
        }
        // 3. Unmatched: the default width and group size.
        if self.default_bit_width.keeps_bfloat16() {
            QuantizationDecisionForModule::KeepBfloat16
        } else {
            QuantizationDecisionForModule::Quantize {
                bit_width: self.default_bit_width,
                group_size: self.default_group_size,
            }
        }
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

/// Maps one decision to the mlx-lm predicate outcome: the keep-bfloat16 tier
/// (16 bits is not a quantized format, it is the lossless tier) tells the
/// requantizer to leave the tensor at its source dtype, everything else emits
/// its width and group size.
fn emit(decision: &ModuleQuantizationDecision) -> QuantizationDecisionForModule {
    if decision.bit_width.keeps_bfloat16() {
        QuantizationDecisionForModule::KeepBfloat16
    } else {
        QuantizationDecisionForModule::Quantize {
            bit_width: decision.bit_width,
            group_size: decision.group_size,
        }
    }
}
