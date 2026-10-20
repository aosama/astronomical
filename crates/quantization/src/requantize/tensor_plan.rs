//! The ordered list of tensors the requantizer will visit, decided before
//! any tensor data is read.
//!
//! Planning up front (rather than deciding per tensor inside the walk, as
//! the Python tool does) gives the progress reporter a true total, lets the
//! CLI print the plan as a dry run, and lets the pipeline compute the
//! expected output size before committing hours of GPU time.

use crate::module_inventory::ModuleInventory;

use super::bit_policy::{BitPolicy, TensorEncodingDecision};

#[cfg(test)]
#[path = "tensor_plan_tests.rs"]
mod tensor_plan_tests;

/// One tensor's visit.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TensorPlanEntry {
    pub shard_file_name: String,
    pub tensor_key: String,
    pub shape: Vec<usize>,
    /// Present when the source tensor is already affine-quantized and must
    /// be dequantized with these parameters first.
    pub source_quantization: Option<(u32, u32)>,
    pub decision: TensorEncodingDecision,
    /// Bytes this tensor will occupy on disk after the decision.
    pub estimated_output_bytes: u64,
}

/// The complete plan.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RequantizeTensorPlan {
    pub entries: Vec<TensorPlanEntry>,
}

impl RequantizeTensorPlan {
    /// Builds the plan from the inventory and the policy.
    ///
    /// Work to port: visit every tensor of every shard in sorted order
    /// (`.scales`/`.biases` siblings are folded into their `.weight` entry);
    /// for a quantizable module apply `policy.decide` then
    /// [`BitPolicy::adjust_for_shape`]; everything in the inventory's
    /// pass-through list is `PassThrough`. A mixed-precision source carries
    /// per-module entries beside the global defaults in its `config.json`,
    /// and they are not decoration: dequantizing an 8-bit tensor with the
    /// global 4 fails outright, so `source_quantization` is resolved
    /// per module from the source config's overrides.
    #[must_use]
    pub fn from_inventory(inventory: &ModuleInventory, policy: &BitPolicy) -> Self {
        let _ = (inventory, policy);
        todo!("walk the inventory, decide per tensor, and estimate output bytes")
    }

    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.entries.len()
    }

    #[must_use]
    pub fn quantized_tensor_count(&self) -> usize {
        self.entries
            .iter()
            .filter(|entry| matches!(entry.decision, TensorEncodingDecision::Encode { .. }))
            .count()
    }

    #[must_use]
    pub fn estimated_output_bytes(&self) -> u64 {
        self.entries
            .iter()
            .map(|entry| entry.estimated_output_bytes)
            .sum()
    }

    /// Bytes of packed weight, scales, and biases a tensor of `shape`
    /// occupies at `(bits, group_size)`, or its bfloat16 size when passed
    /// through. Scales and biases are 2 bytes each per group.
    #[must_use]
    pub fn estimate_output_bytes(shape: &[usize], decision: TensorEncodingDecision) -> u64 {
        let _ = (shape, decision);
        todo!(
            "elements * bits / 8 + 2 * 2 * elements / group_size, or elements * 2 for pass-through"
        )
    }
}
