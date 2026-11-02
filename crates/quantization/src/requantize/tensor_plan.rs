//! The ordered list of tensors the requantizer will visit, decided before
//! any tensor data is read.
//!
//! Planning up front (rather than deciding per tensor inside the walk, as
//! the Python tool does) gives the progress reporter a true total, lets the
//! CLI print the plan as a dry run, and lets the pipeline compute the
//! expected output size before committing hours of GPU time. The per-tensor
//! decisions are OptiQ's `requantize_shardwise` rules
//! (`optiq/core/shardwise.py`), moved ahead of the walk.

use crate::module_inventory::{ModuleInventory, ModuleWeightSource, QuantizableModuleDescriptor};

use super::bit_policy::{BitPolicy, TensorEncodingDecision};

#[cfg(test)]
#[path = "tensor_plan_tests.rs"]
mod tensor_plan_tests;

/// One tensor's visit.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TensorPlanEntry {
    /// The shard holding the source (the first shard for a stacked module).
    pub shard_file_name: String,
    /// The output key: the module's mlx-lm name plus `.weight`, or the
    /// pass-through tensor's own key.
    pub tensor_key: String,
    pub shape: Vec<usize>,
    /// Present when the source tensor is already affine-quantized and must
    /// be dequantized with these parameters first.
    pub source_quantization: Option<(u32, u32)>,
    pub decision: TensorEncodingDecision,
    /// Bytes this tensor will occupy on disk after the decision.
    pub estimated_output_bytes: u64,
    /// The module and its on-disk layout (fused half, stacked experts, FP8,
    /// packed), so the driver can assemble the weight without the
    /// inventory; `None` for a pass-through tensor.
    pub module: Option<QuantizableModuleDescriptor>,
}

/// The complete plan.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RequantizeTensorPlan {
    pub entries: Vec<TensorPlanEntry>,
}

impl RequantizeTensorPlan {
    /// Builds the plan from the inventory and the policy, in shard then key
    /// order so the driver maps each shard once.
    ///
    /// Each module is decided by `policy.decide` on its output key, then
    /// [`BitPolicy::adjust_for_shape`] on its logical shape. A packed
    /// source's `.scales` and `.biases` are folded into the module entry, and
    /// its `source_quantization` comes from the inventory, which resolved it
    /// per module from the source config's overrides. Everything on the
    /// inventory's pass-through list is `PassThrough`.
    #[must_use]
    pub fn from_inventory(inventory: &ModuleInventory, policy: &BitPolicy) -> Self {
        let mut entries = Vec::new();
        for descriptor in inventory.modules() {
            entries.push(module_entry(descriptor, policy));
        }
        for pass_through_tensor in inventory.pass_through_tensors() {
            entries.push(TensorPlanEntry {
                shard_file_name: pass_through_tensor.shard_file_name.clone(),
                tensor_key: pass_through_tensor.tensor_name.clone(),
                shape: pass_through_tensor.shape.clone(),
                source_quantization: None,
                decision: TensorEncodingDecision::PassThrough,
                estimated_output_bytes: Self::estimate_output_bytes(
                    &pass_through_tensor.shape,
                    TensorEncodingDecision::PassThrough,
                ),
                module: None,
            });
        }
        entries.sort_by(compare_by_shard_then_key);
        Self { entries }
    }

    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.entries.len()
    }

    #[must_use]
    pub fn quantized_tensor_count(&self) -> usize {
        let mut quantized_tensor_count = 0;
        for entry in &self.entries {
            if matches!(entry.decision, TensorEncodingDecision::Encode { .. }) {
                quantized_tensor_count += 1;
            }
        }
        quantized_tensor_count
    }

    #[must_use]
    pub fn estimated_output_bytes(&self) -> u64 {
        let mut estimated_output_bytes = 0;
        for entry in &self.entries {
            estimated_output_bytes += entry.estimated_output_bytes;
        }
        estimated_output_bytes
    }

    /// Bytes of packed weight, scales, and biases a tensor of `shape`
    /// occupies at `(bits, group_size)`, or its bfloat16 size when passed
    /// through. Scales and biases are 2 bytes each per group.
    #[must_use]
    pub fn estimate_output_bytes(shape: &[usize], decision: TensorEncodingDecision) -> u64 {
        let element_count: u64 = shape.iter().product::<usize>() as u64;
        match decision {
            TensorEncodingDecision::Encode {
                bit_width,
                group_size,
            } => {
                let packed_weight_bytes = element_count * u64::from(bit_width.bits()) / 8;
                let group_count = element_count / u64::from(group_size.elements());
                packed_weight_bytes + group_count * SCALE_AND_BIAS_BYTES_PER_GROUP
            }
            TensorEncodingDecision::PassThrough => element_count * BFLOAT16_BYTES,
        }
    }
}

/// A bfloat16 scale plus a bfloat16 bias per group.
const SCALE_AND_BIAS_BYTES_PER_GROUP: u64 = 4;

/// Pass-through floats are written at bfloat16.
const BFLOAT16_BYTES: u64 = 2;

/// The plan entry for one module.
fn module_entry(descriptor: &QuantizableModuleDescriptor, policy: &BitPolicy) -> TensorPlanEntry {
    let tensor_key = format!("{}.weight", descriptor.module_name);
    let planned_decision = policy.decide(&tensor_key);
    let decision = BitPolicy::adjust_for_shape(planned_decision, &descriptor.logical_shape);
    TensorPlanEntry {
        shard_file_name: first_shard_of(&descriptor.weight_source),
        tensor_key,
        shape: descriptor.logical_shape.clone(),
        source_quantization: descriptor.source_quantization,
        decision,
        estimated_output_bytes: RequantizeTensorPlan::estimate_output_bytes(
            &descriptor.logical_shape,
            decision,
        ),
        module: Some(descriptor.clone()),
    }
}

/// The shard a module's weight is read from first.
fn first_shard_of(weight_source: &ModuleWeightSource) -> String {
    match weight_source {
        ModuleWeightSource::Whole {
            shard_file_name, ..
        }
        | ModuleWeightSource::FusedGateUpHalf {
            shard_file_name, ..
        } => shard_file_name.clone(),
        ModuleWeightSource::StackedPerExpert { per_expert_sources } => per_expert_sources
            .first()
            .map(first_of_pair)
            .unwrap_or_default(),
    }
}

/// The shard name of a `(shard, tensor)` pair.
fn first_of_pair(shard_and_tensor: &(String, String)) -> String {
    shard_and_tensor.0.clone()
}

/// Shard order first, so the driver maps each shard once, then key order
/// within a shard for a deterministic output.
fn compare_by_shard_then_key(
    left: &TensorPlanEntry,
    right: &TensorPlanEntry,
) -> std::cmp::Ordering {
    (left.shard_file_name.as_str(), left.tensor_key.as_str())
        .cmp(&(right.shard_file_name.as_str(), right.tensor_key.as_str()))
}
