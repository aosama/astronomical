//! The shape rules that decide which tensors become modules: 2-D linears,
//! the fused 3-D expert layouts, and the stacking of unfused per-expert
//! tensors. A module is kept only when its logical last dimension divides
//! the target group size.

use super::super::{ModuleWeightSource, SourceTensorDtype};
use super::{InventoryWalk, LocatedTensor, PerExpertTensor, descriptor_for};

impl InventoryWalk {
    /// Adds the module or modules this tensor holds; false when it holds
    /// none and should pass through.
    pub(super) fn add_modules_for(
        &mut self,
        located_tensor: &LocatedTensor,
        source_dtype: SourceTensorDtype,
    ) -> bool {
        let tensor_name = located_tensor.header_tensor.tensor_name.as_str();
        let Some(logical_shape) = self.logical_shape_of(located_tensor, source_dtype) else {
            return false;
        };
        let Some(last_dimension) = logical_shape.last().copied() else {
            return false;
        };
        if last_dimension % self.group_elements != 0 {
            return false;
        }
        let whole_source = ModuleWeightSource::Whole {
            shard_file_name: located_tensor.shard_file_name.clone(),
            tensor_name: tensor_name.to_owned(),
        };
        match logical_shape.len() {
            2 => match tensor_name.strip_suffix(".weight") {
                Some(module_name) => {
                    self.insert_module(
                        module_name.to_owned(),
                        logical_shape,
                        source_dtype,
                        whole_source,
                    );
                    true
                }
                None => false,
            },
            3 => self.add_three_dimensional_modules(located_tensor, logical_shape, source_dtype),
            _ => false,
        }
    }

    /// The fused expert layouts, in OptiQ's order: DiffusionGemma's direct
    /// mapping, the Hugging Face `experts.gate_up_proj` split and
    /// `experts.down_proj`, then any mlx-lm `switch_mlp`/`switch_glu` weight.
    fn add_three_dimensional_modules(
        &mut self,
        located_tensor: &LocatedTensor,
        logical_shape: Vec<usize>,
        source_dtype: SourceTensorDtype,
    ) -> bool {
        let tensor_name = located_tensor.header_tensor.tensor_name.as_str();
        let name_without_weight = tensor_name.strip_suffix(".weight").unwrap_or(tensor_name);
        let whole_source = ModuleWeightSource::Whole {
            shard_file_name: located_tensor.shard_file_name.clone(),
            tensor_name: tensor_name.to_owned(),
        };
        let is_fused_expert_name = name_without_weight.ends_with(".experts.gate_up_proj")
            || name_without_weight.ends_with(".experts.down_proj");
        if self.naming_rules.maps_fused_experts_directly && is_fused_expert_name {
            self.insert_module(
                name_without_weight.to_owned(),
                logical_shape,
                source_dtype,
                whole_source,
            );
            return true;
        }
        if let Some(experts_base) = tensor_name.strip_suffix(".gate_up_proj") {
            return self.split_fused_gate_up(
                located_tensor,
                experts_base,
                logical_shape,
                source_dtype,
            );
        }
        if let Some(experts_base) = tensor_name.strip_suffix(".down_proj")
            && experts_base.ends_with(".experts")
        {
            let module_name = format!(
                "{}.{}.down_proj",
                self.moe_base(experts_base),
                self.naming_rules.wrapper_segment
            );
            self.insert_module(module_name, logical_shape, source_dtype, whole_source);
            return true;
        }
        let is_wrapped_expert =
            tensor_name.contains(".switch_mlp.") || tensor_name.contains(".switch_glu.");
        if is_wrapped_expert && tensor_name.ends_with(".weight") {
            self.insert_module(
                name_without_weight.to_owned(),
                logical_shape,
                source_dtype,
                whole_source,
            );
            return true;
        }
        false
    }

    /// Splits `[experts, 2 * intermediate, hidden]` into gate and up halves
    /// along the middle axis.
    fn split_fused_gate_up(
        &mut self,
        located_tensor: &LocatedTensor,
        experts_base: &str,
        logical_shape: Vec<usize>,
        source_dtype: SourceTensorDtype,
    ) -> bool {
        if !experts_base.ends_with(".experts") || logical_shape[1] % 2 != 0 {
            return false;
        }
        let half_shape = vec![logical_shape[0], logical_shape[1] / 2, logical_shape[2]];
        let module_base = format!(
            "{}.{}",
            self.moe_base(experts_base),
            self.naming_rules.wrapper_segment
        );
        for (half_index, projection) in [(0_u8, "gate_proj"), (1_u8, "up_proj")] {
            let half_source = ModuleWeightSource::FusedGateUpHalf {
                shard_file_name: located_tensor.shard_file_name.clone(),
                tensor_name: located_tensor.header_tensor.tensor_name.clone(),
                half_index,
            };
            self.insert_module(
                format!("{module_base}.{projection}"),
                half_shape.clone(),
                source_dtype,
                half_source,
            );
        }
        true
    }

    /// Stacks each collected `(base, projection)` group in expert order into
    /// one `[experts, out, in]` module. A group of fewer than two tensors, or
    /// one whose input width does not divide the group size, or one whose
    /// experts disagree on shape, passes through.
    pub(super) fn stack_unfused_experts(&mut self) {
        let unfused_experts = std::mem::take(&mut self.unfused_experts);
        for ((base_name, projection), mut per_expert_tensors) in unfused_experts {
            per_expert_tensors.sort_by_key(expert_index_of);
            let first_shape = per_expert_tensors[0].shape.clone();
            let is_stackable = per_expert_tensors.len() >= 2
                && first_shape.len() == 2
                && first_shape[1] % self.group_elements == 0
                && have_one_shape(&per_expert_tensors, &first_shape);
            if !is_stackable {
                for per_expert_tensor in per_expert_tensors {
                    self.pass_through_tensor_names
                        .push(per_expert_tensor.tensor_name);
                }
                continue;
            }
            let logical_shape = vec![per_expert_tensors.len(), first_shape[0], first_shape[1]];
            let source_dtype = per_expert_tensors[0].source_dtype;
            let mut per_expert_sources = Vec::with_capacity(per_expert_tensors.len());
            for per_expert_tensor in per_expert_tensors {
                per_expert_sources.push((
                    per_expert_tensor.shard_file_name,
                    per_expert_tensor.tensor_name,
                ));
            }
            let module_name = format!(
                "{base_name}.{}.{projection}",
                self.naming_rules.wrapper_segment
            );
            self.insert_module(
                module_name,
                logical_shape,
                source_dtype,
                ModuleWeightSource::StackedPerExpert { per_expert_sources },
            );
        }
    }

    /// The header shape, or for a packed MLX source the logical shape rebuilt
    /// from the scales: `scales.shape[-1] * source_group_size` is the real
    /// input width, where the packed `uint32` width understates it by
    /// `32 / bits` and once made the allocator buy 8-bit experts it could not
    /// afford. `None` when a packed source does not declare its group size.
    fn logical_shape_of(
        &self,
        located_tensor: &LocatedTensor,
        source_dtype: SourceTensorDtype,
    ) -> Option<Vec<usize>> {
        let header_shape = located_tensor.header_tensor.shape.clone();
        if source_dtype != SourceTensorDtype::MlxAffinePacked {
            return Some(header_shape);
        }
        let module_base = located_tensor
            .header_tensor
            .tensor_name
            .strip_suffix(".weight")?;
        let scales_shape = self
            .shape_by_tensor_name
            .get(&format!("{module_base}.scales"))?;
        let source_group_size = self.source_group_sizes.group_size_for(module_base)? as usize;
        let mut logical_shape = header_shape;
        let last_index = logical_shape.len().checked_sub(1)?;
        logical_shape[last_index] = scales_shape.last()? * source_group_size;
        Some(logical_shape)
    }

    /// Qwen drops the `experts` segment when it fuses.
    fn moe_base(&self, experts_base: &str) -> String {
        if self.naming_rules.drops_experts_segment
            && let Some(parent_base) = experts_base.strip_suffix(".experts")
        {
            return parent_base.to_owned();
        }
        experts_base.to_owned()
    }

    fn insert_module(
        &mut self,
        module_name: String,
        logical_shape: Vec<usize>,
        source_dtype: SourceTensorDtype,
        weight_source: ModuleWeightSource,
    ) {
        let descriptor = descriptor_for(
            module_name.clone(),
            logical_shape,
            source_dtype,
            weight_source,
        );
        self.modules.insert(module_name, descriptor);
    }
}

/// True when every expert tensor has `expected_shape`; a mismatched expert
/// would make the stacked shape a lie, so the group passes through instead.
fn have_one_shape(per_expert_tensors: &[PerExpertTensor], expected_shape: &[usize]) -> bool {
    for per_expert_tensor in per_expert_tensors {
        if per_expert_tensor.shape != expected_shape {
            return false;
        }
    }
    true
}

/// Sort key for stacking in expert order.
fn expert_index_of(per_expert_tensor: &PerExpertTensor) -> usize {
    per_expert_tensor.expert_index
}
