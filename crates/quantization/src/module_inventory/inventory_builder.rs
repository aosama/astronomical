//! The header walk that turns a source directory into an inventory.
//!
//! Ported from OptiQ's `_index_bf16_layers`, widened in two places: sources
//! already quantized in MLX layout are inventoried with their logical shape
//! rebuilt from the scales, and any 3-D `.weight` under an mlx-lm expert
//! wrapper is accepted, so an mlx-lm-format bf16 checkpoint inventories as
//! completely as a Hugging Face one.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use crate::bit_width::GroupSize;
use crate::error::QuantizationError;
use crate::module_taxonomy::{
    classify_module, is_non_language_tower_tensor, transformer_block_index,
};
// The requantizer's router rule, not the allocator taxonomy's tighter one:
// the inventory's pass-through list is exactly what the requantizer copies
// unencoded, so both must agree on what a router is.
use crate::requantize::bit_policy::is_router_tensor_key;

use super::expert_naming::{ExpertNamingRules, SourceQuantizationParameters, read_model_config};
use super::safetensors_header::{HeaderTensor, read_header_tensors};
use super::{
    ModuleWeightSource, PassThroughTensor, QuantizableModuleDescriptor, SourceTensorDtype,
};

/// Everything the walk produces, before it is wrapped in a `ModuleInventory`.
pub struct InventoryContents {
    pub modules: BTreeMap<String, QuantizableModuleDescriptor>,
    pub pass_through_tensors: Vec<PassThroughTensor>,
    pub aliases: BTreeMap<String, String>,
}

/// One tensor's header entry with the shard it lives in.
struct LocatedTensor {
    shard_file_name: String,
    header_tensor: HeaderTensor,
}

/// One unfused per-expert 2-D tensor awaiting stacking.
struct PerExpertTensor {
    expert_index: usize,
    shard_file_name: String,
    tensor_name: String,
    shape: Vec<usize>,
    source_dtype: SourceTensorDtype,
}

/// The walk's state across every shard.
struct InventoryWalk {
    naming_rules: ExpertNamingRules,
    source_quantization_parameters: SourceQuantizationParameters,
    group_elements: usize,
    shape_by_tensor_name: BTreeMap<String, Vec<usize>>,
    modules: BTreeMap<String, QuantizableModuleDescriptor>,
    pass_through_tensors: Vec<PassThroughTensor>,
    unfused_experts: BTreeMap<(String, String), Vec<PerExpertTensor>>,
}

/// Reads `config.json` and every shard header under `source_directory`.
pub fn build_inventory_contents(
    source_directory: &Path,
    group_size: GroupSize,
) -> Result<InventoryContents, QuantizationError> {
    let shard_paths = list_shard_paths(source_directory)?;
    let (naming_rules, source_quantization_parameters) = read_model_config(source_directory)?;
    let located_tensors = read_every_header(&shard_paths)?;
    let mut walk = InventoryWalk {
        naming_rules,
        source_quantization_parameters,
        group_elements: group_size.elements() as usize,
        shape_by_tensor_name: BTreeMap::new(),
        modules: BTreeMap::new(),
        pass_through_tensors: Vec::new(),
        unfused_experts: BTreeMap::new(),
    };
    for located_tensor in &located_tensors {
        let tensor_name = located_tensor.header_tensor.tensor_name.clone();
        walk.shape_by_tensor_name
            .insert(tensor_name, located_tensor.header_tensor.shape.clone());
    }
    for located_tensor in &located_tensors {
        walk.visit(located_tensor);
    }
    walk.stack_unfused_experts();
    walk.keep_siblings_of_passed_through_weights(&located_tensors);
    let aliases = aliases_for(&walk.modules);
    Ok(InventoryContents {
        modules: walk.modules,
        pass_through_tensors: walk.pass_through_tensors,
        aliases,
    })
}

/// Every `*.safetensors` directly under the directory, sorted by name. The
/// walk does not descend: OptiQ's private sidecars live under `optiq/` and
/// are not part of the model's weights.
pub(crate) fn list_shard_paths(source_directory: &Path) -> Result<Vec<PathBuf>, QuantizationError> {
    let no_shards = QuantizationError::NoSafetensorsShards {
        source_directory: source_directory.to_path_buf(),
    };
    let Ok(directory_entries) = std::fs::read_dir(source_directory) else {
        return Err(no_shards);
    };
    let mut shard_paths = Vec::new();
    for directory_entry in directory_entries.flatten() {
        let entry_path = directory_entry.path();
        let is_shard =
            entry_path.extension().and_then(std::ffi::OsStr::to_str) == Some("safetensors");
        if is_shard && entry_path.is_file() {
            shard_paths.push(entry_path);
        }
    }
    if shard_paths.is_empty() {
        return Err(no_shards);
    }
    shard_paths.sort();
    Ok(shard_paths)
}

fn read_every_header(shard_paths: &[PathBuf]) -> Result<Vec<LocatedTensor>, QuantizationError> {
    let mut located_tensors = Vec::new();
    for shard_path in shard_paths {
        let shard_file_name = shard_path
            .file_name()
            .and_then(std::ffi::OsStr::to_str)
            .unwrap_or_default()
            .to_owned();
        for header_tensor in read_header_tensors(shard_path)? {
            located_tensors.push(LocatedTensor {
                shard_file_name: shard_file_name.clone(),
                header_tensor,
            });
        }
    }
    Ok(located_tensors)
}

impl InventoryWalk {
    /// Routes one tensor to a module, to the unfused-expert collection, to
    /// nowhere (a sibling another module consumes), or to pass-through.
    fn visit(&mut self, located_tensor: &LocatedTensor) {
        let tensor_name = located_tensor.header_tensor.tensor_name.as_str();
        if self.is_consumed_sibling(tensor_name) {
            return;
        }
        let Some(source_dtype) = self.source_dtype_of(&located_tensor.header_tensor) else {
            self.pass_through(located_tensor);
            return;
        };
        if is_router_tensor_key(tensor_name) || is_non_language_tower_tensor(tensor_name) {
            self.pass_through(located_tensor);
            return;
        }
        if let Some((base_name, expert_index, projection)) = parse_unfused_expert(tensor_name) {
            let projection = self.renamed_projection(projection);
            let per_expert_tensor = PerExpertTensor {
                expert_index,
                shard_file_name: located_tensor.shard_file_name.clone(),
                tensor_name: tensor_name.to_owned(),
                shape: located_tensor.header_tensor.shape.clone(),
                source_dtype,
            };
            self.unfused_experts
                .entry((base_name, projection))
                .or_default()
                .push(per_expert_tensor);
            return;
        }
        if !self.add_modules_for(located_tensor, source_dtype) {
            self.pass_through(located_tensor);
        }
    }

    /// `.scales`/`.biases` of a packed weight and the `_scale_inv` of an FP8
    /// weight belong to that weight's module, not to the pass-through list.
    fn is_consumed_sibling(&self, tensor_name: &str) -> bool {
        for packed_suffix in [".scales", ".biases"] {
            if let Some(module_base) = tensor_name.strip_suffix(packed_suffix) {
                return self
                    .shape_by_tensor_name
                    .contains_key(&format!("{module_base}.weight"));
            }
        }
        match tensor_name.strip_suffix("_scale_inv") {
            Some(scaled_tensor_name) => self.shape_by_tensor_name.contains_key(scaled_tensor_name),
            None => false,
        }
    }

    /// The source dtype the requantizer will read, or `None` when the tensor
    /// is not a weight it can encode (integer buffers, FP8 without a block
    /// scale beside it, packed words without scales).
    fn source_dtype_of(&self, header_tensor: &HeaderTensor) -> Option<SourceTensorDtype> {
        let tensor_name = header_tensor.tensor_name.as_str();
        match header_tensor.dtype.as_str() {
            "BF16" => Some(SourceTensorDtype::BFloat16),
            "F16" => Some(SourceTensorDtype::Float16),
            "F32" => Some(SourceTensorDtype::Float32),
            "U32" if self.has_packed_scales(tensor_name) => {
                Some(SourceTensorDtype::MlxAffinePacked)
            }
            fp8_dtype
                if (fp8_dtype.starts_with("F8_") || fp8_dtype == "U8")
                    && self
                        .shape_by_tensor_name
                        .contains_key(&format!("{tensor_name}_scale_inv")) =>
            {
                Some(SourceTensorDtype::Float8E4M3WithBlockScale)
            }
            _ => None,
        }
    }

    fn has_packed_scales(&self, tensor_name: &str) -> bool {
        match tensor_name.strip_suffix(".weight") {
            Some(module_base) => self
                .shape_by_tensor_name
                .contains_key(&format!("{module_base}.scales")),
            None => false,
        }
    }

    fn renamed_projection(&self, projection: String) -> String {
        match self
            .naming_rules
            .projection_renames
            .get(projection.as_str())
        {
            Some(renamed_projection) => (*renamed_projection).to_owned(),
            None => projection,
        }
    }

    /// A packed or FP8 weight that became no module (its last dimension does
    /// not divide the group size) is copied as stored, so its `.scales`,
    /// `.biases`, or `_scale_inv`, skipped by the walk as a module's
    /// siblings, must be copied beside it or the written weight is
    /// undecodable.
    fn keep_siblings_of_passed_through_weights(&mut self, located_tensors: &[LocatedTensor]) {
        let mut sibling_names = BTreeSet::new();
        for pass_through_tensor in &self.pass_through_tensors {
            let tensor_name = pass_through_tensor.tensor_name.as_str();
            if let Some(module_base) = tensor_name.strip_suffix(".weight") {
                sibling_names.insert(format!("{module_base}.scales"));
                sibling_names.insert(format!("{module_base}.biases"));
            }
            sibling_names.insert(format!("{tensor_name}_scale_inv"));
        }
        for located_tensor in located_tensors {
            if sibling_names.contains(&located_tensor.header_tensor.tensor_name) {
                self.pass_through(located_tensor);
            }
        }
    }

    fn pass_through(&mut self, located_tensor: &LocatedTensor) {
        self.pass_through_tensors.push(PassThroughTensor {
            shard_file_name: located_tensor.shard_file_name.clone(),
            tensor_name: located_tensor.header_tensor.tensor_name.clone(),
            shape: located_tensor.header_tensor.shape.clone(),
        });
    }
}

/// OptiQ's unfused-expert pattern,
/// `^(.*\.(mlp|feed_forward|mixer))\.experts\.(\d+)\.(gate_proj|up_proj|down_proj|w1|w2|w3)\.weight$`,
/// returning the base, the expert index, and the projection with the
/// Llama-style `w1→gate_proj`, `w2→down_proj`, `w3→up_proj` rename applied.
fn parse_unfused_expert(tensor_name: &str) -> Option<(String, usize, String)> {
    let segments: Vec<&str> = tensor_name.split('.').collect();
    let segment_count = segments.len();
    if segment_count < 6 || segments[segment_count - 1] != "weight" {
        return None;
    }
    let parent_segment = segments[segment_count - 5];
    let is_expert_parent = matches!(parent_segment, "mlp" | "feed_forward" | "mixer");
    let is_experts_segment = segments[segment_count - 4] == "experts";
    let expert_index_segment = segments[segment_count - 3];
    if !is_expert_parent || !is_experts_segment || !is_ascii_digits(expert_index_segment) {
        return None;
    }
    let projection = match segments[segment_count - 2] {
        "w1" | "gate_proj" => "gate_proj",
        "w2" | "down_proj" => "down_proj",
        "w3" | "up_proj" => "up_proj",
        _ => return None,
    };
    let base_name = segments[..segment_count - 4].join(".");
    let expert_index = expert_index_segment.parse().ok()?;
    Some((base_name, expert_index, projection.to_owned()))
}

fn is_ascii_digits(segment: &str) -> bool {
    if segment.is_empty() {
        return false;
    }
    for character in segment.chars() {
        if !character.is_ascii_digit() {
            return false;
        }
    }
    true
}

/// The alternate spellings mlx-lm uses for each canonical module, mapped
/// back to it: `model.` stripped; `model.language_model.` reordered to
/// `language_model.model.`; and flattened to `model.` for adapters whose
/// sanitize flattens the tower (muse_glimmer matched 1 of 417 modules
/// without it). A spelling that is itself a canonical name is never aliased.
fn aliases_for(
    modules: &BTreeMap<String, QuantizableModuleDescriptor>,
) -> BTreeMap<String, String> {
    let mut aliases = BTreeMap::new();
    for canonical_name in modules.keys() {
        let mut spellings = BTreeSet::new();
        if let Some(stripped_name) = canonical_name.strip_prefix("model.") {
            spellings.insert(stripped_name.to_owned());
        }
        if canonical_name.contains("model.language_model.") {
            spellings.insert(canonical_name.replacen(
                "model.language_model.",
                "language_model.model.",
                1,
            ));
            spellings.insert(canonical_name.replacen("model.language_model.", "model.", 1));
        }
        for spelling in spellings {
            if !modules.contains_key(&spelling) {
                aliases.insert(spelling, canonical_name.clone());
            }
        }
    }
    aliases
}

/// Builds a descriptor with its role and block index from the taxonomy.
fn descriptor_for(
    module_name: String,
    logical_shape: Vec<usize>,
    source_dtype: SourceTensorDtype,
    weight_source: ModuleWeightSource,
    source_quantization: Option<(u32, u32)>,
) -> QuantizableModuleDescriptor {
    let logical_parameter_count = logical_shape.iter().product::<usize>() as u64;
    QuantizableModuleDescriptor {
        source_quantization,
        role: classify_module(&module_name),
        transformer_block_index: transformer_block_index(&module_name),
        module_name,
        logical_shape,
        logical_parameter_count,
        source_dtype,
        weight_source,
    }
}

#[path = "module_shapes.rs"]
mod module_shapes;
