//! The list of group-quantizable modules in a source artifact, read from
//! safetensors headers alone.
//!
//! OptiQ obtains this list three different ways (walking an instantiated
//! mlx-lm model, walking a uniform-4-bit baseline, or indexing bf16 shards
//! for the streaming reference), and each way grew its own naming
//! corrections. The port reads the headers once and produces one inventory
//! that every later stage shares: the static scorer, the exact sweep's module
//! list, the allocator's parameter counts, and the requantizer's tensor plan.
//! Reading headers only means a 300B checkpoint is inventoried in
//! milliseconds with no tensor data touched.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::bit_width::GroupSize;
use crate::error::QuantizationError;
use crate::module_taxonomy::{ModuleRole, is_embedding_table_module};

use output_layout::OutputValueTransform;

mod expert_naming;
pub(crate) mod inventory_builder;
pub mod output_layout;
pub(crate) mod safetensors_header;
pub mod traversal_order;

#[cfg(test)]
#[path = "module_inventory_tests.rs"]
mod module_inventory_tests;

/// The on-disk element type of a source tensor.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum SourceTensorDtype {
    BFloat16,
    Float16,
    Float32,
    /// Block-scaled 8-bit float with a `<key>_scale_inv` tensor beside it
    /// (Qwen3.8-Flash-Next-FP8 style). Dequantized to bf16 on read.
    Float8E4M3WithBlockScale,
    /// Already affine-quantized in MLX layout (packed `uint32` weight with
    /// `.scales` and `.biases` siblings). The requantizer dequantizes with the
    /// source's own bits and group size before re-encoding.
    MlxAffinePacked,
}

/// Where one logical module's weight comes from on disk.
///
/// A module is usually one tensor, but mixture-of-experts sources are not
/// uniform: Gemma-4 ships a fused `experts.gate_up_proj` that splits into two
/// modules along the middle axis, Laguna and DeepSeek ship one 2-D tensor per
/// expert that must be stacked into one `[n_experts, out, in]` module, and
/// mlx-lm-format sources already carry the fused `switch_mlp.*` layout.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum ModuleWeightSource {
    /// One tensor, used whole.
    Whole {
        shard_file_name: String,
        tensor_name: String,
    },
    /// The lower or upper half of a fused gate-up tensor along axis 1.
    FusedGateUpHalf {
        shard_file_name: String,
        tensor_name: String,
        /// Zero for `gate_proj`, one for `up_proj`.
        half_index: u8,
    },
    /// Per-expert 2-D tensors stacked along a new leading axis in expert
    /// order; entries may span several shards.
    StackedPerExpert {
        per_expert_sources: Vec<(String, String)>,
    },
}

/// One group-quantizable module and everything later stages need to know
/// about it without re-reading any header.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct QuantizableModuleDescriptor {
    /// The canonical mlx-lm module path (for example
    /// `model.layers.3.mlp.switch_mlp.gate_proj`), which is also the key the
    /// output `config.json` and `optiq/metadata.json` use.
    pub module_name: String,
    /// The logical weight shape after any split or stack, `[out, in]` or
    /// `[n_experts, out, in]`.
    pub logical_shape: Vec<usize>,
    /// The logical parameter count. For an already-quantized source this is
    /// rebuilt from the scales tensor (`scales.shape[-1] * group_size` gives
    /// the real input width), never from the packed `uint32` array: measuring
    /// the packed array understates every quantized module by `32 / bits` and
    /// made the allocator buy 8-bit experts it could not afford.
    pub logical_parameter_count: u64,
    pub source_dtype: SourceTensorDtype,
    pub weight_source: ModuleWeightSource,
    pub role: ModuleRole,
    pub transformer_block_index: Option<usize>,
    /// `(bits, group_size)` an already-quantized source was written with,
    /// resolved per module from the source config's overrides; `None` for a
    /// float source.
    pub source_quantization: Option<(u32, u32)>,
}

/// A tensor the requantizer copies unencoded, with where it lives so the
/// plan and the driver need no second header walk.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PassThroughTensor {
    pub shard_file_name: String,
    /// The name the source stores the tensor under.
    pub tensor_name: String,
    /// The stored shape, before any axis reorder.
    pub shape: Vec<usize>,
    /// The name it is written under (mlx-lm's spelling for the family).
    pub output_name: String,
    pub value_transform: OutputValueTransform,
    /// Written at its stored float width rather than bfloat16.
    pub keeps_stored_dtype: bool,
}

impl PassThroughTensor {
    /// A tensor copied under its own name with its values untouched.
    #[must_use]
    pub fn unchanged(shard_file_name: String, tensor_name: String, shape: Vec<usize>) -> Self {
        Self {
            shard_file_name,
            output_name: tensor_name.clone(),
            tensor_name,
            shape,
            value_transform: OutputValueTransform::Unchanged,
            keeps_stored_dtype: false,
        }
    }
}

/// The complete inventory of one source directory.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModuleInventory {
    source_directory: PathBuf,
    group_size: GroupSize,
    /// Modules keyed by canonical name, in a deterministic order so sweeps,
    /// checkpoints, and reports line up run to run.
    modules: BTreeMap<String, QuantizableModuleDescriptor>,
    /// Tensors that are not quantizable (norms, biases, convolution weights,
    /// routers, anything whose last dimension does not divide the group size)
    /// and pass through the requantizer untouched. Kept so the requantizer's
    /// tensor plan is complete without a second header walk.
    pass_through_tensors: Vec<PassThroughTensor>,
    /// Vision and audio tower tensors, written at bfloat16 to the
    /// `optiq_vision.safetensors` sidecar under their sidecar names.
    #[serde(default)]
    vision_sidecar_tensors: Vec<PassThroughTensor>,
    /// The multi-token-prediction head's tensors, written to
    /// `optiq/mtp.safetensors` under the names the release gives them.
    #[serde(default)]
    mtp_sidecar_tensors: Vec<PassThroughTensor>,
    /// Whether the source config declares a vision or audio tower.
    #[serde(default)]
    declares_multimodal_towers: bool,
    /// Aliases from the alternate spellings mlx-lm uses for the same module
    /// (`model.` stripped, `model.language_model.` reordered or flattened) to
    /// the canonical name, so a probe model's own module names resolve.
    aliases: BTreeMap<String, String>,
}

impl ModuleInventory {
    /// Builds the inventory from the safetensors headers under
    /// `source_directory` and its `config.json`, OptiQ's `_index_bf16_layers`
    /// and `_quantizable_linears` in one header walk:
    /// 1. `config.json` decides how routed experts are named in the mlx-lm
    ///    module tree (the wrapper segment, whether `experts` is dropped,
    ///    projection renames), reading the text tower's `model_type` as well
    ///    as the top level, and the group sizes of an already-quantized source.
    /// 2. Every top-level `*.safetensors` header is read; no tensor data is.
    /// 3. The output layout decides each tensor's place first: tower tensors
    ///    go to the vision sidecar list, and tensors mlx-lm drops (the
    ///    multi-token-prediction head, a tied output head) are left out.
    /// 4. 2-D `.weight` tensors are modules; 3-D tensors follow the fused
    ///    expert layouts (`experts.gate_up_proj` splits into two halves);
    ///    unfused `experts.N.*` tensors are stacked in expert order. Norms,
    ///    integer buffers, routers of families whose router is not a plain
    ///    linear layer, and anything whose last dimension does not divide
    ///    the group size pass through.
    /// 5. Each module is named the way mlx-lm names it, gets its taxonomy
    ///    role and block index, and the source's spelling and the other
    ///    mlx-lm spellings of its name are registered as aliases.
    pub fn from_source_directory(
        source_directory: &Path,
        group_size: GroupSize,
    ) -> Result<Self, QuantizationError> {
        let contents = inventory_builder::build_inventory_contents(source_directory, group_size)?;
        Ok(Self {
            source_directory: source_directory.to_path_buf(),
            group_size,
            modules: contents.modules,
            pass_through_tensors: contents.pass_through_tensors,
            vision_sidecar_tensors: contents.vision_sidecar_tensors,
            mtp_sidecar_tensors: contents.mtp_sidecar_tensors,
            declares_multimodal_towers: contents.declares_multimodal_towers,
            aliases: contents.aliases,
        })
    }

    /// Builds an inventory from descriptors already in hand (tests, the probe
    /// model's own module list, or a sensitivity document being re-targeted).
    #[must_use]
    pub fn from_descriptors(
        source_directory: PathBuf,
        group_size: GroupSize,
        descriptors: Vec<QuantizableModuleDescriptor>,
    ) -> Self {
        let mut modules = BTreeMap::new();
        for descriptor in descriptors {
            modules.insert(descriptor.module_name.clone(), descriptor);
        }
        Self {
            source_directory,
            group_size,
            modules,
            pass_through_tensors: Vec::new(),
            vision_sidecar_tensors: Vec::new(),
            mtp_sidecar_tensors: Vec::new(),
            declares_multimodal_towers: false,
            aliases: BTreeMap::new(),
        }
    }

    #[must_use]
    pub fn source_directory(&self) -> &Path {
        &self.source_directory
    }

    #[must_use]
    pub const fn group_size(&self) -> GroupSize {
        self.group_size
    }

    pub fn modules(&self) -> impl Iterator<Item = &QuantizableModuleDescriptor> {
        self.modules.values()
    }

    #[must_use]
    pub fn module_count(&self) -> usize {
        self.modules.len()
    }

    /// The modules a sensitivity method scores and the allocator spends
    /// bits on: every module except the token embedding table. OptiQ scores
    /// linear layers only, so an embedding table never enters the allocation
    /// and is written at the allocation's highest width instead; scoring it
    /// here would change both the bit map and the achieved bits per weight.
    ///
    /// The order is mlx-lm's module traversal order, not name order: the
    /// allocator breaks ties by table order, and OptiQ's table is in
    /// traversal order.
    #[must_use]
    pub fn allocation_modules(&self) -> Vec<&QuantizableModuleDescriptor> {
        let mut allocation_modules = Vec::with_capacity(self.modules.len());
        for descriptor in self.modules.values() {
            if !is_embedding_table_module(&descriptor.module_name) {
                allocation_modules.push(descriptor);
            }
        }
        allocation_modules.sort_by(compare_descriptors_in_traversal_order);
        allocation_modules
    }

    /// Whether any module's weight has to be assembled from the source:
    /// split out of a fused tensor, stacked from per-expert tensors, or read
    /// from a tensor that is not its own `<name>.weight`. A serving engine
    /// reads mlx-lm's layout, where every module is one such tensor, so a
    /// source like this cannot be loaded for probing as it stands.
    #[must_use]
    pub fn has_assembled_modules(&self) -> bool {
        for descriptor in self.modules.values() {
            let is_stored_as_its_own_weight = match &descriptor.weight_source {
                ModuleWeightSource::Whole { tensor_name, .. } => tensor_name.ends_with(".weight"),
                ModuleWeightSource::FusedGateUpHalf { .. }
                | ModuleWeightSource::StackedPerExpert { .. } => false,
            };
            if !is_stored_as_its_own_weight {
                return true;
            }
        }
        false
    }

    #[must_use]
    pub fn vision_sidecar_tensors(&self) -> &[PassThroughTensor] {
        &self.vision_sidecar_tensors
    }

    #[must_use]
    pub fn mtp_sidecar_tensors(&self) -> &[PassThroughTensor] {
        &self.mtp_sidecar_tensors
    }

    #[must_use]
    pub const fn declares_multimodal_towers(&self) -> bool {
        self.declares_multimodal_towers
    }

    #[must_use]
    pub fn pass_through_tensor_names(&self) -> Vec<String> {
        let mut tensor_names = Vec::with_capacity(self.pass_through_tensors.len());
        for pass_through_tensor in &self.pass_through_tensors {
            tensor_names.push(pass_through_tensor.tensor_name.clone());
        }
        tensor_names
    }

    #[must_use]
    pub fn pass_through_tensors(&self) -> &[PassThroughTensor] {
        &self.pass_through_tensors
    }

    /// Resolves a canonical name or any registered alias.
    #[must_use]
    pub fn resolve(&self, module_name_or_alias: &str) -> Option<&QuantizableModuleDescriptor> {
        if let Some(descriptor) = self.modules.get(module_name_or_alias) {
            return Some(descriptor);
        }
        let canonical_name = self.aliases.get(module_name_or_alias)?;
        self.modules.get(canonical_name)
    }

    /// Sum of logical parameter counts across every quantizable module; the
    /// denominator of every bits-per-weight figure.
    #[must_use]
    pub fn total_logical_parameter_count(&self) -> u64 {
        let mut total_logical_parameter_count = 0_u64;
        for descriptor in self.modules.values() {
            total_logical_parameter_count += descriptor.logical_parameter_count;
        }
        total_logical_parameter_count
    }

    /// The highest block index plus one, which the static scorer needs to
    /// know where the network's edges are.
    #[must_use]
    pub fn transformer_block_count(&self) -> usize {
        let mut transformer_block_count = 1_usize;
        for descriptor in self.modules.values() {
            if let Some(block_index) = descriptor.transformer_block_index {
                transformer_block_count = transformer_block_count.max(block_index + 1);
            }
        }
        transformer_block_count
    }
}

/// Sort comparator for [`ModuleInventory::allocation_modules`].
fn compare_descriptors_in_traversal_order(
    left: &&QuantizableModuleDescriptor,
    right: &&QuantizableModuleDescriptor,
) -> std::cmp::Ordering {
    traversal_order::compare_in_mlx_lm_traversal_order(&left.module_name, &right.module_name)
}
