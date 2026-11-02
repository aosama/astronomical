//! Reads a plan entry's source weight as a float MLX array, assembling the
//! on-disk layouts the inventory recorded: whole tensors, fused gate-up
//! halves, per-expert stacks, block-scaled FP8, and already-packed affine
//! weights. OptiQ's `_load_bf16_weight` and `_dequant_fp8`, plus the
//! dequantize step of `requantize_shardwise`.
//!
//! Every array handed back is lazy: a shard is opened through a retained
//! descriptor and MLX reads only the byte range of the tensor being
//! evaluated, so a 300B source is never resident at once.
//!
//! Ported from OptiQ's `_load_bf16_weight` and `_dequant_fp8` (`optiq/core/sensitivity.py`) and the dequantize step of `requantize_shardwise` (`optiq/core/shardwise.py`).

use std::collections::BTreeMap;
use std::fs::File;
use std::path::{Path, PathBuf};

use astronomical_runtime_integration::{MlxRuntime, MlxSafetensors};
use mlx_c_rust::{MlxArray, MlxDtype};

use crate::error::QuantizationError;
use crate::module_inventory::{ModuleWeightSource, QuantizableModuleDescriptor, SourceTensorDtype};

/// The block edge of DeepSeek-style FP8 block scales: one `scale_inv` per
/// 128x128 tile.
const FP8_SCALE_BLOCK_EDGE: i32 = 128;

/// The shards opened so far, keyed by file name.
pub struct SourceShardMaps {
    source_directory: PathBuf,
    open_shards: BTreeMap<String, MlxSafetensors>,
}

impl SourceShardMaps {
    #[must_use]
    pub fn new(source_directory: &Path) -> Self {
        Self {
            source_directory: source_directory.to_path_buf(),
            open_shards: BTreeMap::new(),
        }
    }

    /// Closes every shard except `shard_file_name`. The plan visits shards in
    /// order, so once the walk moves on an earlier shard is only needed again
    /// by a stacked module that spans shards, which reopens it.
    pub fn retain_only(&mut self, shard_file_name: &str) {
        let kept_shard = self.open_shards.remove(shard_file_name);
        self.open_shards.clear();
        if let Some(kept_shard) = kept_shard {
            self.open_shards
                .insert(shard_file_name.to_owned(), kept_shard);
        }
    }

    /// One named tensor, lazily.
    pub fn tensor(
        &mut self,
        runtime: &MlxRuntime,
        shard_file_name: &str,
        tensor_name: &str,
    ) -> Result<MlxArray, QuantizationError> {
        if !self.open_shards.contains_key(shard_file_name) {
            let shard_path = self.source_directory.join(shard_file_name);
            let shard_file = match File::open(&shard_path) {
                Ok(shard_file) => shard_file,
                Err(open_error) => {
                    return Err(QuantizationError::SafetensorsHeaderUnreadable {
                        shard_path,
                        description: open_error.to_string(),
                    });
                }
            };
            let shard_map = runtime.load_safetensors(shard_file, None)?;
            self.open_shards
                .insert(shard_file_name.to_owned(), shard_map);
        }
        Ok(self.open_shards[shard_file_name].tensor(tensor_name)?)
    }
}

/// The module's weight in its logical shape, dequantized from FP8 or a
/// packed source as needed.
pub fn read_module_weight(
    runtime: &MlxRuntime,
    shard_maps: &mut SourceShardMaps,
    module: &QuantizableModuleDescriptor,
) -> Result<MlxArray, QuantizationError> {
    match &module.weight_source {
        ModuleWeightSource::Whole {
            shard_file_name,
            tensor_name,
        } => read_float_tensor(runtime, shard_maps, module, shard_file_name, tensor_name),
        ModuleWeightSource::FusedGateUpHalf {
            shard_file_name,
            tensor_name,
            half_index,
        } => {
            let fused_weight =
                read_float_tensor(runtime, shard_maps, module, shard_file_name, tensor_name)?;
            fused_half(runtime, &fused_weight, *half_index)
        }
        ModuleWeightSource::StackedPerExpert { per_expert_sources } => {
            stacked_experts(runtime, shard_maps, module, per_expert_sources)
        }
    }
}

/// Per-expert 2-D tensors stacked to `[experts, out, in]` in expert order.
fn stacked_experts(
    runtime: &MlxRuntime,
    shard_maps: &mut SourceShardMaps,
    module: &QuantizableModuleDescriptor,
    per_expert_sources: &[(String, String)],
) -> Result<MlxArray, QuantizationError> {
    let mut expert_weights = Vec::with_capacity(per_expert_sources.len());
    for (shard_file_name, tensor_name) in per_expert_sources {
        expert_weights.push(read_float_tensor(
            runtime,
            shard_maps,
            module,
            shard_file_name,
            tensor_name,
        )?);
    }
    let mut expert_weight_references = Vec::with_capacity(expert_weights.len());
    for expert_weight in &expert_weights {
        expert_weight_references.push(expert_weight);
    }
    Ok(runtime.stack_axis(&expert_weight_references, 0)?)
}

/// One stored tensor as floats: FP8 is block-dequantized to bfloat16, a
/// packed affine weight is dequantized with the source's own parameters,
/// and a float tensor is returned as stored.
fn read_float_tensor(
    runtime: &MlxRuntime,
    shard_maps: &mut SourceShardMaps,
    module: &QuantizableModuleDescriptor,
    shard_file_name: &str,
    tensor_name: &str,
) -> Result<MlxArray, QuantizationError> {
    let stored_tensor = shard_maps.tensor(runtime, shard_file_name, tensor_name)?;
    match module.source_dtype {
        SourceTensorDtype::Float8E4M3WithBlockScale => {
            let scale_name = format!("{tensor_name}_scale_inv");
            let block_scales = shard_maps.tensor(runtime, shard_file_name, &scale_name)?;
            dequantize_block_scaled_fp8(runtime, &stored_tensor, &block_scales)
        }
        SourceTensorDtype::MlxAffinePacked => {
            let module_base = tensor_name.strip_suffix(".weight").unwrap_or(tensor_name);
            let scales =
                shard_maps.tensor(runtime, shard_file_name, &format!("{module_base}.scales"))?;
            let biases =
                shard_maps.tensor(runtime, shard_file_name, &format!("{module_base}.biases"))?;
            let Some((source_bits, source_group_size)) = module.source_quantization else {
                return Err(missing_source_quantization(
                    &shard_maps.source_directory,
                    module,
                ));
            };
            Ok(runtime.dequantize_affine(
                &stored_tensor,
                &scales,
                &biases,
                source_group_size as i32,
                source_bits as i32,
            )?)
        }
        SourceTensorDtype::BFloat16 | SourceTensorDtype::Float16 | SourceTensorDtype::Float32 => {
            Ok(stored_tensor)
        }
    }
}

/// The lower (`gate_proj`) or upper (`up_proj`) half of a fused
/// `[experts, 2 * intermediate, hidden]` tensor along axis 1.
fn fused_half(
    runtime: &MlxRuntime,
    fused_weight: &MlxArray,
    half_index: u8,
) -> Result<MlxArray, QuantizationError> {
    let fused_shape = fused_weight.shape();
    let half_extent = fused_shape[1] / 2;
    let mut starts = vec![0_i32; fused_shape.len()];
    let mut stops = fused_shape.clone();
    starts[1] = half_extent * i32::from(half_index);
    stops[1] = starts[1] + half_extent;
    let strides = vec![1_i32; fused_shape.len()];
    Ok(runtime.slice(fused_weight, &starts, &stops, &strides)?)
}

/// OptiQ's `_dequant_fp8`: widen to bfloat16, pad each axis to a whole
/// number of 128-blocks, multiply every tile by its block scale, and crop
/// back to the stored shape.
fn dequantize_block_scaled_fp8(
    runtime: &MlxRuntime,
    fp8_weight: &MlxArray,
    block_scales: &MlxArray,
) -> Result<MlxArray, QuantizationError> {
    let widened_weight = runtime.from_fp8(fp8_weight, MlxDtype::BFloat16)?;
    let bfloat16_scales = runtime.astype(block_scales, MlxDtype::BFloat16)?;
    let stored_shape = widened_weight.shape();
    let (rows, columns) = (stored_shape[0], stored_shape[1]);
    let row_padding = (FP8_SCALE_BLOCK_EDGE - rows % FP8_SCALE_BLOCK_EDGE) % FP8_SCALE_BLOCK_EDGE;
    let column_padding =
        (FP8_SCALE_BLOCK_EDGE - columns % FP8_SCALE_BLOCK_EDGE) % FP8_SCALE_BLOCK_EDGE;
    let padded_weight = runtime.pad(
        &widened_weight,
        &[0, 1],
        &[0, 0],
        &[row_padding, column_padding],
        0.0,
    )?;
    let row_blocks = (rows + row_padding) / FP8_SCALE_BLOCK_EDGE;
    let column_blocks = (columns + column_padding) / FP8_SCALE_BLOCK_EDGE;
    let tiled_weight = runtime.reshape(
        &padded_weight,
        &[
            row_blocks,
            FP8_SCALE_BLOCK_EDGE,
            column_blocks,
            FP8_SCALE_BLOCK_EDGE,
        ],
    )?;
    let broadcast_scales = runtime.reshape(&bfloat16_scales, &[row_blocks, 1, column_blocks, 1])?;
    let scaled_weight = runtime.reshape(
        &runtime.multiply(&tiled_weight, &broadcast_scales)?,
        &[rows + row_padding, columns + column_padding],
    )?;
    Ok(runtime.slice(&scaled_weight, &[0, 0], &[rows, columns], &[1, 1])?)
}

/// A packed weight is undecodable without the source config's bits and
/// group size for it, so the config is what the operator must fix.
fn missing_source_quantization(
    source_directory: &Path,
    module: &QuantizableModuleDescriptor,
) -> QuantizationError {
    QuantizationError::ModelConfigUnreadable {
        config_path: source_directory.join("config.json"),
        description: format!(
            "{} is stored packed but the source config gives no bits and group size for it",
            module.module_name
        ),
    }
}
