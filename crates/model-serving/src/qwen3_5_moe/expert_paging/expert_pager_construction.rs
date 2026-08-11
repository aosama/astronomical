//! Startup construction of validated expert layer plans and memory admission.

use std::collections::HashMap;
use std::path::PathBuf;

use astronomical_runtime_integration::{
    MlxDtype, MlxNativeExpertCache, MlxNativeExpertLayerDescriptor, MlxNativeExpertParameter,
    MlxNativeExpertProjection, MlxNativeExpertTensorSourceDescriptor, MlxRuntime,
};

use super::expert_pager::{ExpertPagingError, Qwen3_5ExpertPager};
use super::quantized_expert_layer_plan::build_quantized_expert_layer_plan;
use crate::expert_paging::{
    LiveMetalBudget, QuantizationMode, QuantizedExpertLayerPlan, SafetensorsDtype,
    build_quantized_expert_page_manifest_from_plan,
};
use crate::qwen3_5::{ModelWeightStorage, Qwen3_5Config};

impl Qwen3_5ExpertPager {
    /// Returns the number of MoE layers with validated layer plans.
    #[must_use]
    pub fn layer_count(&self) -> usize {
        self.layer_plans.len()
    }

    /// Builds layer plans for all MoE layers at startup.
    ///
    /// Reads safetensors headers, validates tensor geometry, and pre-computes
    /// per-expert byte strides. Does NOT load expert weights.
    ///
    /// `weight_map` maps tensor names (e.g.,
    /// `language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight`) to
    /// shard file names (e.g., `model-00001-of-00005.safetensors`).
    /// `model_dir` is the directory containing the shard files.
    ///
    /// `configured_mlx_memory_cap_bytes` is the worker's resolved numeric
    /// admission ceiling for all expert paging operations.
    pub fn new(
        runtime: &MlxRuntime,
        model_dir: PathBuf,
        weight_map: &HashMap<String, String>,
        config: &Qwen3_5Config,
        configured_mlx_memory_cap_bytes: usize,
        include_mtp_sparse_expert_layer: bool,
    ) -> Result<Self, ExpertPagingError> {
        let decoder_layer_count = config.layer_count() as usize;
        let mut layer_plans =
            Vec::with_capacity(decoder_layer_count + usize::from(include_mtp_sparse_expert_layer));
        let quantization_mode = match config.model_weight_storage() {
            ModelWeightStorage::NativeBfloat16 => QuantizationMode::NativeBfloat16,
            ModelWeightStorage::AffineQuantized => QuantizationMode::Affine,
        };
        for decoder_layer_index in 0..decoder_layer_count {
            let layer_prefix = format!("language_model.model.layers.{decoder_layer_index}.mlp");
            let layer_plan = build_quantized_expert_layer_plan(
                &model_dir,
                weight_map,
                &layer_prefix,
                config,
                quantization_mode,
            )?;
            layer_plans.push(layer_plan);
        }
        // MTP has a separate tensor namespace but shares the same pager and
        // native cache. Appending its one sparse layer gives it a stable index
        // immediately after the target decoder layers without merging artifact
        // inventories during validation.
        if include_mtp_sparse_expert_layer {
            let mtp_layer_plan = build_quantized_expert_layer_plan(
                &model_dir,
                weight_map,
                "language_model.mtp.layers.0.mlp",
                config,
                quantization_mode,
            )?;
            layer_plans.push(mtp_layer_plan);
        }

        let routed_expert_count = usize::try_from(config.experts_per_token()).map_err(|_| {
            ExpertPagingError::Runtime {
                description: "experts per token exceed the host integer range".to_owned(),
            }
        })?;
        // Page size depends on tensor geometry and routed top-K count, not on
        // which expert IDs were selected. A representative contiguous route
        // therefore computes the exact largest non-evictable route reserve
        // without reading any expert payload.
        let representative_routed_expert_ids = (0..routed_expert_count).collect::<Vec<_>>();
        let maximum_expert_page_bytes = layer_plans.iter().try_fold(
            0u64,
            |maximum_expert_page_bytes, layer_plan| -> Result<_, ExpertPagingError> {
                let routed_page_payload_byte_count =
                    build_quantized_expert_page_manifest_from_plan(
                        layer_plan,
                        &representative_routed_expert_ids,
                    )?
                    .payload_byte_count;
                Ok(maximum_expert_page_bytes.max(routed_page_payload_byte_count))
            },
        )?;
        let memory_budget = LiveMetalBudget::new(
            maximum_expert_page_bytes,
            configured_mlx_memory_cap_bytes as u64,
        );
        // These Rust descriptors own paths and shapes only for this call. The
        // native constructor validates and copies all nested metadata before
        // returning, so no borrowed pointer crosses the startup boundary.
        let native_layer_descriptors = layer_plans
            .iter()
            .map(native_layer_descriptor)
            .collect::<Result<Vec<_>, _>>()?;
        let native_expert_cache = MlxNativeExpertCache::new(
            runtime,
            &native_layer_descriptors,
            configured_mlx_memory_cap_bytes as u64,
        )
        .map_err(|error| ExpertPagingError::Runtime {
            description: error.to_string(),
        })?;
        Ok(Self {
            layer_plans,
            memory_budget,
            native_expert_cache,
        })
    }
}

fn native_layer_descriptor(
    layer_plan: &QuantizedExpertLayerPlan,
) -> Result<MlxNativeExpertLayerDescriptor, ExpertPagingError> {
    let tensor_sources = layer_plan
        .tensor_sources
        .iter()
        .map(|tensor_source| {
            let projection = match tensor_source.projection_name.as_str() {
                "gate_proj" => MlxNativeExpertProjection::Gate,
                "up_proj" => MlxNativeExpertProjection::Up,
                "down_proj" => MlxNativeExpertProjection::Down,
                projection_name => {
                    return Err(ExpertPagingError::Runtime {
                        description: format!(
                            "unsupported native expert projection {projection_name:?}"
                        ),
                    });
                }
            };
            let parameter = match tensor_source.parameter_name.as_str() {
                "weight" => MlxNativeExpertParameter::PackedWeight,
                "scales" => MlxNativeExpertParameter::Scales,
                "biases" => MlxNativeExpertParameter::Biases,
                parameter_name => {
                    return Err(ExpertPagingError::Runtime {
                        description: format!(
                            "unsupported native expert parameter {parameter_name:?}"
                        ),
                    });
                }
            };
            // Native slots store one expert, while safetensors sources describe
            // the complete expert stack. Replace only the leading capacity
            // dimension with one; all remaining geometry and dtype stay exact.
            let expert_shape = tensor_source
                .full_shape
                .iter()
                .enumerate()
                .map(|(dimension_index, dimension)| {
                    let expert_local_dimension = if dimension_index == 0 { 1 } else { *dimension };
                    i32::try_from(expert_local_dimension).map_err(|_| ExpertPagingError::Runtime {
                        description: format!(
                            "native expert tensor {:?} dimension exceeds the MLX range",
                            tensor_source.tensor_name
                        ),
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            let dtype =
                mlx_dtype(tensor_source.dtype).ok_or_else(|| ExpertPagingError::Runtime {
                    description: format!(
                        "native expert tensor {:?} uses unsupported dtype {}",
                        tensor_source.tensor_name, tensor_source.dtype
                    ),
                })?;
            Ok(MlxNativeExpertTensorSourceDescriptor::new(
                projection,
                parameter,
                tensor_source.quantization_group_size,
                tensor_source.quantization_bits,
                tensor_source.source_file.clone(),
                tensor_source.tensor_payload_offset,
                tensor_source.bytes_per_expert,
                expert_shape,
                dtype,
            ))
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(MlxNativeExpertLayerDescriptor::new(
        layer_plan.expert_capacity,
        tensor_sources,
    ))
}

const fn mlx_dtype(safetensors_dtype: SafetensorsDtype) -> Option<MlxDtype> {
    match safetensors_dtype {
        SafetensorsDtype::Bool => Some(MlxDtype::Bool),
        SafetensorsDtype::Int8 => Some(MlxDtype::Int8),
        SafetensorsDtype::Uint8 => Some(MlxDtype::UInt8),
        SafetensorsDtype::Int16 => Some(MlxDtype::Int16),
        SafetensorsDtype::Uint16 => Some(MlxDtype::UInt16),
        SafetensorsDtype::Float16 => Some(MlxDtype::Float16),
        SafetensorsDtype::BFloat16 => Some(MlxDtype::BFloat16),
        SafetensorsDtype::Int32 => Some(MlxDtype::Int32),
        SafetensorsDtype::Uint32 => Some(MlxDtype::UInt32),
        SafetensorsDtype::Float32 => Some(MlxDtype::Float32),
        SafetensorsDtype::Int64 => Some(MlxDtype::Int64),
        SafetensorsDtype::Uint64 => Some(MlxDtype::UInt64),
        SafetensorsDtype::Float8E4M3 | SafetensorsDtype::Float8E5M2 => None,
    }
}
