use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use astronomical_runtime_integration::{
    MlxArray, MlxDtype, MlxRuntime, MlxSafetensors, PositionalFileReadMetrics,
};

use crate::expert_paging::{
    QuantizationMode, QuantizedExpertLayerPlan, QuantizedTensorSource, SafetensorsDtype,
};
use crate::qwen3_5::model::decoder_layer_weights::Qwen3_5AffineWeights;
use crate::qwen3_5::model::{Qwen3_5ExecutionError, Qwen3_5Model};
use crate::qwen3_5_moe::ExpertPagingError;

use super::{
    Qwen3_5ResidentExpertLayerWeights, Qwen3_5ResidentExpertWeights, Qwen3_5ResidentGateUpWeights,
};

impl Qwen3_5ResidentExpertWeights {
    /// Builds a private complete-model candidate from startup-validated plans.
    ///
    /// The caller retires and clears native streaming ownership before entering here and
    /// publishes the returned owner only after every layer has materialized.
    /// Therefore any error drops this local candidate without changing the
    /// model's externally visible `Paged` state.
    pub(crate) fn load(
        model: &Qwen3_5Model,
        positional_file_read_metrics: Option<Arc<PositionalFileReadMetrics>>,
    ) -> Result<Self, Qwen3_5ExecutionError> {
        let expert_pager =
            model
                .expert_pager
                .as_ref()
                .ok_or(Qwen3_5ExecutionError::InvalidInput {
                    description: "resident expert loading requires a sparse model pager",
                })?;
        let layer_plans = expert_pager.layer_plans();
        let complete_model_payload_bytes = expert_pager.complete_expert_payload_byte_count()?;
        let complete_model_expert_entry_count = expert_pager.complete_expert_entry_count();
        // Duplicate retained descriptors instead of reopening paths. The pager's
        // original descriptors remain valid for recovery, while each clone has
        // an independent lifetime for this promotion attempt.
        let mut resident_source_shards = HashMap::new();
        for (source_file_path, source_file) in expert_pager.clone_resident_expert_source_files()? {
            let resident_source_shard = model.runtime.load_safetensors(
                source_file,
                positional_file_read_metrics.as_ref().map(Arc::clone),
            )?;
            resident_source_shards.insert(source_file_path, resident_source_shard);
        }
        let mut resident_layers = Vec::with_capacity(layer_plans.len());
        let mut fused_gate_up_layer_count = 0_usize;
        let mut separate_gate_up_layer_count = 0_usize;

        // Evaluate one complete layer before advancing. This bounds lazy source
        // graphs and makes each progress record correspond to materialized MLX
        // storage, not merely constructed safetensors views.
        for (layer_index, layer_plan) in layer_plans.iter().enumerate() {
            let complete_layer_payload_bytes = layer_plan
                .complete_expert_payload_byte_count()
                .map_err(ExpertPagingError::from)?;
            tracing::info!(
                layer_index,
                total_layer_count = layer_plans.len(),
                layer_prefix = layer_plan.layer_prefix,
                complete_layer_payload_bytes,
                "started materializing one complete resident expert layer"
            );
            let resident_layer =
                load_resident_layer(&model.runtime, &resident_source_shards, layer_plan)?;
            let gate_up_fusion_applied = resident_layer.gate_up_weights.is_fused();
            let gate_up_fusion_transient_payload_bytes = resident_layer
                .gate_up_weights
                .materialization_transient_payload_bytes();
            let gate_up_fusion_incompatibility_reason =
                resident_layer.gate_up_weights.incompatibility_reason();
            let mut complete_layer_arrays = Vec::new();
            resident_layer.append_array_references(&mut complete_layer_arrays);
            model.runtime.evaluate_arrays(&complete_layer_arrays)?;
            if gate_up_fusion_applied {
                // Synchronous evaluation detached the fused owner from its source
                // concatenation. Drain reclaimable buffers before the next layer.
                model.runtime.clear_allocator_cache()?;
            }
            if gate_up_fusion_applied {
                fused_gate_up_layer_count = fused_gate_up_layer_count.saturating_add(1);
            } else {
                separate_gate_up_layer_count = separate_gate_up_layer_count.saturating_add(1);
            }
            tracing::info!(
                completed_layer_count = layer_index + 1,
                total_layer_count = layer_plans.len(),
                complete_layer_payload_bytes,
                gate_up_fusion_applied,
                gate_up_fusion_transient_payload_bytes,
                gate_up_fusion_incompatibility_reason,
                "materialized one complete resident expert layer"
            );
            resident_layers.push(resident_layer);
        }

        tracing::info!(
            total_layer_count = layer_plans.len(),
            fused_gate_up_layer_count,
            separate_gate_up_layer_count,
            "completed resident expert gate/up materialization"
        );

        Ok(Self::new(
            resident_layers,
            complete_model_expert_entry_count,
            complete_model_payload_bytes,
        ))
    }
}

fn load_resident_layer(
    runtime: &MlxRuntime,
    resident_source_shards: &HashMap<PathBuf, MlxSafetensors>,
    layer_plan: &QuantizedExpertLayerPlan,
) -> Result<Qwen3_5ResidentExpertLayerWeights, Qwen3_5ExecutionError> {
    let gate_projection =
        load_resident_projection(resident_source_shards, layer_plan, "gate_proj")?;
    let up_projection = load_resident_projection(resident_source_shards, layer_plan, "up_proj")?;
    Ok(Qwen3_5ResidentExpertLayerWeights::new(
        Qwen3_5ResidentGateUpWeights::build(runtime, layer_plan, gate_projection, up_projection)?,
        load_resident_projection(resident_source_shards, layer_plan, "down_proj")?,
    ))
}

fn load_resident_projection(
    resident_source_shards: &HashMap<PathBuf, MlxSafetensors>,
    layer_plan: &QuantizedExpertLayerPlan,
    projection_name: &str,
) -> Result<Qwen3_5AffineWeights, Qwen3_5ExecutionError> {
    let weight_source = projection_parameter_source(layer_plan, projection_name, "weight")?;
    let weight = load_validated_source_tensor(resident_source_shards, weight_source)?;
    // Preserve the validated artifact representation exactly. Resident mode is
    // an ownership change, never an opportunity to requantize or widen weights.
    match layer_plan.quantization_mode {
        QuantizationMode::NativeBfloat16 => Ok(Qwen3_5AffineWeights::NativeBfloat16 { weight }),
        QuantizationMode::Affine => {
            let scales_source = projection_parameter_source(layer_plan, projection_name, "scales")?;
            let biases_source = projection_parameter_source(layer_plan, projection_name, "biases")?;
            Ok(Qwen3_5AffineWeights::Quantized {
                packed_weight: weight,
                quantization_scales: load_validated_source_tensor(
                    resident_source_shards,
                    scales_source,
                )?,
                quantization_biases: load_validated_source_tensor(
                    resident_source_shards,
                    biases_source,
                )?,
                quantization_bits: weight_source.quantization_bits,
                quantization_group_size: weight_source.quantization_group_size,
            })
        }
    }
}

fn projection_parameter_source<'plan>(
    layer_plan: &'plan QuantizedExpertLayerPlan,
    projection_name: &str,
    parameter_name: &str,
) -> Result<&'plan QuantizedTensorSource, Qwen3_5ExecutionError> {
    layer_plan
        .tensor_sources
        .iter()
        .find(|tensor_source| {
            tensor_source.projection_name == projection_name
                && tensor_source.parameter_name == parameter_name
        })
        .ok_or_else(|| Qwen3_5ExecutionError::MissingTensor {
            tensor_name: format!(
                "{}.switch_mlp.{projection_name}.{parameter_name}",
                layer_plan.layer_prefix
            ),
        })
}

fn load_validated_source_tensor(
    resident_source_shards: &HashMap<PathBuf, MlxSafetensors>,
    tensor_source: &QuantizedTensorSource,
) -> Result<MlxArray, Qwen3_5ExecutionError> {
    let resident_source_shard = resident_source_shards
        .get(&tensor_source.source_file)
        .ok_or_else(|| Qwen3_5ExecutionError::MissingTensor {
            tensor_name: tensor_source.tensor_name.clone(),
        })?;
    let source_tensor = resident_source_shard.tensor(&tensor_source.tensor_name)?;
    validate_source_tensor(tensor_source, &source_tensor)?;
    Ok(source_tensor)
}

fn validate_source_tensor(
    tensor_source: &QuantizedTensorSource,
    source_tensor: &MlxArray,
) -> Result<(), Qwen3_5ExecutionError> {
    // Plans came from bounded safetensors-header validation. Rechecking the MLX
    // view closes the boundary between header metadata and the array actually
    // retained by the resident owner.
    let expected_shape = tensor_source
        .full_shape
        .iter()
        .map(|dimension| {
            i32::try_from(*dimension).map_err(|_| Qwen3_5ExecutionError::InvalidTensor {
                tensor_name: tensor_source.tensor_name.clone(),
                description: "resident expert tensor shape exceeds the MLX range",
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    if source_tensor.shape() != expected_shape {
        return Err(Qwen3_5ExecutionError::InvalidTensor {
            tensor_name: tensor_source.tensor_name.clone(),
            description: "resident expert tensor shape differs from the validated layer plan",
        });
    }
    if source_tensor.dtype() != mlx_dtype(tensor_source.dtype)? {
        return Err(Qwen3_5ExecutionError::InvalidTensor {
            tensor_name: tensor_source.tensor_name.clone(),
            description: "resident expert tensor dtype differs from the validated layer plan",
        });
    }
    let expected_payload_bytes = u64::try_from(tensor_source.bytes_per_expert)
        .ok()
        .and_then(|bytes_per_expert| {
            u64::try_from(tensor_source.expert_capacity)
                .ok()
                .and_then(|expert_capacity| bytes_per_expert.checked_mul(expert_capacity))
        })
        .ok_or_else(|| Qwen3_5ExecutionError::InvalidTensor {
            tensor_name: tensor_source.tensor_name.clone(),
            description: "resident expert tensor payload byte count overflowed",
        })?;
    let actual_payload_bytes = u64::try_from(source_tensor.byte_count()).map_err(|_| {
        Qwen3_5ExecutionError::InvalidTensor {
            tensor_name: tensor_source.tensor_name.clone(),
            description: "resident expert tensor payload exceeds the u64 range",
        }
    })?;
    if actual_payload_bytes != expected_payload_bytes {
        return Err(Qwen3_5ExecutionError::TensorPayloadMismatch {
            actual_payload_bytes,
            expected_payload_bytes,
        });
    }
    Ok(())
}

fn mlx_dtype(safetensors_dtype: SafetensorsDtype) -> Result<MlxDtype, Qwen3_5ExecutionError> {
    match safetensors_dtype {
        SafetensorsDtype::Uint32 => Ok(MlxDtype::UInt32),
        SafetensorsDtype::Float16 => Ok(MlxDtype::Float16),
        SafetensorsDtype::BFloat16 => Ok(MlxDtype::BFloat16),
        SafetensorsDtype::Float32 => Ok(MlxDtype::Float32),
        _ => Err(Qwen3_5ExecutionError::InvalidInput {
            description: "resident expert tensor plan contains an unsupported dtype",
        }),
    }
}
