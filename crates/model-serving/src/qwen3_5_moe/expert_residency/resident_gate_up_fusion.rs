//! Resident-only ownership for one routed expert gate/up pair.
//!
//! Durable complete experts can pay for one physical concatenation while loading
//! and then remove one gathered projection from every forward. Streamed pages do
//! not use this owner because repeatedly materializing concatenations would add
//! work to their one-operation lifetime.

use astronomical_runtime_integration::{MlxArray, MlxRuntime};

use crate::expert_paging::{QuantizationMode, QuantizedExpertLayerPlan, QuantizedTensorSource};
use crate::qwen3_5::model::Qwen3_5ExecutionError;
use crate::qwen3_5::model::decoder_layer_weights::Qwen3_5AffineWeights;

/// Resident gate/up ownership selected from startup-validated source geometry.
#[derive(Debug)]
pub(crate) enum Qwen3_5ResidentGateUpWeights {
    /// One `[gate, up]` projection concatenated on expert output axis 1.
    Fused {
        projection: Qwen3_5AffineWeights,
        materialization_transient_payload_bytes: u64,
    },
    /// A valid mixed pair that cannot be concatenated without changing storage.
    Separate {
        gate_projection: Qwen3_5AffineWeights,
        up_projection: Qwen3_5AffineWeights,
        incompatibility_reason: &'static str,
    },
}

impl Qwen3_5ResidentGateUpWeights {
    pub(super) fn build(
        runtime: &MlxRuntime,
        layer_plan: &QuantizedExpertLayerPlan,
        gate_projection: Qwen3_5AffineWeights,
        up_projection: Qwen3_5AffineWeights,
    ) -> Result<Self, Qwen3_5ExecutionError> {
        let fusion_plan = ResidentGateUpFusionPlan::from_layer_plan(layer_plan)?;
        match fusion_plan {
            ResidentGateUpFusionPlan::Separate {
                incompatibility_reason,
            } => Ok(Self::Separate {
                gate_projection,
                up_projection,
                incompatibility_reason,
            }),
            ResidentGateUpFusionPlan::Fused {
                materialization_transient_payload_bytes,
            } => Ok(Self::Fused {
                projection: fuse_compatible_projections(runtime, gate_projection, up_projection)?,
                materialization_transient_payload_bytes,
            }),
        }
    }

    pub(crate) const fn is_fused(&self) -> bool {
        matches!(self, Self::Fused { .. })
    }

    pub(crate) const fn materialization_transient_payload_bytes(&self) -> u64 {
        match self {
            Self::Fused {
                materialization_transient_payload_bytes,
                ..
            } => *materialization_transient_payload_bytes,
            Self::Separate { .. } => 0,
        }
    }

    pub(crate) const fn incompatibility_reason(&self) -> Option<&'static str> {
        match self {
            Self::Fused { .. } => None,
            Self::Separate {
                incompatibility_reason,
                ..
            } => Some(*incompatibility_reason),
        }
    }

    pub(crate) fn append_array_references<'weights>(
        &'weights self,
        arrays: &mut Vec<&'weights MlxArray>,
    ) {
        match self {
            Self::Fused { projection, .. } => projection.append_array_references(arrays),
            Self::Separate {
                gate_projection,
                up_projection,
                ..
            } => {
                gate_projection.append_array_references(arrays);
                up_projection.append_array_references(arrays);
            }
        }
    }
}

/// Maximum temporary duplicate needed while materializing one compatible layer.
pub fn maximum_resident_gate_up_fusion_transient_payload_bytes(
    layer_plans: &[QuantizedExpertLayerPlan],
) -> Result<u64, Qwen3_5ExecutionError> {
    layer_plans
        .iter()
        .try_fold(0_u64, |maximum_bytes, layer_plan| {
            let fusion_plan = ResidentGateUpFusionPlan::from_layer_plan(layer_plan)?;
            Ok(maximum_bytes.max(fusion_plan.materialization_transient_payload_bytes()))
        })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ResidentGateUpFusionPlan {
    Fused {
        materialization_transient_payload_bytes: u64,
    },
    Separate {
        incompatibility_reason: &'static str,
    },
}

impl ResidentGateUpFusionPlan {
    fn from_layer_plan(
        layer_plan: &QuantizedExpertLayerPlan,
    ) -> Result<Self, Qwen3_5ExecutionError> {
        let parameter_names: &[&str] = match layer_plan.quantization_mode {
            QuantizationMode::NativeBfloat16 => &["weight"],
            QuantizationMode::Affine => &["weight", "scales", "biases"],
        };
        let mut materialization_transient_payload_bytes = 0_u64;
        for parameter_name in parameter_names {
            let gate_source = projection_parameter_source(layer_plan, "gate_proj", parameter_name)?;
            let up_source = projection_parameter_source(layer_plan, "up_proj", parameter_name)?;
            if gate_source.full_shape != up_source.full_shape {
                return Ok(Self::Separate {
                    incompatibility_reason: "gate and up tensor shapes differ",
                });
            }
            if gate_source.dtype != up_source.dtype {
                return Ok(Self::Separate {
                    incompatibility_reason: "gate and up tensor data types differ",
                });
            }
            let gate_payload_bytes = complete_source_payload_bytes(layer_plan, gate_source)?;
            let up_payload_bytes = complete_source_payload_bytes(layer_plan, up_source)?;
            materialization_transient_payload_bytes = materialization_transient_payload_bytes
                .checked_add(gate_payload_bytes)
                .and_then(|payload_bytes| payload_bytes.checked_add(up_payload_bytes))
                .ok_or(Qwen3_5ExecutionError::InvalidInput {
                    description: "resident gate/up fusion transient payload overflowed",
                })?;
        }
        if layer_plan.quantization_mode == QuantizationMode::Affine {
            let gate_weight = projection_parameter_source(layer_plan, "gate_proj", "weight")?;
            let up_weight = projection_parameter_source(layer_plan, "up_proj", "weight")?;
            if gate_weight.quantization_bits != up_weight.quantization_bits {
                return Ok(Self::Separate {
                    incompatibility_reason: "gate and up quantization bit widths differ",
                });
            }
            if gate_weight.quantization_group_size != up_weight.quantization_group_size {
                return Ok(Self::Separate {
                    incompatibility_reason: "gate and up quantization group sizes differ",
                });
            }
        }
        Ok(Self::Fused {
            materialization_transient_payload_bytes,
        })
    }

    const fn materialization_transient_payload_bytes(self) -> u64 {
        match self {
            Self::Fused {
                materialization_transient_payload_bytes,
            } => materialization_transient_payload_bytes,
            Self::Separate { .. } => 0,
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

fn complete_source_payload_bytes(
    layer_plan: &QuantizedExpertLayerPlan,
    tensor_source: &QuantizedTensorSource,
) -> Result<u64, Qwen3_5ExecutionError> {
    u64::try_from(tensor_source.bytes_per_expert)
        .ok()
        .and_then(|bytes_per_expert| {
            u64::try_from(tensor_source.expert_capacity)
                .ok()
                .and_then(|expert_capacity| bytes_per_expert.checked_mul(expert_capacity))
        })
        .ok_or_else(|| Qwen3_5ExecutionError::InvalidTensor {
            tensor_name: format!(
                "{}.switch_mlp.{}.{}",
                layer_plan.layer_prefix,
                tensor_source.projection_name,
                tensor_source.parameter_name
            ),
            description: "resident gate/up source payload byte count overflowed",
        })
}

fn fuse_compatible_projections(
    runtime: &MlxRuntime,
    gate_projection: Qwen3_5AffineWeights,
    up_projection: Qwen3_5AffineWeights,
) -> Result<Qwen3_5AffineWeights, Qwen3_5ExecutionError> {
    match (gate_projection, up_projection) {
        (
            Qwen3_5AffineWeights::NativeBfloat16 {
                weight: gate_weight,
            },
            Qwen3_5AffineWeights::NativeBfloat16 { weight: up_weight },
        ) => Ok(Qwen3_5AffineWeights::NativeBfloat16 {
            weight: runtime.concatenate_axis(&[&gate_weight, &up_weight], 1)?,
        }),
        (
            Qwen3_5AffineWeights::Quantized {
                packed_weight: gate_packed_weight,
                quantization_scales: gate_quantization_scales,
                quantization_biases: gate_quantization_biases,
                quantization_bits: gate_quantization_bits,
                quantization_group_size: gate_quantization_group_size,
            },
            Qwen3_5AffineWeights::Quantized {
                packed_weight: up_packed_weight,
                quantization_scales: up_quantization_scales,
                quantization_biases: up_quantization_biases,
                quantization_bits: up_quantization_bits,
                quantization_group_size: up_quantization_group_size,
            },
        ) if gate_quantization_bits == up_quantization_bits
            && gate_quantization_group_size == up_quantization_group_size =>
        {
            Ok(Qwen3_5AffineWeights::Quantized {
                packed_weight: runtime
                    .concatenate_axis(&[&gate_packed_weight, &up_packed_weight], 1)?,
                quantization_scales: runtime
                    .concatenate_axis(&[&gate_quantization_scales, &up_quantization_scales], 1)?,
                quantization_biases: runtime
                    .concatenate_axis(&[&gate_quantization_biases, &up_quantization_biases], 1)?,
                quantization_bits: gate_quantization_bits,
                quantization_group_size: gate_quantization_group_size,
            })
        }
        _ => Err(Qwen3_5ExecutionError::InvalidInput {
            description: "validated resident gate/up fusion plan disagreed with loaded weights",
        }),
    }
}
