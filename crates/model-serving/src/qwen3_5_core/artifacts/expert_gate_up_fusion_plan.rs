use thiserror::Error;

use crate::expert_paging::{QuantizationMode, QuantizedExpertLayerPlan, QuantizedTensorSource};

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum ExpertGateUpFusionPlanError {
    #[error("expert gate/up fusion plan is missing source tensor {tensor_name:?}")]
    MissingTensor { tensor_name: String },
    #[error("expert gate/up fusion transient payload byte count overflowed")]
    PayloadOverflow,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExpertGateUpFusionPlan {
    Fused {
        materialization_transient_payload_bytes: u64,
    },
    Separate {
        incompatibility_reason: &'static str,
    },
}

impl ExpertGateUpFusionPlan {
    pub fn from_layer_plan(
        layer_plan: &QuantizedExpertLayerPlan,
    ) -> Result<Self, ExpertGateUpFusionPlanError> {
        let gate_quantization_mode = layer_plan.quantization_mode_for_projection("gate_proj");
        let up_quantization_mode = layer_plan.quantization_mode_for_projection("up_proj");
        if gate_quantization_mode != up_quantization_mode {
            return Ok(Self::Separate {
                incompatibility_reason: "gate and up storage encodings differ",
            });
        }
        let parameter_names: &[&str] = match gate_quantization_mode {
            QuantizationMode::NativeBfloat16 => &["weight"],
            QuantizationMode::Affine => &["weight", "scales", "biases"],
        };
        let mut transient_payload_bytes = 0_u64;
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
            let gate_payload_bytes = complete_source_payload_bytes(gate_source)?;
            let up_payload_bytes = complete_source_payload_bytes(up_source)?;
            transient_payload_bytes = transient_payload_bytes
                .checked_add(gate_payload_bytes)
                .and_then(|payload_bytes| payload_bytes.checked_add(up_payload_bytes))
                .ok_or(ExpertGateUpFusionPlanError::PayloadOverflow)?;
        }
        if gate_quantization_mode == QuantizationMode::Affine {
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
            materialization_transient_payload_bytes: transient_payload_bytes,
        })
    }

    #[must_use]
    pub const fn materialization_transient_payload_bytes(self) -> u64 {
        match self {
            Self::Fused {
                materialization_transient_payload_bytes,
            } => materialization_transient_payload_bytes,
            Self::Separate { .. } => 0,
        }
    }
}

pub fn maximum_expert_gate_up_fusion_transient_payload_bytes(
    layer_plans: &[QuantizedExpertLayerPlan],
) -> Result<u64, ExpertGateUpFusionPlanError> {
    layer_plans
        .iter()
        .try_fold(0_u64, |maximum_bytes, layer_plan| {
            let fusion_plan = ExpertGateUpFusionPlan::from_layer_plan(layer_plan)?;
            Ok(maximum_bytes.max(fusion_plan.materialization_transient_payload_bytes()))
        })
}

fn projection_parameter_source<'plan>(
    layer_plan: &'plan QuantizedExpertLayerPlan,
    projection_name: &str,
    parameter_name: &str,
) -> Result<&'plan QuantizedTensorSource, ExpertGateUpFusionPlanError> {
    layer_plan
        .tensor_sources
        .iter()
        .find(|tensor_source| {
            tensor_source.projection_name == projection_name
                && tensor_source.parameter_name == parameter_name
        })
        .ok_or_else(|| ExpertGateUpFusionPlanError::MissingTensor {
            tensor_name: format!(
                "{}.switch_mlp.{projection_name}.{parameter_name}",
                layer_plan.layer_prefix
            ),
        })
}

fn complete_source_payload_bytes(
    tensor_source: &QuantizedTensorSource,
) -> Result<u64, ExpertGateUpFusionPlanError> {
    u64::try_from(tensor_source.bytes_per_expert)
        .ok()
        .and_then(|bytes_per_expert| {
            u64::try_from(tensor_source.expert_capacity)
                .ok()
                .and_then(|expert_capacity| bytes_per_expert.checked_mul(expert_capacity))
        })
        .ok_or(ExpertGateUpFusionPlanError::PayloadOverflow)
}
