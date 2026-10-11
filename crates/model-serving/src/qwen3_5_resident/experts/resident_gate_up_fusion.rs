//! Resident-only ownership for one routed expert gate/up pair.
//!
//! Durable complete experts can pay for one physical concatenation while loading
//! and then remove one gathered projection from every forward. Streamed pages do
//! not use this owner because repeatedly materializing concatenations would add
//! work to their one-operation lifetime.

use astronomical_runtime_integration::MlxRuntime;

use crate::expert_paging::{QuantizationMode, QuantizedExpertLayerPlan};
use crate::qwen3_5_core::artifacts::expert_gate_up_fusion_plan::{
    ExpertGateUpFusionPlan, ExpertGateUpFusionPlanError,
    maximum_expert_gate_up_fusion_transient_payload_bytes,
};
use crate::qwen3_5_core::model_math::decoder_layer_weights::Qwen3_5AffineWeights;
use crate::qwen3_5_core::model_math::error::Qwen3_5ExecutionError;
use crate::qwen3_5_core::model_math::expert_gate_up_fusion::fuse_compatible_expert_gate_up_projections;
use astronomical_mlx_c_rust::MlxArray;

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
        let fusion_plan =
            ExpertGateUpFusionPlan::from_layer_plan(layer_plan).map_err(map_fusion_plan_error)?;
        match fusion_plan {
            ExpertGateUpFusionPlan::Separate {
                incompatibility_reason,
            } => Ok(Self::Separate {
                gate_projection,
                up_projection,
                incompatibility_reason,
            }),
            ExpertGateUpFusionPlan::Fused {
                materialization_transient_payload_bytes,
            } => Ok(Self::Fused {
                projection: fuse_compatible_expert_gate_up_projections(
                    runtime,
                    gate_projection,
                    up_projection,
                )?,
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
    maximum_expert_gate_up_fusion_transient_payload_bytes(layer_plans)
        .map_err(map_fusion_plan_error)
}

fn map_fusion_plan_error(fusion_plan_error: ExpertGateUpFusionPlanError) -> Qwen3_5ExecutionError {
    match fusion_plan_error {
        ExpertGateUpFusionPlanError::MissingTensor { tensor_name } => {
            Qwen3_5ExecutionError::MissingTensor { tensor_name }
        }
        ExpertGateUpFusionPlanError::PayloadOverflow => Qwen3_5ExecutionError::InvalidInput {
            description: "resident gate/up fusion transient payload overflowed",
        },
    }
}

/// One projection's retained arrays, in the owner's storage representation.
///
/// `quantization_scales` and `quantization_biases` are `None` for a native
/// floating-point projection and `Some` for an affine-quantized one.
#[doc(hidden)]
#[derive(Debug)]
pub struct ResidentProjectionArraysForTests {
    pub packed_weight: MlxArray,
    pub quantization_scales: Option<MlxArray>,
    pub quantization_biases: Option<MlxArray>,
}

/// The arrays one resident expert layer retains, as the production construction
/// produced them.
///
/// `gate_up` holds one entry when the plan fused the pair and two entries
/// (gate, then up) when the plan kept them separate.
#[doc(hidden)]
#[derive(Debug)]
pub struct ResidentLayerArraysForTests {
    pub gate_up: Vec<ResidentProjectionArraysForTests>,
    pub is_fused: bool,
    pub down: ResidentProjectionArraysForTests,
}

/// Runs the production resident layer construction on caller-supplied arrays.
///
/// Issue #503 compares two array provenances for one layer: the whole-shard
/// read the resident loader performs and the bounded source-interval read the
/// pager performs. The construction decision and the gate/up fusion stay in
/// production code; this adapter only wraps the arrays a test read into the
/// owner's input type and hands the retained arrays back, because the owner
/// types are crate-private and an integration test cannot name them.
#[doc(hidden)]
pub fn resident_layer_arrays_for_tests(
    runtime: &MlxRuntime,
    layer_plan: &QuantizedExpertLayerPlan,
    gate: ResidentProjectionArraysForTests,
    up: ResidentProjectionArraysForTests,
    down: ResidentProjectionArraysForTests,
) -> Result<ResidentLayerArraysForTests, Qwen3_5ExecutionError> {
    let affine_projection = |projection_name: &str,
                             projection: ResidentProjectionArraysForTests|
     -> Result<Qwen3_5AffineWeights, Qwen3_5ExecutionError> {
        match layer_plan.quantization_mode_for_projection(projection_name) {
            QuantizationMode::NativeBfloat16 => Ok(Qwen3_5AffineWeights::NativeBfloat16 {
                weight: projection.packed_weight,
            }),
            QuantizationMode::Affine => {
                let quantization_scales = projection.quantization_scales.ok_or_else(|| {
                    Qwen3_5ExecutionError::InvalidInput {
                        description: "affine construction requires quantization scales",
                    }
                })?;
                let quantization_biases = projection.quantization_biases.ok_or_else(|| {
                    Qwen3_5ExecutionError::InvalidInput {
                        description: "affine construction requires quantization biases",
                    }
                })?;
                Ok(Qwen3_5AffineWeights::Quantized {
                    packed_weight: projection.packed_weight,
                    quantization_scales,
                    quantization_biases,
                    quantization_bits: layer_plan.quantization_bits,
                    quantization_group_size: layer_plan.quantization_group_size,
                })
            }
        }
    };
    let gate_projection = affine_projection("gate_proj", gate)?;
    let up_projection = affine_projection("up_proj", up)?;
    let down_projection = affine_projection("down_proj", down)?;

    let gate_up_weights =
        Qwen3_5ResidentGateUpWeights::build(runtime, layer_plan, gate_projection, up_projection)?;
    Ok(match gate_up_weights {
        Qwen3_5ResidentGateUpWeights::Fused { projection, .. } => ResidentLayerArraysForTests {
            gate_up: vec![projection_arrays_for_tests(&projection)?],
            is_fused: true,
            down: projection_arrays_for_tests(&down_projection)?,
        },
        Qwen3_5ResidentGateUpWeights::Separate {
            gate_projection,
            up_projection,
            incompatibility_reason: _,
        } => ResidentLayerArraysForTests {
            gate_up: vec![
                projection_arrays_for_tests(&gate_projection)?,
                projection_arrays_for_tests(&up_projection)?,
            ],
            is_fused: false,
            down: projection_arrays_for_tests(&down_projection)?,
        },
    })
}

fn projection_arrays_for_tests(
    projection: &Qwen3_5AffineWeights,
) -> Result<ResidentProjectionArraysForTests, Qwen3_5ExecutionError> {
    // Retain through the production accessor so the returned arrays are the
    // ones the resident owner would hold, not borrowed views.
    let retained = projection.retained_reference().map_err(|_runtime_error| {
        Qwen3_5ExecutionError::InvalidInput {
            description: "a resident projection could not be retained for parity comparison",
        }
    })?;
    Ok(match retained {
        Qwen3_5AffineWeights::NativeBfloat16 { weight } => ResidentProjectionArraysForTests {
            packed_weight: weight,
            quantization_scales: None,
            quantization_biases: None,
        },
        Qwen3_5AffineWeights::Quantized {
            packed_weight,
            quantization_scales,
            quantization_biases,
            ..
        } => ResidentProjectionArraysForTests {
            packed_weight,
            quantization_scales: Some(quantization_scales),
            quantization_biases: Some(quantization_biases),
        },
    })
}
