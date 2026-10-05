//! Weight-owner types binding stacked affine tensors for one family member.

use astronomical_runtime_integration::MlxRuntime;

use crate::PerformanceAttribution;
use crate::k2_horizon_mova::artifacts::ValidatedK2HorizonMoVAArtifact;
use crate::k2_horizon_mova::expert_geometry::K2HorizonMoVASparseLayerExpertPayload;

use super::affine::K2HorizonMoVAAffineLinear;
use super::error::K2HorizonMoVAExecutionError;
use super::weight_binding;
use astronomical_mlx_c_rust::MlxArray;

#[derive(Debug)]
pub struct K2HorizonMoVADenseAttentionWeights {
    /// One fused q/k/v projection; rows are q rows then k rows then v rows.
    pub query_key_value: K2HorizonMoVAAffineLinear,
    pub query_row_count: usize,
    pub key_value_row_count: usize,
    pub o_proj: K2HorizonMoVAAffineLinear,
    pub gate_proj: Option<K2HorizonMoVAAffineLinear>,
}

#[derive(Debug)]
pub struct K2HorizonMoVAMoVAAttentionWeights {
    /// One fused q/k projection; v routes through value experts instead.
    pub query_key: K2HorizonMoVAAffineLinear,
    pub query_row_count: usize,
    pub key_value_row_count: usize,
    pub o_proj: K2HorizonMoVAAffineLinear,
    pub gate_proj: Option<K2HorizonMoVAAffineLinear>,
    pub v_router: K2HorizonMoVAAffineLinear,
    pub v_experts: K2HorizonMoVAAffineLinear,
}

#[derive(Debug)]
pub struct K2HorizonMoVADenseMlpWeights {
    pub gate_proj: K2HorizonMoVAAffineLinear,
    pub up_proj: K2HorizonMoVAAffineLinear,
    pub down_proj: K2HorizonMoVAAffineLinear,
}

#[derive(Debug)]
pub struct K2HorizonMoVASparseMlpWeights {
    pub router: K2HorizonMoVAAffineLinear,
    pub switch_gate_up: K2HorizonMoVAAffineLinear,
    pub switch_down: K2HorizonMoVAAffineLinear,
    pub shared_gate_up: K2HorizonMoVAAffineLinear,
    pub shared_down: K2HorizonMoVAAffineLinear,
}

#[derive(Debug)]
pub enum K2HorizonMoVALayerWeights {
    Dense {
        attention: K2HorizonMoVADenseAttentionWeights,
        mlp: K2HorizonMoVADenseMlpWeights,
        input_norm: MlxArray,
        post_attention_norm: MlxArray,
    },
    SparseMixtureOfValues {
        attention: K2HorizonMoVAMoVAAttentionWeights,
        mlp: K2HorizonMoVASparseMlpWeights,
        input_norm: MlxArray,
        post_attention_norm: MlxArray,
    },
}

#[derive(Debug)]
pub struct K2HorizonMoVAWeights {
    pub embed_tokens: K2HorizonMoVAAffineLinear,
    pub layers: Vec<K2HorizonMoVALayerWeights>,
    pub final_norm: MlxArray,
    pub lm_head: K2HorizonMoVAAffineLinear,
    pub(super) expert_payload_bytes: u64,
    pub(super) model_core_payload_bytes: u64,
}

impl K2HorizonMoVAWeights {
    pub fn load(
        runtime: &MlxRuntime,
        validated_artifact: &ValidatedK2HorizonMoVAArtifact,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<Self, K2HorizonMoVAExecutionError> {
        weight_binding::load_weights(runtime, validated_artifact, performance_attribution)
    }
}

impl K2HorizonMoVAWeights {
    /// Routed MoVA and FFN expert stacks currently seated in MLX.
    #[must_use]
    pub const fn expert_payload_bytes(&self) -> u64 {
        self.expert_payload_bytes
    }

    /// Always-resident non-expert payload currently seated in MLX.
    #[must_use]
    pub const fn model_core_payload_bytes(&self) -> u64 {
        self.model_core_payload_bytes
    }

    /// Measured FFN and MoVA stacks for each seated sparse decoder layer.
    #[must_use]
    pub fn sparse_layer_expert_payloads(&self) -> Vec<K2HorizonMoVASparseLayerExpertPayload> {
        self.layers
            .iter()
            .enumerate()
            .filter_map(|(decoder_layer_index, layer_weights)| match layer_weights {
                K2HorizonMoVALayerWeights::SparseMixtureOfValues { attention, mlp, .. } => {
                    Some(K2HorizonMoVASparseLayerExpertPayload {
                        decoder_layer_index,
                        feed_forward_stack_bytes: mlp.expert_payload_bytes(),
                        mixture_of_values_stack_bytes: attention.expert_payload_bytes(),
                    })
                }
                K2HorizonMoVALayerWeights::Dense { .. } => None,
            })
            .collect()
    }
}

impl K2HorizonMoVALayerWeights {
    /// Splits one layer into routed-expert payload versus always-resident core.
    #[must_use]
    pub fn payload_bytes(&self) -> (u64, u64) {
        match self {
            Self::Dense {
                attention,
                mlp,
                input_norm,
                post_attention_norm,
            } => {
                let core_payload_bytes = attention
                    .payload_bytes()
                    .saturating_add(mlp.payload_bytes())
                    .saturating_add(u64::try_from(input_norm.byte_count()).unwrap_or(u64::MAX))
                    .saturating_add(
                        u64::try_from(post_attention_norm.byte_count()).unwrap_or(u64::MAX),
                    );
                (0, core_payload_bytes)
            }
            Self::SparseMixtureOfValues {
                attention,
                mlp,
                input_norm,
                post_attention_norm,
            } => {
                let expert_payload_bytes = attention
                    .expert_payload_bytes()
                    .saturating_add(mlp.expert_payload_bytes());
                let core_payload_bytes = attention
                    .core_payload_bytes()
                    .saturating_add(mlp.core_payload_bytes())
                    .saturating_add(u64::try_from(input_norm.byte_count()).unwrap_or(u64::MAX))
                    .saturating_add(
                        u64::try_from(post_attention_norm.byte_count()).unwrap_or(u64::MAX),
                    );
                (expert_payload_bytes, core_payload_bytes)
            }
        }
    }
}

impl K2HorizonMoVADenseAttentionWeights {
    #[must_use]
    pub fn payload_bytes(&self) -> u64 {
        self.query_key_value
            .payload_bytes()
            .saturating_add(self.o_proj.payload_bytes())
            .saturating_add(
                self.gate_proj
                    .as_ref()
                    .map(K2HorizonMoVAAffineLinear::payload_bytes)
                    .unwrap_or(0),
            )
    }
}

impl K2HorizonMoVADenseMlpWeights {
    #[must_use]
    pub fn payload_bytes(&self) -> u64 {
        self.gate_proj
            .payload_bytes()
            .saturating_add(self.up_proj.payload_bytes())
            .saturating_add(self.down_proj.payload_bytes())
    }
}

impl K2HorizonMoVAMoVAAttentionWeights {
    #[must_use]
    pub fn expert_payload_bytes(&self) -> u64 {
        self.v_experts.payload_bytes()
    }

    #[must_use]
    pub fn core_payload_bytes(&self) -> u64 {
        self.query_key
            .payload_bytes()
            .saturating_add(self.o_proj.payload_bytes())
            .saturating_add(self.v_router.payload_bytes())
            .saturating_add(
                self.gate_proj
                    .as_ref()
                    .map(K2HorizonMoVAAffineLinear::payload_bytes)
                    .unwrap_or(0),
            )
    }
}

impl K2HorizonMoVASparseMlpWeights {
    #[must_use]
    pub fn expert_payload_bytes(&self) -> u64 {
        self.switch_gate_up
            .payload_bytes()
            .saturating_add(self.switch_down.payload_bytes())
    }

    #[must_use]
    pub fn core_payload_bytes(&self) -> u64 {
        self.router
            .payload_bytes()
            .saturating_add(self.shared_gate_up.payload_bytes())
            .saturating_add(self.shared_down.payload_bytes())
    }
}
