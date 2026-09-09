//! Binds stacked affine tensors from loaded safetensors shards.

use std::collections::HashMap;
use std::fs::File;

use astronomical_runtime_integration::{MlxArray, MlxRuntime, MlxSafetensors};

use crate::k2_horizon_mova::artifacts::ValidatedK2HorizonMoVAArtifact;
use crate::k2_horizon_mova::configuration::{K2HorizonMoVAConfig, K2HorizonMoVALayerKind};
use crate::k2_horizon_mova::expert_geometry::K2HorizonMoVASparseLayerExpertPayload;
use crate::{PerformanceAttribution, PerformanceOperation};

use super::affine::K2HorizonMoVAAffineLinear;
use super::error::K2HorizonMoVAExecutionError;

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
    expert_payload_bytes: u64,
    model_core_payload_bytes: u64,
}

impl K2HorizonMoVAWeights {
    pub fn load(
        runtime: &MlxRuntime,
        validated_artifact: &ValidatedK2HorizonMoVAArtifact,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<Self, K2HorizonMoVAExecutionError> {
        let shard_index = validated_artifact.shard_index();
        let shards = performance_attribution.measure_operation(
            PerformanceOperation::ModelSafetensorsMapping,
            |_| -> Result<_, K2HorizonMoVAExecutionError> {
                let mut shards = HashMap::new();
                for shard_file_name in shard_index.shard_file_names() {
                    let shard_path = validated_artifact.model_directory().join(shard_file_name);
                    let shard_file = File::open(&shard_path).map_err(|source| {
                        K2HorizonMoVAExecutionError::InvalidExecution {
                            description: format!("failed to open shard: {source}"),
                        }
                    })?;
                    shards.insert(
                        shard_file_name.clone(),
                        runtime.load_safetensors(shard_file, None)?,
                    );
                }
                Ok(shards)
            },
        )?;
        performance_attribution.measure_operation(PerformanceOperation::ModelTensorBinding, |_| {
            bind_weights(runtime, validated_artifact.config(), shard_index, shards)
        })
    }
}

fn bind_weights(
    runtime: &MlxRuntime,
    config: &K2HorizonMoVAConfig,
    shard_index: &crate::k2_horizon_mova::K2HorizonMoVAShardIndex,
    shards: HashMap<String, MlxSafetensors>,
) -> Result<K2HorizonMoVAWeights, K2HorizonMoVAExecutionError> {
    let embed_tokens = bind_affine(config, shard_index, &shards, "model.embed_tokens", false)?;
    let lm_head = if config.tie_word_embeddings() {
        bind_affine(config, shard_index, &shards, "model.embed_tokens", false)?
    } else {
        bind_affine(config, shard_index, &shards, "lm_head", false)?
    };
    let final_norm = tensor(&shards, shard_index, "model.norm.weight")?;
    let mut layers = Vec::with_capacity(config.num_hidden_layers());
    for decoder_layer_index in 0..config.num_hidden_layers() {
        let prefix = format!("model.layers.{decoder_layer_index}");
        let input_norm = tensor(
            &shards,
            shard_index,
            &format!("{prefix}.input_layernorm.weight"),
        )?;
        let post_attention_norm = tensor(
            &shards,
            shard_index,
            &format!("{prefix}.post_attention_layernorm.weight"),
        )?;
        let gate_proj = if config.attention_gate_func().is_some() {
            Some(bind_affine(
                config,
                shard_index,
                &shards,
                &format!("{prefix}.self_attn.gate_proj"),
                false,
            )?)
        } else {
            None
        };
        let layer = match config.layer_kind(decoder_layer_index) {
            K2HorizonMoVALayerKind::Dense | K2HorizonMoVALayerKind::SparseFeedForward => {
                let query_projection = bind_affine(
                    config,
                    shard_index,
                    &shards,
                    &format!("{prefix}.self_attn.q_proj"),
                    false,
                )?;
                let key_projection = bind_affine(
                    config,
                    shard_index,
                    &shards,
                    &format!("{prefix}.self_attn.k_proj"),
                    false,
                )?;
                let value_projection = bind_affine(
                    config,
                    shard_index,
                    &shards,
                    &format!("{prefix}.self_attn.v_proj"),
                    false,
                )?;
                let query_key_value = K2HorizonMoVAAffineLinear::fuse_output_rows(
                    runtime,
                    &[&query_projection, &key_projection, &value_projection],
                )?
                .ok_or_else(|| K2HorizonMoVAExecutionError::InvalidExecution {
                    description: "K2 Horizon MoVA q/k/v projections must share affine geometry"
                        .to_owned(),
                })?;
                let attention = K2HorizonMoVADenseAttentionWeights {
                    query_key_value,
                    query_row_count: config.num_attention_heads() * config.head_dim(),
                    key_value_row_count: config.num_key_value_heads() * config.head_dim(),
                    o_proj: bind_affine(
                        config,
                        shard_index,
                        &shards,
                        &format!("{prefix}.self_attn.o_proj"),
                        false,
                    )?,
                    gate_proj,
                };
                let mlp = K2HorizonMoVADenseMlpWeights {
                    gate_proj: bind_affine(
                        config,
                        shard_index,
                        &shards,
                        &format!("{prefix}.mlp.gate_proj"),
                        false,
                    )?,
                    up_proj: bind_affine(
                        config,
                        shard_index,
                        &shards,
                        &format!("{prefix}.mlp.up_proj"),
                        false,
                    )?,
                    down_proj: bind_affine(
                        config,
                        shard_index,
                        &shards,
                        &format!("{prefix}.mlp.down_proj"),
                        false,
                    )?,
                };
                K2HorizonMoVALayerWeights::Dense {
                    attention,
                    mlp,
                    input_norm,
                    post_attention_norm,
                }
            }
            K2HorizonMoVALayerKind::SparseMixtureOfValues => {
                let query_projection = bind_affine(
                    config,
                    shard_index,
                    &shards,
                    &format!("{prefix}.self_attn.q_proj"),
                    false,
                )?;
                let key_projection = bind_affine(
                    config,
                    shard_index,
                    &shards,
                    &format!("{prefix}.self_attn.k_proj"),
                    false,
                )?;
                let query_key = K2HorizonMoVAAffineLinear::fuse_output_rows(
                    runtime,
                    &[&query_projection, &key_projection],
                )?
                .ok_or_else(|| K2HorizonMoVAExecutionError::InvalidExecution {
                    description: "K2 Horizon MoVA q/k projections must share affine geometry"
                        .to_owned(),
                })?;
                let attention = K2HorizonMoVAMoVAAttentionWeights {
                    query_key,
                    query_row_count: config.num_attention_heads() * config.head_dim(),
                    key_value_row_count: config.num_key_value_heads() * config.head_dim(),
                    o_proj: bind_affine(
                        config,
                        shard_index,
                        &shards,
                        &format!("{prefix}.self_attn.o_proj"),
                        false,
                    )?,
                    gate_proj,
                    v_router: bind_affine(
                        config,
                        shard_index,
                        &shards,
                        &format!("{prefix}.self_attn.v_router"),
                        config.moe_gate_bias(),
                    )?,
                    v_experts: bind_affine(
                        config,
                        shard_index,
                        &shards,
                        &format!("{prefix}.self_attn.v_experts"),
                        false,
                    )?,
                };
                let switch_gate = bind_affine(
                    config,
                    shard_index,
                    &shards,
                    &format!("{prefix}.mlp.switch_mlp.gate_proj"),
                    false,
                )?;
                let switch_up = bind_affine(
                    config,
                    shard_index,
                    &shards,
                    &format!("{prefix}.mlp.switch_mlp.up_proj"),
                    false,
                )?;
                let shared_gate = bind_affine(
                    config,
                    shard_index,
                    &shards,
                    &format!("{prefix}.mlp.shared_experts.gate_proj"),
                    false,
                )?;
                let shared_up = bind_affine(
                    config,
                    shard_index,
                    &shards,
                    &format!("{prefix}.mlp.shared_experts.up_proj"),
                    false,
                )?;
                let mlp = K2HorizonMoVASparseMlpWeights {
                    router: bind_affine(
                        config,
                        shard_index,
                        &shards,
                        &format!("{prefix}.mlp.gate"),
                        config.moe_gate_bias(),
                    )?,
                    switch_gate_up: fuse_affine(runtime, &switch_gate, &switch_up)?,
                    switch_down: bind_affine(
                        config,
                        shard_index,
                        &shards,
                        &format!("{prefix}.mlp.switch_mlp.down_proj"),
                        false,
                    )?,
                    shared_gate_up: fuse_affine(runtime, &shared_gate, &shared_up)?,
                    shared_down: bind_affine(
                        config,
                        shard_index,
                        &shards,
                        &format!("{prefix}.mlp.shared_experts.down_proj"),
                        false,
                    )?,
                };
                K2HorizonMoVALayerWeights::SparseMixtureOfValues {
                    attention,
                    mlp,
                    input_norm,
                    post_attention_norm,
                }
            }
        };
        layers.push(layer);
    }
    let mut expert_payload_bytes = 0_u64;
    let mut model_core_payload_bytes = embed_tokens
        .payload_bytes()
        .saturating_add(lm_head.payload_bytes())
        .saturating_add(u64::try_from(final_norm.byte_count()).unwrap_or(u64::MAX));
    for layer in &layers {
        let (layer_expert_payload_bytes, layer_core_payload_bytes) = layer.payload_bytes();
        expert_payload_bytes = expert_payload_bytes.saturating_add(layer_expert_payload_bytes);
        model_core_payload_bytes =
            model_core_payload_bytes.saturating_add(layer_core_payload_bytes);
    }
    Ok(K2HorizonMoVAWeights {
        embed_tokens,
        layers,
        final_norm,
        lm_head,
        expert_payload_bytes,
        model_core_payload_bytes,
    })
}

fn fuse_affine(
    runtime: &MlxRuntime,
    gate: &K2HorizonMoVAAffineLinear,
    up: &K2HorizonMoVAAffineLinear,
) -> Result<K2HorizonMoVAAffineLinear, K2HorizonMoVAExecutionError> {
    K2HorizonMoVAAffineLinear::fuse_matching_output_rows(runtime, gate, up)?.ok_or(
        K2HorizonMoVAExecutionError::InvalidExecution {
            description: "K2 Horizon MoVA gate and up projections must share affine geometry"
                .to_owned(),
        },
    )
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

fn bind_affine(
    config: &K2HorizonMoVAConfig,
    shard_index: &crate::k2_horizon_mova::K2HorizonMoVAShardIndex,
    shards: &HashMap<String, MlxSafetensors>,
    module_path: &str,
    include_dense_bias: bool,
) -> Result<K2HorizonMoVAAffineLinear, K2HorizonMoVAExecutionError> {
    let profile = config.affine_profile_for_module(module_path);
    let dense_bias = if include_dense_bias {
        Some(tensor(shards, shard_index, &format!("{module_path}.bias"))?)
    } else {
        None
    };
    Ok(K2HorizonMoVAAffineLinear::new(
        tensor(shards, shard_index, &format!("{module_path}.weight"))?,
        tensor(shards, shard_index, &format!("{module_path}.scales"))?,
        tensor(shards, shard_index, &format!("{module_path}.biases"))?,
        profile.bits(),
        profile.group_size(),
        dense_bias,
    ))
}

fn tensor(
    shards: &HashMap<String, MlxSafetensors>,
    shard_index: &crate::k2_horizon_mova::K2HorizonMoVAShardIndex,
    tensor_name: &str,
) -> Result<MlxArray, K2HorizonMoVAExecutionError> {
    let shard_file_name = shard_index.shard_file_name_for_tensor(tensor_name).ok_or(
        K2HorizonMoVAExecutionError::InvalidExecution {
            description: format!("missing tensor {tensor_name}"),
        },
    )?;
    let shard =
        shards
            .get(shard_file_name)
            .ok_or(K2HorizonMoVAExecutionError::InvalidExecution {
                description: format!("missing shard for {tensor_name}"),
            })?;
    Ok(shard.tensor(tensor_name)?)
}
