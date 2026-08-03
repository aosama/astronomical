use std::collections::HashMap;

use astronomical_runtime_integration::{MlxArray, MlxDtype, MlxRuntime, MlxSafetensors};

use super::super::{
    artifacts::tensor_spec::is_sparse_selected_expert_tensor_name, qwen3_5_moe_mtp_tensor_profiles,
};
use super::decoder_layer_weights::{
    Qwen3_5DecoderLayerMlpWeights, Qwen3_5MoEAffineWeights, Qwen3_5MoEAttentionWeights,
    Qwen3_5MoEDecoderLayerWeights,
};
use super::model::Qwen3_5MoEModel;
use super::weights::{take_full_attention_weights, take_mixture_of_experts_weights, take_tensor};
use super::weights_validation::{validate_bound_tensor, validate_quantized_tensor_bits};
use super::{Qwen3_5MoEConfig, Qwen3_5MoEExecutionError, Qwen3_5MoEShardIndex, Qwen3_5MoEWeights};

const MTP_NORMALIZATION_REPAIR_MARGIN: f32 = 0.4;

/// Resident weights for Qwen's one-layer oQ4e multi-token prediction head.
#[derive(Debug)]
pub(super) struct Qwen3_5MoEMtpWeights {
    pub(super) pre_fc_normalization_embedding_weight: MlxArray,
    pub(super) pre_fc_normalization_hidden_weight: MlxArray,
    pub(super) fusion_projection: Qwen3_5MoEAffineWeights,
    pub(super) decoder_layer_weights: Qwen3_5MoEDecoderLayerWeights,
    pub(super) final_normalization_weight: MlxArray,
    tensor_count: usize,
}

impl Qwen3_5MoEMtpWeights {
    pub(super) fn bind(
        qwen3_5_moe_config: &Qwen3_5MoEConfig,
        shard_index: &Qwen3_5MoEShardIndex,
        model_shards: &[MlxSafetensors],
    ) -> Result<Option<Self>, Qwen3_5MoEExecutionError> {
        let mtp_tensor_profiles = qwen3_5_moe_mtp_tensor_profiles(qwen3_5_moe_config);
        if mtp_tensor_profiles.is_empty() || shard_index.mtp_tensor_count() == 0 {
            return Ok(None);
        }
        let resident_mtp_tensor_profiles = if qwen3_5_moe_config.is_dense_model() {
            mtp_tensor_profiles
        } else {
            mtp_tensor_profiles
                .into_iter()
                .filter(|tensor_profile| {
                    !is_sparse_selected_expert_tensor_name(&tensor_profile.name)
                })
                .collect()
        };
        let mut bound_mtp_tensors = HashMap::with_capacity(resident_mtp_tensor_profiles.len());
        for tensor_profile in &resident_mtp_tensor_profiles {
            let shard_file_name = shard_index
                .shard_file_name_for_mtp_tensor(&tensor_profile.name)
                .ok_or_else(|| Qwen3_5MoEExecutionError::MissingTensor {
                    tensor_name: tensor_profile.name.clone(),
                })?;
            let shard_position = shard_index
                .model_shard_file_names()
                .iter()
                .position(|model_shard_file_name| model_shard_file_name == shard_file_name)
                .ok_or_else(|| Qwen3_5MoEExecutionError::InvalidTensor {
                    tensor_name: tensor_profile.name.clone(),
                    description: "MTP tensor resolves outside loaded model shards",
                })?;
            let model_shard = model_shards.get(shard_position).ok_or_else(|| {
                Qwen3_5MoEExecutionError::InvalidTensor {
                    tensor_name: tensor_profile.name.clone(),
                    description: "MTP model shard is missing",
                }
            })?;
            let bound_tensor = model_shard.tensor(&tensor_profile.name)?;
            validate_bound_tensor(tensor_profile, &bound_tensor)?;
            validate_quantized_tensor_bits(qwen3_5_moe_config, tensor_profile)?;
            bound_mtp_tensors.insert(tensor_profile.name.clone(), bound_tensor);
        }
        let mtp_layer_prefix = "language_model.mtp.layers.0";
        let pre_fc_normalization_embedding_weight = take_tensor(
            &mut bound_mtp_tensors,
            "language_model.mtp.pre_fc_norm_embedding.weight".to_owned(),
        )?;
        let pre_fc_normalization_hidden_weight = take_tensor(
            &mut bound_mtp_tensors,
            "language_model.mtp.pre_fc_norm_hidden.weight".to_owned(),
        )?;
        let fusion_projection = Qwen3_5MoEAffineWeights::NativeBfloat16 {
            weight: take_tensor(
                &mut bound_mtp_tensors,
                "language_model.mtp.fc.weight".to_owned(),
            )?,
        };
        let decoder_layer_weights = Qwen3_5MoEDecoderLayerWeights {
            input_normalization_weight: take_tensor(
                &mut bound_mtp_tensors,
                format!("{mtp_layer_prefix}.input_layernorm.weight"),
            )?,
            attention_weights: Qwen3_5MoEAttentionWeights::Full(take_full_attention_weights(
                &mut bound_mtp_tensors,
                qwen3_5_moe_config,
                mtp_layer_prefix,
            )?),
            post_attention_normalization_weight: take_tensor(
                &mut bound_mtp_tensors,
                format!("{mtp_layer_prefix}.post_attention_layernorm.weight"),
            )?,
            mlp_weights: if qwen3_5_moe_config.is_dense_model() {
                Qwen3_5DecoderLayerMlpWeights::Dense(super::dense_mlp::take_dense_mlp_weights(
                    &mut bound_mtp_tensors,
                    qwen3_5_moe_config,
                    mtp_layer_prefix,
                )?)
            } else {
                Qwen3_5DecoderLayerMlpWeights::Sparse(take_mixture_of_experts_weights(
                    &mut bound_mtp_tensors,
                    qwen3_5_moe_config,
                    mtp_layer_prefix,
                )?)
            },
        };
        let final_normalization_weight = take_tensor(
            &mut bound_mtp_tensors,
            "language_model.mtp.norm.weight".to_owned(),
        )?;
        if let Some(unassigned_tensor_name) = bound_mtp_tensors.keys().next() {
            return Err(Qwen3_5MoEExecutionError::UnassignedTensor {
                tensor_name: unassigned_tensor_name.clone(),
            });
        }
        Ok(Some(Self {
            pre_fc_normalization_embedding_weight,
            pre_fc_normalization_hidden_weight,
            fusion_projection,
            decoder_layer_weights,
            final_normalization_weight,
            tensor_count: resident_mtp_tensor_profiles.len(),
        }))
    }

    pub(super) fn materialize(&self, runtime: &MlxRuntime) -> Result<(), Qwen3_5MoEExecutionError> {
        let mut array_references = Vec::with_capacity(self.tensor_count);
        self.append_array_references(&mut array_references);
        if array_references.len() != self.tensor_count {
            return Err(Qwen3_5MoEExecutionError::TypedTensorCountMismatch {
                actual_tensor_count: array_references.len(),
                expected_tensor_count: self.tensor_count,
            });
        }
        Ok(runtime.evaluate_arrays(&array_references)?)
    }

    pub(super) fn payload_byte_count(&self) -> u64 {
        let mut array_references = Vec::with_capacity(self.tensor_count);
        self.append_array_references(&mut array_references);
        array_references
            .into_iter()
            .map(|tensor| tensor.byte_count() as u64)
            .sum()
    }

    fn append_array_references<'weights>(
        &'weights self,
        array_references: &mut Vec<&'weights MlxArray>,
    ) {
        array_references.push(&self.pre_fc_normalization_embedding_weight);
        array_references.push(&self.pre_fc_normalization_hidden_weight);
        self.fusion_projection
            .append_array_references(array_references);
        self.decoder_layer_weights
            .append_array_references(array_references);
        array_references.push(&self.final_normalization_weight);
    }

    /// Repairs raw Hugging Face MTP RMSNorm gammas without changing trunk weights.
    ///
    /// A raw MTP gamma sits about one below its already converted trunk counterpart.
    /// Comparing to the counterpart avoids an unreliable absolute cutoff and makes a
    /// second repair pass a no-op.
    pub(super) fn repair_raw_normalization_weights(
        &mut self,
        runtime: &MlxRuntime,
        trunk_weights: &Qwen3_5MoEWeights,
    ) -> Result<(), Qwen3_5MoEExecutionError> {
        let (trunk_query_normalization_mean, trunk_key_normalization_mean) =
            full_attention_normalization_means(runtime, trunk_weights)?;
        let trunk_post_attention_normalization_mean = normalization_mean(
            runtime,
            trunk_weights
                .decoder_layer_weights
                .iter()
                .map(|decoder_layer_weights| {
                    &decoder_layer_weights.post_attention_normalization_weight
                }),
        )?;
        let trunk_final_normalization_mean =
            normalization_mean(runtime, [&trunk_weights.final_normalization_weight])?;
        let mut repaired_normalization_count = 0_u8;
        {
            let Qwen3_5MoEAttentionWeights::Full(mtp_full_attention_weights) =
                &mut self.decoder_layer_weights.attention_weights
            else {
                return Err(Qwen3_5MoEExecutionError::InvalidInput {
                    description: "the Qwen MTP head must use full attention",
                });
            };
            repaired_normalization_count += u8::from(repair_normalization_weight_if_raw(
                runtime,
                &mut mtp_full_attention_weights.query_normalization_weight,
                trunk_query_normalization_mean,
            )?);
            repaired_normalization_count += u8::from(repair_normalization_weight_if_raw(
                runtime,
                &mut mtp_full_attention_weights.key_normalization_weight,
                trunk_key_normalization_mean,
            )?);
        }
        repaired_normalization_count += u8::from(repair_normalization_weight_if_raw(
            runtime,
            &mut self
                .decoder_layer_weights
                .post_attention_normalization_weight,
            trunk_post_attention_normalization_mean,
        )?);
        repaired_normalization_count += u8::from(repair_normalization_weight_if_raw(
            runtime,
            &mut self.final_normalization_weight,
            trunk_final_normalization_mean,
        )?);
        if repaired_normalization_count > 0 {
            tracing::info!(
                repaired_normalization_count,
                "repaired raw Qwen MTP RMSNorm weights during model loading"
            );
        }
        Ok(())
    }
}

fn full_attention_normalization_means(
    runtime: &MlxRuntime,
    trunk_weights: &Qwen3_5MoEWeights,
) -> Result<(f32, f32), Qwen3_5MoEExecutionError> {
    let mut query_normalization_weights = Vec::new();
    let mut key_normalization_weights = Vec::new();
    for decoder_layer_weights in &trunk_weights.decoder_layer_weights {
        if let Qwen3_5MoEAttentionWeights::Full(full_attention_weights) =
            &decoder_layer_weights.attention_weights
        {
            query_normalization_weights.push(&full_attention_weights.query_normalization_weight);
            key_normalization_weights.push(&full_attention_weights.key_normalization_weight);
        }
    }
    Ok((
        normalization_mean(runtime, query_normalization_weights)?,
        normalization_mean(runtime, key_normalization_weights)?,
    ))
}

fn normalization_mean<'normalization_weights>(
    runtime: &MlxRuntime,
    normalization_weights: impl IntoIterator<Item = &'normalization_weights MlxArray>,
) -> Result<f32, Qwen3_5MoEExecutionError> {
    let mut normalization_weight_count = 0_usize;
    let mut normalization_weight_mean_sum = 0.0_f64;
    for normalization_weight in normalization_weights {
        let float32_normalization_weight =
            runtime.astype(normalization_weight, MlxDtype::Float32)?;
        let float32_normalization_values = float32_normalization_weight.to_vec_f32()?;
        if float32_normalization_values.is_empty() {
            return Err(Qwen3_5MoEExecutionError::InvalidInput {
                description: "Qwen normalization weight must not be empty",
            });
        }
        let tensor_mean = float32_normalization_values
            .into_iter()
            .map(f64::from)
            .sum::<f64>()
            / normalization_weight.element_count() as f64;
        normalization_weight_mean_sum += tensor_mean;
        normalization_weight_count += 1;
    }
    if normalization_weight_count == 0 {
        return Err(Qwen3_5MoEExecutionError::InvalidInput {
            description: "Qwen MTP normalization repair requires a trunk normalization anchor",
        });
    }
    Ok((normalization_weight_mean_sum / normalization_weight_count as f64) as f32)
}

fn repair_normalization_weight_if_raw(
    runtime: &MlxRuntime,
    mtp_normalization_weight: &mut MlxArray,
    trunk_normalization_mean: f32,
) -> Result<bool, Qwen3_5MoEExecutionError> {
    let mtp_normalization_mean = normalization_mean(runtime, [&*mtp_normalization_weight])?;
    if trunk_normalization_mean - mtp_normalization_mean <= MTP_NORMALIZATION_REPAIR_MARGIN {
        return Ok(false);
    }
    let one_float32 = runtime.array_from_f32(&[1.0], &[])?;
    let one_bfloat16 = runtime.astype(&one_float32, MlxDtype::BFloat16)?;
    *mtp_normalization_weight = runtime.add(mtp_normalization_weight, &one_bfloat16)?;
    Ok(true)
}

impl Qwen3_5MoEModel {
    pub(in crate::qwen3_5_moe) fn resident_model_payload_byte_count(&self) -> u64 {
        self.weights.total_payload_bytes().saturating_add(
            self.mtp_weights
                .as_ref()
                .map_or(0, Qwen3_5MoEMtpWeights::payload_byte_count),
        )
    }
}
