use std::collections::HashMap;

use astronomical_runtime_integration::{MlxArray, MlxDtype, MlxRuntime, MlxSafetensors};

use crate::artifact_validation::{TensorInventory, TensorSourceId};
use crate::qwen3_5::model::decoder_layer_weights::{
    Qwen3_5AffineWeights, Qwen3_5AttentionWeights, Qwen3_5DecoderFeedForwardWeights,
    Qwen3_5DecoderLayerWeights,
};
use crate::qwen3_5::model::weights::{take_full_attention_weights, take_tensor};
use crate::qwen3_5::model::weights_validation::{
    validate_bound_tensor, validate_quantized_tensor_bits,
};
use crate::qwen3_5::model::{Qwen3_5ExecutionError, Qwen3_5Model, Qwen3_5Weights};
use crate::qwen3_5::multi_token_prediction::qwen3_5_mtp_tensor_profiles;
use crate::qwen3_5::{
    Qwen3_5Config, Qwen3_5FeedForwardArchitecture, Qwen3_5MtpArtifactCapability, Qwen3_5ShardIndex,
};
use crate::qwen3_5_moe::artifacts::tensor_spec::is_sparse_selected_expert_tensor_name;
use crate::qwen3_5_moe::bind_qwen3_5_moe_feed_forward_weights;

const MTP_NORMALIZATION_REPAIR_MARGIN: f32 = 0.4;

/// Resident weights for the supported one-layer Qwen multi-token prediction head.
#[derive(Debug)]
pub(crate) struct Qwen3_5MtpWeights {
    pub(super) pre_fc_normalization_embedding_weight: MlxArray,
    pub(super) pre_fc_normalization_hidden_weight: MlxArray,
    pub(super) fusion_projection: Qwen3_5AffineWeights,
    pub(super) decoder_layer_weights: Qwen3_5DecoderLayerWeights,
    pub(super) final_normalization_weight: MlxArray,
    /// Owners for MTP tensors stored outside target-language shards.
    ///
    /// The opaque source identity handles both indexed MTP-only files and architecture sidecars;
    /// tensor binding never needs to reverse-map either category through a file-name position.
    #[allow(dead_code)]
    auxiliary_mtp_sources: HashMap<TensorSourceId, MlxSafetensors>,
    tensor_count: usize,
}

impl Qwen3_5MtpWeights {
    pub(crate) fn bind(
        qwen3_5_config: &Qwen3_5Config,
        shard_index: &Qwen3_5ShardIndex,
        tensor_inventory: &TensorInventory,
        model_shards: &[MlxSafetensors],
        auxiliary_mtp_sources: HashMap<TensorSourceId, MlxSafetensors>,
    ) -> Result<Option<Self>, Qwen3_5ExecutionError> {
        let mtp_tensor_profiles = qwen3_5_mtp_tensor_profiles(qwen3_5_config);
        if mtp_tensor_profiles.is_empty()
            || !mtp_tensor_profiles
                .iter()
                .any(|profile| tensor_inventory.location(&profile.name).is_some())
        {
            return Ok(None);
        }
        let resident_mtp_tensor_profiles = match qwen3_5_config.feed_forward_architecture() {
            Qwen3_5FeedForwardArchitecture::Dense => mtp_tensor_profiles,
            Qwen3_5FeedForwardArchitecture::MixtureOfExperts => mtp_tensor_profiles
                .into_iter()
                .filter(|tensor_profile| {
                    !is_sparse_selected_expert_tensor_name(&tensor_profile.name)
                })
                .collect(),
        };
        let mut bound_mtp_tensors = HashMap::with_capacity(resident_mtp_tensor_profiles.len());
        for tensor_profile in &resident_mtp_tensor_profiles {
            let location = tensor_inventory
                .location(&tensor_profile.name)
                .ok_or_else(|| Qwen3_5ExecutionError::MissingTensor {
                    tensor_name: tensor_profile.name.clone(),
                })?;
            let shard_file_name = shard_index.shard_file_name_for_mtp_tensor(&tensor_profile.name);
            let target_shard_position =
                shard_index
                    .model_shard_file_names()
                    .iter()
                    .position(|model_shard_file_name| {
                        Some(model_shard_file_name.as_str()) == shard_file_name
                    });
            let mtp_tensor_owner = if let Some(target_shard_position) = target_shard_position {
                model_shards.get(target_shard_position)
            } else {
                auxiliary_mtp_sources.get(&location.source_id())
            }
            .ok_or_else(|| Qwen3_5ExecutionError::InvalidTensor {
                tensor_name: tensor_profile.name.clone(),
                description: "MTP tensor resolves outside loaded target and auxiliary sources",
            })?;
            // Canonical profile identity and physical SafeTensors identity may differ.
            let bound_tensor = mtp_tensor_owner.tensor(location.stored_name())?;
            validate_bound_tensor(tensor_profile, &bound_tensor)?;
            validate_quantized_tensor_bits(qwen3_5_config, tensor_profile)?;
            bound_mtp_tensors.insert(tensor_profile.name.clone(), bound_tensor);
        }
        Self::finish_binding(
            qwen3_5_config,
            bound_mtp_tensors,
            auxiliary_mtp_sources,
            resident_mtp_tensor_profiles.len(),
        )
    }

    pub(crate) fn bind_standalone(
        qwen3_5_config: &Qwen3_5Config,
        tensor_inventory: &TensorInventory,
        standalone_sources: HashMap<TensorSourceId, MlxSafetensors>,
    ) -> Result<Option<Self>, Qwen3_5ExecutionError> {
        let mtp_tensor_profiles = qwen3_5_mtp_tensor_profiles(qwen3_5_config);
        let resident_mtp_tensor_profiles = match qwen3_5_config.feed_forward_architecture() {
            Qwen3_5FeedForwardArchitecture::Dense => mtp_tensor_profiles,
            Qwen3_5FeedForwardArchitecture::MixtureOfExperts => mtp_tensor_profiles
                .into_iter()
                .filter(|profile| !is_sparse_selected_expert_tensor_name(&profile.name))
                .collect(),
        };
        let mut bound_mtp_tensors = HashMap::with_capacity(resident_mtp_tensor_profiles.len());
        for tensor_profile in &resident_mtp_tensor_profiles {
            let location = tensor_inventory
                .location(&tensor_profile.name)
                .ok_or_else(|| Qwen3_5ExecutionError::MissingTensor {
                    tensor_name: tensor_profile.name.clone(),
                })?;
            let source = standalone_sources
                .get(&location.source_id())
                .ok_or_else(|| Qwen3_5ExecutionError::InvalidTensor {
                    tensor_name: tensor_profile.name.clone(),
                    description: "standalone MTP tensor source is unavailable",
                })?;
            let bound_tensor = source.tensor(location.stored_name())?;
            validate_bound_tensor(tensor_profile, &bound_tensor)?;
            validate_quantized_tensor_bits(qwen3_5_config, tensor_profile)?;
            bound_mtp_tensors.insert(tensor_profile.name.clone(), bound_tensor);
        }
        Self::finish_binding(
            qwen3_5_config,
            bound_mtp_tensors,
            standalone_sources,
            resident_mtp_tensor_profiles.len(),
        )
    }

    fn finish_binding(
        qwen3_5_config: &Qwen3_5Config,
        mut bound_mtp_tensors: HashMap<String, MlxArray>,
        auxiliary_mtp_sources: HashMap<TensorSourceId, MlxSafetensors>,
        tensor_count: usize,
    ) -> Result<Option<Self>, Qwen3_5ExecutionError> {
        let mtp_layer_prefix = "language_model.mtp.layers.0";
        let pre_fc_normalization_embedding_weight = take_tensor(
            &mut bound_mtp_tensors,
            "language_model.mtp.pre_fc_norm_embedding.weight".to_owned(),
        )?;
        let pre_fc_normalization_hidden_weight = take_tensor(
            &mut bound_mtp_tensors,
            "language_model.mtp.pre_fc_norm_hidden.weight".to_owned(),
        )?;
        let fusion_projection = crate::qwen3_5::model::weights::take_quantized_affine_weights(
            &mut bound_mtp_tensors,
            qwen3_5_config,
            "language_model.mtp.fc",
        )?;
        let decoder_layer_weights = Qwen3_5DecoderLayerWeights {
            input_normalization_weight: take_tensor(
                &mut bound_mtp_tensors,
                format!("{mtp_layer_prefix}.input_layernorm.weight"),
            )?,
            attention_weights: Qwen3_5AttentionWeights::Full(take_full_attention_weights(
                &mut bound_mtp_tensors,
                qwen3_5_config,
                mtp_layer_prefix,
            )?),
            post_attention_normalization_weight: take_tensor(
                &mut bound_mtp_tensors,
                format!("{mtp_layer_prefix}.post_attention_layernorm.weight"),
            )?,
            mlp_weights: match qwen3_5_config.feed_forward_architecture() {
                Qwen3_5FeedForwardArchitecture::Dense => Qwen3_5DecoderFeedForwardWeights::Dense(
                    crate::qwen3_5::dense::mlp::bind_qwen3_5_dense_mlp_weights(
                        &mut bound_mtp_tensors,
                        qwen3_5_config,
                        mtp_layer_prefix,
                    )?,
                ),
                Qwen3_5FeedForwardArchitecture::MixtureOfExperts => {
                    Qwen3_5DecoderFeedForwardWeights::MixtureOfExperts(
                        bind_qwen3_5_moe_feed_forward_weights(
                            &mut bound_mtp_tensors,
                            qwen3_5_config,
                            mtp_layer_prefix,
                        )?,
                    )
                }
            },
        };
        let final_normalization_weight = take_tensor(
            &mut bound_mtp_tensors,
            "language_model.mtp.norm.weight".to_owned(),
        )?;
        if let Some(unassigned_tensor_name) = bound_mtp_tensors.keys().next() {
            return Err(Qwen3_5ExecutionError::UnassignedTensor {
                tensor_name: unassigned_tensor_name.clone(),
            });
        }
        Ok(Some(Self {
            pre_fc_normalization_embedding_weight,
            pre_fc_normalization_hidden_weight,
            fusion_projection,
            decoder_layer_weights,
            final_normalization_weight,
            auxiliary_mtp_sources,
            tensor_count,
        }))
    }

    pub(crate) fn materialize(&self, runtime: &MlxRuntime) -> Result<(), Qwen3_5ExecutionError> {
        let mut array_references = Vec::with_capacity(self.tensor_count);
        self.append_array_references(&mut array_references);
        if array_references.len() != self.tensor_count {
            return Err(Qwen3_5ExecutionError::TypedTensorCountMismatch {
                actual_tensor_count: array_references.len(),
                expected_tensor_count: self.tensor_count,
            });
        }
        Ok(runtime.evaluate_arrays(&array_references)?)
    }

    pub(crate) fn payload_byte_count(&self) -> u64 {
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
    pub(crate) fn repair_raw_normalization_weights(
        &mut self,
        runtime: &MlxRuntime,
        trunk_weights: &Qwen3_5Weights,
    ) -> Result<(), Qwen3_5ExecutionError> {
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
            let Qwen3_5AttentionWeights::Full(mtp_full_attention_weights) =
                &mut self.decoder_layer_weights.attention_weights
            else {
                return Err(Qwen3_5ExecutionError::InvalidInput {
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
    trunk_weights: &Qwen3_5Weights,
) -> Result<(f32, f32), Qwen3_5ExecutionError> {
    let mut query_normalization_weights = Vec::new();
    let mut key_normalization_weights = Vec::new();
    for decoder_layer_weights in &trunk_weights.decoder_layer_weights {
        if let Qwen3_5AttentionWeights::Full(full_attention_weights) =
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
) -> Result<f32, Qwen3_5ExecutionError> {
    let mut normalization_weight_count = 0_usize;
    let mut normalization_weight_mean_sum = 0.0_f64;
    for normalization_weight in normalization_weights {
        let float32_normalization_weight =
            runtime.astype(normalization_weight, MlxDtype::Float32)?;
        let float32_normalization_values = float32_normalization_weight.to_vec_f32()?;
        if float32_normalization_values.is_empty() {
            return Err(Qwen3_5ExecutionError::InvalidInput {
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
        return Err(Qwen3_5ExecutionError::InvalidInput {
            description: "Qwen MTP normalization repair requires a trunk normalization anchor",
        });
    }
    Ok((normalization_weight_mean_sum / normalization_weight_count as f64) as f32)
}

fn repair_normalization_weight_if_raw(
    runtime: &MlxRuntime,
    mtp_normalization_weight: &mut MlxArray,
    trunk_normalization_mean: f32,
) -> Result<bool, Qwen3_5ExecutionError> {
    let mtp_normalization_mean = normalization_mean(runtime, [&*mtp_normalization_weight])?;
    if trunk_normalization_mean - mtp_normalization_mean <= MTP_NORMALIZATION_REPAIR_MARGIN {
        return Ok(false);
    }
    let one_float32 = runtime.array_from_f32(&[1.0], &[])?;
    // Standalone publishers may retain F16, BF16, or F32 normalization storage;
    // matching the source avoids an implicit precision change during one-time repair.
    let one_source_dtype = runtime.astype(&one_float32, mtp_normalization_weight.dtype())?;
    *mtp_normalization_weight = runtime.add(mtp_normalization_weight, &one_source_dtype)?;
    Ok(true)
}

impl Qwen3_5Model {
    pub(crate) fn resident_model_payload_byte_count(&self) -> u64 {
        self.weights.total_payload_bytes().saturating_add(
            self.mtp_weights
                .as_ref()
                .map_or(0, Qwen3_5MtpWeights::payload_byte_count),
        )
    }
}

pub(crate) fn bind_optional_weights(
    bind_mtp_weights: bool,
    mtp_artifact_capability: &Qwen3_5MtpArtifactCapability,
    qwen3_5_config: &Qwen3_5Config,
    shard_index: &Qwen3_5ShardIndex,
    tensor_inventory: &TensorInventory,
    model_shards: &[MlxSafetensors],
    auxiliary_mtp_sources: HashMap<TensorSourceId, MlxSafetensors>,
    target_weights: &Qwen3_5Weights,
    runtime: &MlxRuntime,
) -> Option<Qwen3_5MtpWeights> {
    if !bind_mtp_weights || !mtp_artifact_capability.is_mtp_capable() {
        return None;
    }
    let mut mtp_weights = match Qwen3_5MtpWeights::bind(
        qwen3_5_config,
        shard_index,
        tensor_inventory,
        model_shards,
        auxiliary_mtp_sources,
    ) {
        Ok(mtp_weights) => mtp_weights,
        Err(mtp_weight_binding_error) => {
            tracing::warn!(
                error = %mtp_weight_binding_error,
                "optional MTP weight binding failed; serving target-only"
            );
            None
        }
    };
    if let Some(bound_mtp_weights) = mtp_weights.as_mut()
        && let Err(mtp_normalization_repair_error) =
            bound_mtp_weights.repair_raw_normalization_weights(runtime, target_weights)
    {
        tracing::warn!(
            error = %mtp_normalization_repair_error,
            "optional MTP normalization repair failed; serving target-only"
        );
        mtp_weights = None;
    }
    if mtp_weights.is_none()
        && let Err(mlx_allocator_cleanup_error) = runtime.clear_allocator_cache()
    {
        tracing::warn!(
            error = %mlx_allocator_cleanup_error,
            "failed to reclaim allocator memory after optional MTP initialization failure"
        );
    }
    mtp_weights
}

pub(crate) fn materialize_optional_weights(
    model: &mut Qwen3_5Model,
) -> Result<(), Qwen3_5ExecutionError> {
    let Some(mtp_weights) = model.mtp_weights.as_ref() else {
        return Ok(());
    };
    if let Err(mtp_materialization_error) = mtp_weights.materialize(&model.runtime) {
        model.mtp_weights = None;
        return Err(mtp_materialization_error);
    }
    Ok(())
}
