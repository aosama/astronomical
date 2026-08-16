use std::collections::HashMap;

use astronomical_runtime_integration::{MlxArray, MlxRuntime, MlxSafetensors};

use crate::qwen3_5_moe::bind_qwen3_5_moe_feed_forward_weights;
use crate::{PerformanceAttribution, PerformanceOperation};

use super::weights_validation::{validate_bound_tensor, validate_quantized_tensor_bits};
use super::{
    Qwen3_5Config, Qwen3_5ExecutionError, Qwen3_5FeedForwardArchitecture, ValidatedQwen3_5Artifact,
    decoder_layer_weights::{
        Qwen3_5AffineWeights, Qwen3_5AttentionWeights, Qwen3_5DecoderFeedForwardWeights,
        Qwen3_5DecoderLayerWeights, Qwen3_5FullAttentionWeights, Qwen3_5LinearAttentionWeights,
    },
    qwen3_5_resident_language_tensor_profiles,
};

/// Strict typed ownership for the indexed executable Qwen3.5 language shards.
#[derive(Debug)]
pub struct Qwen3_5Weights {
    pub(crate) embedding_weights: Qwen3_5AffineWeights,
    pub(crate) decoder_layer_weights: Vec<Qwen3_5DecoderLayerWeights>,
    pub(crate) final_normalization_weight: MlxArray,
    pub(crate) language_model_head_weights: Qwen3_5AffineWeights,
    has_tied_embeddings: bool,
    model_shards: Vec<MlxSafetensors>,
    tensor_count: usize,
    total_payload_bytes: u64,
}

impl Qwen3_5Weights {
    /// Loads and binds model weights without diagnostic performance attribution.
    pub fn load(
        runtime: &MlxRuntime,
        validated_artifact: ValidatedQwen3_5Artifact,
    ) -> Result<Self, Qwen3_5ExecutionError> {
        let mut disabled_performance_attribution = PerformanceAttribution::disabled();
        Self::load_with_performance_attribution(
            runtime,
            validated_artifact,
            &mut disabled_performance_attribution,
        )
    }

    /// Consumes validated descriptors and binds every executable tensor exactly once.
    pub fn load_with_performance_attribution(
        runtime: &MlxRuntime,
        mut validated_artifact: ValidatedQwen3_5Artifact,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<Self, Qwen3_5ExecutionError> {
        let qwen3_5_config = validated_artifact.config().clone();
        let shard_index = validated_artifact.shard_index().clone();
        let model_shard_source_ids =
            validated_artifact.source_ids_for_file_names(shard_index.model_shard_file_names())?;
        let model_shard_files =
            validated_artifact.take_safetensors_sources(&model_shard_source_ids)?;
        let model_shards = performance_attribution.measure_operation(
            PerformanceOperation::ModelSafetensorsMapping,
            |performance_attribution| -> Result<_, Qwen3_5ExecutionError> {
                let mut model_shards = Vec::with_capacity(model_shard_files.len());
                let positional_file_read_metrics =
                    performance_attribution.positional_file_read_metrics();
                for model_shard_file in model_shard_files {
                    model_shards.push(
                        runtime.load_safetensors(
                            model_shard_file.into_file(),
                            positional_file_read_metrics
                                .as_ref()
                                .map(std::sync::Arc::clone),
                        )?,
                    );
                }
                Ok(model_shards)
            },
        )?;
        performance_attribution.measure_operation(
            PerformanceOperation::ModelTensorBinding,
            |_performance_attribution| {
                Self::bind_from_model_shards(&qwen3_5_config, &shard_index, model_shards)
            },
        )
    }

    /// Binds language tensors from pre-loaded model shard safetensors.
    ///
    /// Used for models with embedded vision where vision tensors are extracted
    /// from the same model shards before language weight binding. The
    /// caller is responsible for loading the shard files and extracting vision
    /// tensors first.
    pub fn bind_from_model_shards(
        qwen3_5_config: &Qwen3_5Config,
        shard_index: &super::Qwen3_5ShardIndex,
        model_shards: Vec<MlxSafetensors>,
    ) -> Result<Self, Qwen3_5ExecutionError> {
        let language_tensor_profiles = qwen3_5_resident_language_tensor_profiles(qwen3_5_config);
        let mut bound_tensors = HashMap::with_capacity(language_tensor_profiles.len());
        let mut actual_payload_bytes = 0_u64;
        for tensor_profile in &language_tensor_profiles {
            let shard_file_name = shard_index
                .shard_file_name_for_tensor(&tensor_profile.name)
                .ok_or_else(|| Qwen3_5ExecutionError::MissingTensor {
                    tensor_name: tensor_profile.name.clone(),
                })?;
            let shard_position = shard_index
                .model_shard_file_names()
                .iter()
                .position(|indexed_shard_file_name| *indexed_shard_file_name == shard_file_name)
                .ok_or_else(|| Qwen3_5ExecutionError::InvalidTensor {
                    tensor_name: tensor_profile.name.clone(),
                    description: "tensor resolves outside the loaded model shards",
                })?;
            let model_shard = model_shards.get(shard_position).ok_or_else(|| {
                Qwen3_5ExecutionError::InvalidTensor {
                    tensor_name: tensor_profile.name.clone(),
                    description: "loaded model shard is missing",
                }
            })?;
            let tensor = model_shard.tensor(&tensor_profile.name)?;
            validate_bound_tensor(tensor_profile, &tensor)?;
            validate_quantized_tensor_bits(qwen3_5_config, tensor_profile)?;
            let tensor_payload_bytes = u64::try_from(tensor.byte_count()).map_err(|_| {
                Qwen3_5ExecutionError::InvalidTensor {
                    tensor_name: tensor_profile.name.clone(),
                    description: "tensor payload byte count exceeds the u64 range",
                }
            })?;
            actual_payload_bytes = actual_payload_bytes
                .checked_add(tensor_payload_bytes)
                .ok_or(Qwen3_5ExecutionError::InvalidTensor {
                    tensor_name: tensor_profile.name.clone(),
                    description: "total tensor payload byte count overflowed",
                })?;
            if bound_tensors
                .insert(tensor_profile.name.clone(), tensor)
                .is_some()
            {
                return Err(Qwen3_5ExecutionError::InvalidTensor {
                    tensor_name: tensor_profile.name.clone(),
                    description: "tensor was bound more than once",
                });
            }
        }

        let tensor_count = bound_tensors.len();
        let embedding_weights = take_quantized_affine_weights(
            &mut bound_tensors,
            qwen3_5_config,
            "language_model.model.embed_tokens",
        )?;
        let decoder_layer_count = qwen3_5_config.layer_count() as usize;
        let mut decoder_layer_weights = Vec::with_capacity(decoder_layer_count);
        for decoder_layer_index in 0..decoder_layer_count {
            decoder_layer_weights.push(take_decoder_layer_weights(
                &mut bound_tensors,
                qwen3_5_config,
                decoder_layer_index,
            )?);
        }
        let final_normalization_weight = take_tensor(
            &mut bound_tensors,
            "language_model.model.norm.weight".to_owned(),
        )?;
        let language_model_head_weights = if qwen3_5_config.has_tied_embeddings() {
            embedding_weights.retained_reference()?
        } else {
            take_quantized_affine_weights(
                &mut bound_tensors,
                qwen3_5_config,
                "language_model.lm_head",
            )?
        };
        if let Some(unassigned_tensor_name) = bound_tensors.keys().next() {
            return Err(Qwen3_5ExecutionError::UnassignedTensor {
                tensor_name: unassigned_tensor_name.clone(),
            });
        }

        Ok(Self {
            embedding_weights,
            decoder_layer_weights,
            final_normalization_weight,
            language_model_head_weights,
            has_tied_embeddings: qwen3_5_config.has_tied_embeddings(),
            model_shards,
            tensor_count,
            total_payload_bytes: actual_payload_bytes,
        })
    }

    /// Returns the number of retained descriptor-backed model maps.
    #[must_use]
    pub fn shard_count(&self) -> usize {
        self.model_shards.len()
    }

    pub(crate) fn model_shards(&self) -> &[MlxSafetensors] {
        &self.model_shards
    }

    /// Returns the number of exactly bound executable tensors.
    #[must_use]
    pub const fn tensor_count(&self) -> usize {
        self.tensor_count
    }

    /// Returns the number of typed decoder layers bound for execution.
    #[must_use]
    pub fn decoder_layer_count(&self) -> usize {
        self.decoder_layer_weights.len()
    }

    /// Returns the complete bound language payload size without evaluating it.
    #[must_use]
    pub const fn total_payload_bytes(&self) -> u64 {
        self.total_payload_bytes
    }

    pub(crate) fn materialize(&self, runtime: &MlxRuntime) -> Result<(), Qwen3_5ExecutionError> {
        let mut bound_tensor_references = Vec::with_capacity(self.tensor_count);
        self.embedding_weights
            .append_array_references(&mut bound_tensor_references);
        for decoder_layer_weights in &self.decoder_layer_weights {
            decoder_layer_weights.append_array_references(&mut bound_tensor_references);
        }
        bound_tensor_references.push(&self.final_normalization_weight);
        if !self.has_tied_embeddings {
            self.language_model_head_weights
                .append_array_references(&mut bound_tensor_references);
        }
        if bound_tensor_references.len() != self.tensor_count {
            return Err(Qwen3_5ExecutionError::TypedTensorCountMismatch {
                actual_tensor_count: bound_tensor_references.len(),
                expected_tensor_count: self.tensor_count,
            });
        }
        Ok(runtime.evaluate_arrays(&bound_tensor_references)?)
    }
}

fn take_decoder_layer_weights(
    bound_tensors: &mut HashMap<String, MlxArray>,
    qwen3_5_config: &Qwen3_5Config,
    decoder_layer_index: usize,
) -> Result<Qwen3_5DecoderLayerWeights, Qwen3_5ExecutionError> {
    let decoder_layer_prefix = format!("language_model.model.layers.{decoder_layer_index}");
    let input_normalization_weight = take_tensor(
        bound_tensors,
        format!("{decoder_layer_prefix}.input_layernorm.weight"),
    )?;
    let attention_weights = if qwen3_5_config.decoder_layer_is_full_attention(decoder_layer_index) {
        Qwen3_5AttentionWeights::Full(take_full_attention_weights(
            bound_tensors,
            qwen3_5_config,
            &decoder_layer_prefix,
        )?)
    } else {
        Qwen3_5AttentionWeights::Linear(take_linear_attention_weights(
            bound_tensors,
            qwen3_5_config,
            &decoder_layer_prefix,
        )?)
    };
    let post_attention_normalization_weight = take_tensor(
        bound_tensors,
        format!("{decoder_layer_prefix}.post_attention_layernorm.weight"),
    )?;
    let mlp_weights = match qwen3_5_config.feed_forward_architecture() {
        Qwen3_5FeedForwardArchitecture::Dense => Qwen3_5DecoderFeedForwardWeights::Dense(
            crate::qwen3_5::dense::mlp::bind_qwen3_5_dense_mlp_weights(
                bound_tensors,
                qwen3_5_config,
                &decoder_layer_prefix,
            )?,
        ),
        Qwen3_5FeedForwardArchitecture::MixtureOfExperts => {
            Qwen3_5DecoderFeedForwardWeights::MixtureOfExperts(
                bind_qwen3_5_moe_feed_forward_weights(
                    bound_tensors,
                    qwen3_5_config,
                    &decoder_layer_prefix,
                )?,
            )
        }
    };
    Ok(Qwen3_5DecoderLayerWeights {
        input_normalization_weight,
        attention_weights,
        post_attention_normalization_weight,
        mlp_weights,
    })
}

fn take_linear_attention_weights(
    bound_tensors: &mut HashMap<String, MlxArray>,
    qwen3_5_config: &Qwen3_5Config,
    decoder_layer_prefix: &str,
) -> Result<Qwen3_5LinearAttentionWeights, Qwen3_5ExecutionError> {
    let linear_attention_prefix = format!("{decoder_layer_prefix}.linear_attn");
    Ok(Qwen3_5LinearAttentionWeights {
        input_queries_keys_values_projection: take_quantized_affine_weights(
            bound_tensors,
            qwen3_5_config,
            &format!("{linear_attention_prefix}.in_proj_qkv"),
        )?,
        output_gate_projection: take_quantized_affine_weights(
            bound_tensors,
            qwen3_5_config,
            &format!("{linear_attention_prefix}.in_proj_z"),
        )?,
        update_rate_projection: take_quantized_affine_weights(
            bound_tensors,
            qwen3_5_config,
            &format!("{linear_attention_prefix}.in_proj_b"),
        )?,
        decay_interval_projection: take_quantized_affine_weights(
            bound_tensors,
            qwen3_5_config,
            &format!("{linear_attention_prefix}.in_proj_a"),
        )?,
        convolution_weight: take_tensor(
            bound_tensors,
            format!("{linear_attention_prefix}.conv1d.weight"),
        )?,
        decay_interval_bias: take_tensor(
            bound_tensors,
            format!("{linear_attention_prefix}.dt_bias"),
        )?,
        decay_rate_logarithm: take_tensor(
            bound_tensors,
            format!("{linear_attention_prefix}.A_log"),
        )?,
        normalization_weight: take_tensor(
            bound_tensors,
            format!("{linear_attention_prefix}.norm.weight"),
        )?,
        output_projection: take_quantized_affine_weights(
            bound_tensors,
            qwen3_5_config,
            &format!("{linear_attention_prefix}.out_proj"),
        )?,
    })
}

pub(crate) fn take_full_attention_weights(
    bound_tensors: &mut HashMap<String, MlxArray>,
    qwen3_5_config: &Qwen3_5Config,
    decoder_layer_prefix: &str,
) -> Result<Qwen3_5FullAttentionWeights, Qwen3_5ExecutionError> {
    let full_attention_prefix = format!("{decoder_layer_prefix}.self_attn");
    Ok(Qwen3_5FullAttentionWeights {
        query_projection: take_quantized_affine_weights(
            bound_tensors,
            qwen3_5_config,
            &format!("{full_attention_prefix}.q_proj"),
        )?,
        key_projection: take_quantized_affine_weights(
            bound_tensors,
            qwen3_5_config,
            &format!("{full_attention_prefix}.k_proj"),
        )?,
        value_projection: take_quantized_affine_weights(
            bound_tensors,
            qwen3_5_config,
            &format!("{full_attention_prefix}.v_proj"),
        )?,
        output_projection: take_quantized_affine_weights(
            bound_tensors,
            qwen3_5_config,
            &format!("{full_attention_prefix}.o_proj"),
        )?,
        query_normalization_weight: take_tensor(
            bound_tensors,
            format!("{full_attention_prefix}.q_norm.weight"),
        )?,
        key_normalization_weight: take_tensor(
            bound_tensors,
            format!("{full_attention_prefix}.k_norm.weight"),
        )?,
    })
}

pub(crate) fn take_quantized_affine_weights(
    bound_tensors: &mut HashMap<String, MlxArray>,
    qwen3_5_config: &Qwen3_5Config,
    module_name: &str,
) -> Result<Qwen3_5AffineWeights, Qwen3_5ExecutionError> {
    let quantization_profile = qwen3_5_config.quantization_profile_for_module(module_name);
    if quantization_profile.is_unquantized() {
        return Ok(Qwen3_5AffineWeights::NativeBfloat16 {
            weight: take_tensor(bound_tensors, format!("{module_name}.weight"))?,
        });
    }
    let quantization_bits = i32::try_from(quantization_profile.bits).map_err(|_| {
        Qwen3_5ExecutionError::InvalidTensor {
            tensor_name: format!("{module_name}.weight"),
            description: "quantization bit width exceeds the MLX integer range",
        }
    })?;
    let quantization_group_size = i32::try_from(quantization_profile.group_size).map_err(|_| {
        Qwen3_5ExecutionError::InvalidTensor {
            tensor_name: format!("{module_name}.weight"),
            description: "quantization group size exceeds the MLX integer range",
        }
    })?;
    Ok(Qwen3_5AffineWeights::Quantized {
        packed_weight: take_tensor(bound_tensors, format!("{module_name}.weight"))?,
        quantization_scales: take_tensor(bound_tensors, format!("{module_name}.scales"))?,
        quantization_biases: take_tensor(bound_tensors, format!("{module_name}.biases"))?,
        quantization_bits,
        quantization_group_size,
    })
}

pub(crate) fn take_tensor(
    bound_tensors: &mut HashMap<String, MlxArray>,
    tensor_name: String,
) -> Result<MlxArray, Qwen3_5ExecutionError> {
    bound_tensors
        .remove(&tensor_name)
        .ok_or(Qwen3_5ExecutionError::MissingTensor { tensor_name })
}
