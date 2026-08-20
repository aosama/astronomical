//! BF16 tensor binding with complete or descriptor-streamed layer ownership.

use std::collections::{BTreeMap, VecDeque};
use std::fs::File;

use astronomical_runtime_integration::{MlxArray, MlxDtype, MlxRuntime, MlxSafetensors};

use crate::{PerformanceAttribution, PerformanceOperation, ValidatedWeightsFile};

use super::super::Flux2KleinResidencyMode;
use super::error::Flux2KleinTextConditioningError;

pub(super) const HIDDEN_WIDTH: i32 = 2_560;
pub(super) const INTERMEDIATE_WIDTH: i32 = 9_728;
pub(super) const QUERY_HEAD_COUNT: i32 = 32;
pub(super) const KEY_VALUE_HEAD_COUNT: i32 = 8;
pub(super) const HEAD_WIDTH: i32 = 128;
pub(super) const EXECUTED_LAYER_COUNT: usize = 27;
const VOCABULARY_SIZE: i32 = 151_936;

#[derive(Debug)]
pub(super) struct Flux2KleinTextWeights {
    pub(super) embedding: Option<MlxArray>,
    complete_layers: VecDeque<Flux2KleinDecoderLayerWeights>,
    descriptor_source: Option<Flux2KleinTextDescriptorSource>,
    streamed_layer_is_retained: bool,
    peak_streamed_layer_count: usize,
}

#[derive(Debug)]
pub(super) struct Flux2KleinDecoderLayerWeights {
    pub(super) input_norm: MlxArray,
    pub(super) query: MlxArray,
    pub(super) key: MlxArray,
    pub(super) value: MlxArray,
    pub(super) output: MlxArray,
    pub(super) query_norm: MlxArray,
    pub(super) key_norm: MlxArray,
    pub(super) post_attention_norm: MlxArray,
    pub(super) gate: MlxArray,
    pub(super) up: MlxArray,
    pub(super) down: MlxArray,
}

#[derive(Debug)]
struct Flux2KleinTextDescriptorSource {
    shard_files: Vec<File>,
}

impl Flux2KleinTextWeights {
    pub(super) fn load(
        runtime: &MlxRuntime,
        text_shard_files: BTreeMap<String, ValidatedWeightsFile>,
        residency_mode: Flux2KleinResidencyMode,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<Self, Flux2KleinTextConditioningError> {
        let descriptor_source = Flux2KleinTextDescriptorSource {
            shard_files: text_shard_files
                .into_values()
                .map(ValidatedWeightsFile::into_file)
                .collect(),
        };
        match residency_mode {
            Flux2KleinResidencyMode::Complete => {
                let model_shards = descriptor_source.map(runtime, performance_attribution)?;
                let weights = performance_attribution.measure_operation(
                    PerformanceOperation::ImageTextComponentLoading,
                    |_performance_attribution| bind_complete_weights(&model_shards),
                )?;
                performance_attribution.measure_operation(
                    PerformanceOperation::ImageTextComponentLoading,
                    |_performance_attribution| weights.materialize_complete(runtime),
                )?;
                drop(model_shards);
                Ok(weights)
            }
            Flux2KleinResidencyMode::Streamed => {
                let model_shards = descriptor_source.map(runtime, performance_attribution)?;
                let embedding = performance_attribution.measure_operation(
                    PerformanceOperation::ImageTextComponentLoading,
                    |_performance_attribution| bind_embedding(&model_shards),
                )?;
                performance_attribution.measure_operation(
                    PerformanceOperation::ImageTextComponentLoading,
                    |_performance_attribution| runtime.evaluate_arrays(&[&embedding]),
                )?;
                drop(model_shards);
                Ok(Self {
                    embedding: Some(embedding),
                    complete_layers: VecDeque::new(),
                    descriptor_source: Some(descriptor_source),
                    streamed_layer_is_retained: false,
                    peak_streamed_layer_count: 0,
                })
            }
        }
    }

    pub(super) const fn is_streamed(&self) -> bool {
        self.descriptor_source.is_some()
    }

    pub(super) fn take_layer(
        &mut self,
        runtime: &MlxRuntime,
        layer_index: usize,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<Flux2KleinDecoderLayerWeights, Flux2KleinTextConditioningError> {
        if let Some(descriptor_source) = self.descriptor_source.as_ref() {
            if self.streamed_layer_is_retained {
                return Err(Flux2KleinTextConditioningError::WeightsUnavailable);
            }
            let descriptor_maps = descriptor_source.map(runtime, performance_attribution)?;
            let layer_weights = performance_attribution.measure_operation(
                PerformanceOperation::ImageTextComponentLoading,
                |_performance_attribution| bind_decoder_layer(&descriptor_maps, layer_index),
            )?;
            performance_attribution.measure_operation(
                PerformanceOperation::ImageTextComponentLoading,
                |_performance_attribution| layer_weights.materialize(runtime),
            )?;
            // The map retains every lazy array handle. Dropping it after the selected
            // page is durable prevents prior layers from accumulating in map ownership.
            drop(descriptor_maps);
            self.streamed_layer_is_retained = true;
            self.peak_streamed_layer_count = self.peak_streamed_layer_count.max(1);
            Ok(layer_weights)
        } else {
            self.complete_layers
                .pop_front()
                .ok_or(Flux2KleinTextConditioningError::WeightsUnavailable)
        }
    }

    pub(super) fn release_layer(&mut self, layer_weights: Flux2KleinDecoderLayerWeights) {
        drop(layer_weights);
        self.streamed_layer_is_retained = false;
    }

    pub(super) fn release_descriptor_source(
        &mut self,
    ) -> Result<(), Flux2KleinTextConditioningError> {
        if self.streamed_layer_is_retained
            || (self.is_streamed() && self.peak_streamed_layer_count != 1)
        {
            return Err(Flux2KleinTextConditioningError::WeightsUnavailable);
        }
        self.descriptor_source = None;
        Ok(())
    }

    fn materialize_complete(
        &self,
        runtime: &MlxRuntime,
    ) -> Result<(), Flux2KleinTextConditioningError> {
        let mut arrays = Vec::with_capacity(1 + self.complete_layers.len() * 11);
        if let Some(embedding) = &self.embedding {
            arrays.push(embedding);
        }
        for layer in &self.complete_layers {
            layer.append_arrays(&mut arrays);
        }
        runtime.evaluate_arrays(&arrays)?;
        Ok(())
    }
}

impl Flux2KleinDecoderLayerWeights {
    fn materialize(&self, runtime: &MlxRuntime) -> Result<(), Flux2KleinTextConditioningError> {
        let mut arrays = Vec::with_capacity(11);
        self.append_arrays(&mut arrays);
        runtime.evaluate_arrays(&arrays)?;
        Ok(())
    }

    fn append_arrays<'weights>(&'weights self, arrays: &mut Vec<&'weights MlxArray>) {
        arrays.extend([
            &self.input_norm,
            &self.query,
            &self.key,
            &self.value,
            &self.output,
            &self.query_norm,
            &self.key_norm,
            &self.post_attention_norm,
            &self.gate,
            &self.up,
            &self.down,
        ]);
    }
}

fn bind_complete_weights(
    model_shards: &[MlxSafetensors],
) -> Result<Flux2KleinTextWeights, Flux2KleinTextConditioningError> {
    let embedding = bind_embedding(model_shards)?;
    let mut complete_layers = VecDeque::with_capacity(EXECUTED_LAYER_COUNT);
    for layer_index in 0..EXECUTED_LAYER_COUNT {
        complete_layers.push_back(bind_decoder_layer(model_shards, layer_index)?);
    }
    Ok(Flux2KleinTextWeights {
        embedding: Some(embedding),
        complete_layers,
        descriptor_source: None,
        streamed_layer_is_retained: false,
        peak_streamed_layer_count: 0,
    })
}

impl Flux2KleinTextDescriptorSource {
    fn map(
        &self,
        runtime: &MlxRuntime,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<Vec<MlxSafetensors>, Flux2KleinTextConditioningError> {
        performance_attribution.measure_operation(
            PerformanceOperation::ImageTextComponentMapping,
            |performance_attribution| {
                let positional_file_read_metrics =
                    performance_attribution.positional_file_read_metrics();
                self.shard_files
                    .iter()
                    .map(|shard_file| {
                        let cloned_file = shard_file
                            .try_clone()
                            .map_err(Flux2KleinTextConditioningError::WeightDescriptorIo)?;
                        runtime
                            .load_safetensors(
                                cloned_file,
                                positional_file_read_metrics
                                    .as_ref()
                                    .map(std::sync::Arc::clone),
                            )
                            .map_err(Flux2KleinTextConditioningError::from)
                    })
                    .collect::<Result<Vec<_>, _>>()
            },
        )
    }
}

fn bind_embedding(
    model_shards: &[MlxSafetensors],
) -> Result<MlxArray, Flux2KleinTextConditioningError> {
    load_bf16_tensor(
        model_shards,
        "model.embed_tokens.weight",
        &[VOCABULARY_SIZE, HIDDEN_WIDTH],
    )
}

fn bind_decoder_layer(
    model_shards: &[MlxSafetensors],
    layer_index: usize,
) -> Result<Flux2KleinDecoderLayerWeights, Flux2KleinTextConditioningError> {
    let prefix = format!("model.layers.{layer_index}");
    Ok(Flux2KleinDecoderLayerWeights {
        input_norm: load_bf16_tensor(
            model_shards,
            &format!("{prefix}.input_layernorm.weight"),
            &[HIDDEN_WIDTH],
        )?,
        query: load_bf16_tensor(
            model_shards,
            &format!("{prefix}.self_attn.q_proj.weight"),
            &[QUERY_HEAD_COUNT * HEAD_WIDTH, HIDDEN_WIDTH],
        )?,
        key: load_bf16_tensor(
            model_shards,
            &format!("{prefix}.self_attn.k_proj.weight"),
            &[KEY_VALUE_HEAD_COUNT * HEAD_WIDTH, HIDDEN_WIDTH],
        )?,
        value: load_bf16_tensor(
            model_shards,
            &format!("{prefix}.self_attn.v_proj.weight"),
            &[KEY_VALUE_HEAD_COUNT * HEAD_WIDTH, HIDDEN_WIDTH],
        )?,
        output: load_bf16_tensor(
            model_shards,
            &format!("{prefix}.self_attn.o_proj.weight"),
            &[HIDDEN_WIDTH, QUERY_HEAD_COUNT * HEAD_WIDTH],
        )?,
        query_norm: load_bf16_tensor(
            model_shards,
            &format!("{prefix}.self_attn.q_norm.weight"),
            &[HEAD_WIDTH],
        )?,
        key_norm: load_bf16_tensor(
            model_shards,
            &format!("{prefix}.self_attn.k_norm.weight"),
            &[HEAD_WIDTH],
        )?,
        post_attention_norm: load_bf16_tensor(
            model_shards,
            &format!("{prefix}.post_attention_layernorm.weight"),
            &[HIDDEN_WIDTH],
        )?,
        gate: load_bf16_tensor(
            model_shards,
            &format!("{prefix}.mlp.gate_proj.weight"),
            &[INTERMEDIATE_WIDTH, HIDDEN_WIDTH],
        )?,
        up: load_bf16_tensor(
            model_shards,
            &format!("{prefix}.mlp.up_proj.weight"),
            &[INTERMEDIATE_WIDTH, HIDDEN_WIDTH],
        )?,
        down: load_bf16_tensor(
            model_shards,
            &format!("{prefix}.mlp.down_proj.weight"),
            &[HIDDEN_WIDTH, INTERMEDIATE_WIDTH],
        )?,
    })
}

fn load_bf16_tensor(
    model_shards: &[MlxSafetensors],
    tensor_name: &str,
    expected_shape: &[i32],
) -> Result<MlxArray, Flux2KleinTextConditioningError> {
    let tensor = model_shards
        .iter()
        .find_map(|model_shard| model_shard.tensor(tensor_name).ok())
        .ok_or_else(|| Flux2KleinTextConditioningError::MissingTensor {
            tensor_name: tensor_name.to_owned(),
        })?;
    if tensor.dtype() != MlxDtype::BFloat16 {
        return Err(Flux2KleinTextConditioningError::InvalidTensor {
            tensor_name: tensor_name.to_owned(),
            description: "storage dtype must be BF16",
        });
    }
    if tensor.shape() != expected_shape {
        return Err(Flux2KleinTextConditioningError::InvalidTensor {
            tensor_name: tensor_name.to_owned(),
            description: "shape disagrees with the official Qwen3-4B profile",
        });
    }
    Ok(tensor)
}
