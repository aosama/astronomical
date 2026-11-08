//! Tensor binding with complete or descriptor-streamed layer ownership.
//! Tensors come through the tensor store: projections and the embedding stay
//! packed in the 4-bit build (about 28% of the BF16 memory once the group
//! scales and offsets are counted), norm scales are
//! dense BF16 in both builds.

use std::collections::VecDeque;

use astronomical_runtime_integration::MlxRuntime;

use crate::flux2_klein::tensor_store::{
    Flux2KleinTensorStore, Flux2KleinWeight, map_component_weight_files,
};
use crate::flux2_klein::weight_layout::Flux2KleinComponentWeightFiles;
use crate::{PerformanceAttribution, PerformanceOperation};

use super::super::Flux2KleinResidencyMode;
use super::error::Flux2KleinTextConditioningError;
use mlx_c_rust::{MlxArray, MlxDtype};

pub(super) const HIDDEN_WIDTH: i32 = 2_560;
pub(super) const INTERMEDIATE_WIDTH: i32 = 9_728;
pub(super) const QUERY_HEAD_COUNT: i32 = 32;
pub(super) const KEY_VALUE_HEAD_COUNT: i32 = 8;
pub(super) const HEAD_WIDTH: i32 = 128;
pub(super) const EXECUTED_LAYER_COUNT: usize = 27;
const VOCABULARY_SIZE: i32 = 151_936;
/// A packed projection is backed by three arrays (weight, scales, offsets).
const PACKED_PROJECTION_ARRAY_COUNT: usize = 3;
/// Four norm scales plus seven projections, each packed in the 4-bit build.
const LAYER_ARRAY_CAPACITY: usize = 4 + 7 * PACKED_PROJECTION_ARRAY_COUNT;

#[derive(Debug)]
pub(super) struct Flux2KleinTextWeights {
    pub(super) embedding: Option<Flux2KleinWeight>,
    complete_layers: VecDeque<Flux2KleinDecoderLayerWeights>,
    text_encoder_weight_files: Option<Flux2KleinComponentWeightFiles>,
    streamed_layer_is_retained: bool,
    peak_streamed_layer_count: usize,
}

#[derive(Debug)]
pub(super) struct Flux2KleinDecoderLayerWeights {
    pub(super) input_norm: MlxArray,
    pub(super) query: Flux2KleinWeight,
    pub(super) key: Flux2KleinWeight,
    pub(super) value: Flux2KleinWeight,
    pub(super) output: Flux2KleinWeight,
    pub(super) query_norm: MlxArray,
    pub(super) key_norm: MlxArray,
    pub(super) post_attention_norm: MlxArray,
    pub(super) gate: Flux2KleinWeight,
    pub(super) up: Flux2KleinWeight,
    pub(super) down: Flux2KleinWeight,
}

impl Flux2KleinTextWeights {
    pub(super) fn load(
        runtime: &MlxRuntime,
        text_encoder_weight_files: Flux2KleinComponentWeightFiles,
        residency_mode: Flux2KleinResidencyMode,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<Self, Flux2KleinTextConditioningError> {
        match residency_mode {
            Flux2KleinResidencyMode::Complete => {
                let text_encoder_store =
                    map_text_encoder(&text_encoder_weight_files, runtime, performance_attribution)?;
                let weights = performance_attribution.measure_operation(
                    PerformanceOperation::ImageTextComponentLoading,
                    |_performance_attribution| bind_complete_weights(runtime, &text_encoder_store),
                )?;
                performance_attribution.measure_operation(
                    PerformanceOperation::ImageTextComponentLoading,
                    |_performance_attribution| weights.materialize_complete(runtime),
                )?;
                drop(text_encoder_store);
                Ok(weights)
            }
            Flux2KleinResidencyMode::Streamed => {
                let text_encoder_store =
                    map_text_encoder(&text_encoder_weight_files, runtime, performance_attribution)?;
                let embedding = performance_attribution.measure_operation(
                    PerformanceOperation::ImageTextComponentLoading,
                    |_performance_attribution| bind_embedding(&text_encoder_store),
                )?;
                performance_attribution.measure_operation(
                    PerformanceOperation::ImageTextComponentLoading,
                    |_performance_attribution| runtime.evaluate_arrays(&embedding.arrays()),
                )?;
                drop(text_encoder_store);
                Ok(Self {
                    embedding: Some(embedding),
                    complete_layers: VecDeque::new(),
                    text_encoder_weight_files: Some(text_encoder_weight_files),
                    streamed_layer_is_retained: false,
                    peak_streamed_layer_count: 0,
                })
            }
        }
    }

    pub(super) const fn is_streamed(&self) -> bool {
        self.text_encoder_weight_files.is_some()
    }

    pub(super) fn take_layer(
        &mut self,
        runtime: &MlxRuntime,
        layer_index: usize,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<Flux2KleinDecoderLayerWeights, Flux2KleinTextConditioningError> {
        if let Some(text_encoder_weight_files) = self.text_encoder_weight_files.as_ref() {
            if self.streamed_layer_is_retained {
                return Err(Flux2KleinTextConditioningError::WeightsUnavailable);
            }
            let layer_store =
                map_text_encoder(text_encoder_weight_files, runtime, performance_attribution)?;
            let layer_weights = performance_attribution.measure_operation(
                PerformanceOperation::ImageTextComponentLoading,
                |_performance_attribution| bind_decoder_layer(runtime, &layer_store, layer_index),
            )?;
            performance_attribution.measure_operation(
                PerformanceOperation::ImageTextComponentLoading,
                |_performance_attribution| layer_weights.materialize(runtime),
            )?;
            // The map retains every lazy array handle. Dropping it after the selected
            // page is durable prevents prior layers from accumulating in map ownership.
            drop(layer_store);
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
        self.text_encoder_weight_files = None;
        Ok(())
    }

    fn materialize_complete(
        &self,
        runtime: &MlxRuntime,
    ) -> Result<(), Flux2KleinTextConditioningError> {
        let mut arrays = Vec::with_capacity(
            PACKED_PROJECTION_ARRAY_COUNT + self.complete_layers.len() * LAYER_ARRAY_CAPACITY,
        );
        if let Some(embedding) = &self.embedding {
            arrays.extend(embedding.arrays());
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
        let mut arrays = Vec::with_capacity(LAYER_ARRAY_CAPACITY);
        self.append_arrays(&mut arrays);
        runtime.evaluate_arrays(&arrays)?;
        Ok(())
    }

    fn append_arrays<'weights>(&'weights self, arrays: &mut Vec<&'weights MlxArray>) {
        arrays.extend([
            &self.input_norm,
            &self.query_norm,
            &self.key_norm,
            &self.post_attention_norm,
        ]);
        arrays.extend(self.query.arrays());
        arrays.extend(self.key.arrays());
        arrays.extend(self.value.arrays());
        arrays.extend(self.output.arrays());
        arrays.extend(self.gate.arrays());
        arrays.extend(self.up.arrays());
        arrays.extend(self.down.arrays());
    }
}

fn bind_complete_weights(
    runtime: &MlxRuntime,
    text_encoder_store: &Flux2KleinTensorStore,
) -> Result<Flux2KleinTextWeights, Flux2KleinTextConditioningError> {
    let embedding = bind_embedding(text_encoder_store)?;
    let mut complete_layers = VecDeque::with_capacity(EXECUTED_LAYER_COUNT);
    for layer_index in 0..EXECUTED_LAYER_COUNT {
        complete_layers.push_back(bind_decoder_layer(
            runtime,
            text_encoder_store,
            layer_index,
        )?);
    }
    Ok(Flux2KleinTextWeights {
        embedding: Some(embedding),
        complete_layers,
        text_encoder_weight_files: None,
        streamed_layer_is_retained: false,
        peak_streamed_layer_count: 0,
    })
}

/// Maps the text encoder's shards, attributed to the text mapping operation.
fn map_text_encoder(
    text_encoder_weight_files: &Flux2KleinComponentWeightFiles,
    runtime: &MlxRuntime,
    performance_attribution: &mut PerformanceAttribution,
) -> Result<Flux2KleinTensorStore, Flux2KleinTextConditioningError> {
    performance_attribution.measure_operation(
        PerformanceOperation::ImageTextComponentMapping,
        |performance_attribution| {
            let positional_file_read_metrics =
                performance_attribution.positional_file_read_metrics();
            Ok(map_component_weight_files(
                text_encoder_weight_files,
                runtime,
                positional_file_read_metrics.as_ref(),
            )?)
        },
    )
}

fn bind_embedding(
    text_encoder_store: &Flux2KleinTensorStore,
) -> Result<Flux2KleinWeight, Flux2KleinTextConditioningError> {
    load_projection(
        text_encoder_store,
        "model.embed_tokens.weight",
        &[VOCABULARY_SIZE, HIDDEN_WIDTH],
    )
}

fn bind_decoder_layer(
    runtime: &MlxRuntime,
    text_encoder_store: &Flux2KleinTensorStore,
    layer_index: usize,
) -> Result<Flux2KleinDecoderLayerWeights, Flux2KleinTextConditioningError> {
    let prefix = format!("model.layers.{layer_index}");
    Ok(Flux2KleinDecoderLayerWeights {
        input_norm: load_bf16_tensor(
            runtime,
            text_encoder_store,
            &format!("{prefix}.input_layernorm.weight"),
            &[HIDDEN_WIDTH],
        )?,
        query: load_projection(
            text_encoder_store,
            &format!("{prefix}.self_attn.q_proj.weight"),
            &[QUERY_HEAD_COUNT * HEAD_WIDTH, HIDDEN_WIDTH],
        )?,
        key: load_projection(
            text_encoder_store,
            &format!("{prefix}.self_attn.k_proj.weight"),
            &[KEY_VALUE_HEAD_COUNT * HEAD_WIDTH, HIDDEN_WIDTH],
        )?,
        value: load_projection(
            text_encoder_store,
            &format!("{prefix}.self_attn.v_proj.weight"),
            &[KEY_VALUE_HEAD_COUNT * HEAD_WIDTH, HIDDEN_WIDTH],
        )?,
        output: load_projection(
            text_encoder_store,
            &format!("{prefix}.self_attn.o_proj.weight"),
            &[HIDDEN_WIDTH, QUERY_HEAD_COUNT * HEAD_WIDTH],
        )?,
        query_norm: load_bf16_tensor(
            runtime,
            text_encoder_store,
            &format!("{prefix}.self_attn.q_norm.weight"),
            &[HEAD_WIDTH],
        )?,
        key_norm: load_bf16_tensor(
            runtime,
            text_encoder_store,
            &format!("{prefix}.self_attn.k_norm.weight"),
            &[HEAD_WIDTH],
        )?,
        post_attention_norm: load_bf16_tensor(
            runtime,
            text_encoder_store,
            &format!("{prefix}.post_attention_layernorm.weight"),
            &[HIDDEN_WIDTH],
        )?,
        gate: load_projection(
            text_encoder_store,
            &format!("{prefix}.mlp.gate_proj.weight"),
            &[INTERMEDIATE_WIDTH, HIDDEN_WIDTH],
        )?,
        up: load_projection(
            text_encoder_store,
            &format!("{prefix}.mlp.up_proj.weight"),
            &[INTERMEDIATE_WIDTH, HIDDEN_WIDTH],
        )?,
        down: load_projection(
            text_encoder_store,
            &format!("{prefix}.mlp.down_proj.weight"),
            &[HIDDEN_WIDTH, INTERMEDIATE_WIDTH],
        )?,
    })
}

/// A projection (or the embedding) as stored: packed in the 4-bit build, whose
/// packed shapes the store checks against `expected_shape`; dense BF16 in the
/// BF16 build, checked here like every other dense tensor.
fn load_projection(
    text_encoder_store: &Flux2KleinTensorStore,
    tensor_name: &str,
    expected_shape: &[i32],
) -> Result<Flux2KleinWeight, Flux2KleinTextConditioningError> {
    let mut logical_shape = Vec::with_capacity(expected_shape.len());
    for dimension in expected_shape {
        logical_shape.push(usize::try_from(*dimension).map_err(|_| {
            Flux2KleinTextConditioningError::InvalidTensor {
                tensor_name: tensor_name.to_owned(),
                description: "expected shape has a negative dimension",
            }
        })?);
    }
    let projection = text_encoder_store.weight(tensor_name, &logical_shape)?;
    if let Some(dense_projection) = projection.dense() {
        validate_bf16_tensor(tensor_name, dense_projection, expected_shape)?;
    }
    Ok(projection)
}

fn load_bf16_tensor(
    runtime: &MlxRuntime,
    text_encoder_store: &Flux2KleinTensorStore,
    tensor_name: &str,
    expected_shape: &[i32],
) -> Result<MlxArray, Flux2KleinTextConditioningError> {
    let tensor = text_encoder_store.dense(runtime, tensor_name)?;
    validate_bf16_tensor(tensor_name, &tensor, expected_shape)?;
    Ok(tensor)
}

fn validate_bf16_tensor(
    tensor_name: &str,
    tensor: &MlxArray,
    expected_shape: &[i32],
) -> Result<(), Flux2KleinTextConditioningError> {
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
    Ok(())
}
