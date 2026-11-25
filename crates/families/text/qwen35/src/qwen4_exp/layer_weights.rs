//! The weights Flash-Next adds to a Qwen 3.5 decoder layer: the two
//! hyper-connections that replace its norms, the sparse-attention indexer of
//! a full-attention layer, and the n-gram layer's projections.
//!
//! The n-gram table itself is not here: it stays on the SSD and is read row
//! by row (`ngram_table`), so only its small projections are bound.

use std::collections::HashMap;

use mlx_bridge::MlxArray;
use text_generation::execution_error::TextExecutionError;

use crate::qwen3_5_core::configuration::{Qwen3_5Config, Qwen4ExpArchitecture};
use crate::qwen3_5_core::model_math::dense_projection::Qwen3_5DenseProjection;
use crate::qwen3_5_core::model_math::weights::{take_dense_projection, take_tensor};

use super::hyper_connection::HyperConnectionWeights;

/// Everything Flash-Next adds to one decoder layer.
#[derive(Debug)]
pub struct FlashNextLayerWeights {
    pub lane_connections: LayerLaneConnections,
    /// On full-attention layers.
    pub sparse_attention_indexer: Option<SparseAttentionIndexerWeights>,
    /// On the one n-gram layer.
    pub ngram_layer: Option<NgramLayerWeights>,
}

impl FlashNextLayerWeights {
    pub fn take(
        bound_tensors: &mut HashMap<String, MlxArray>,
        config: &Qwen3_5Config,
        architecture: &Qwen4ExpArchitecture,
        layer_index: usize,
    ) -> Result<Self, TextExecutionError> {
        let layer_prefix = format!("language_model.model.layers.{layer_index}");
        let sparse_attention_indexer = if config.decoder_layer_is_full_attention(layer_index) {
            Some(SparseAttentionIndexerWeights::take(
                bound_tensors,
                config,
                &layer_prefix,
            )?)
        } else {
            None
        };
        let ngram_layer = if architecture.ngram_layer_index as usize == layer_index {
            Some(NgramLayerWeights::take(
                bound_tensors,
                config,
                &layer_prefix,
            )?)
        } else {
            None
        };
        Ok(Self {
            lane_connections: LayerLaneConnections::take(bound_tensors, config, &layer_prefix)?,
            sparse_attention_indexer,
            ngram_layer,
        })
    }

    pub fn append_array_references<'weights>(&'weights self, arrays: &mut Vec<&'weights MlxArray>) {
        self.lane_connections.append_array_references(arrays);
        if let Some(indexer) = &self.sparse_attention_indexer {
            indexer.append_array_references(arrays);
        }
        if let Some(ngram_layer) = &self.ngram_layer {
            ngram_layer.append_array_references(arrays);
        }
    }
}

/// The hyper-connections around a layer's attention and expert blocks.
#[derive(Debug)]
pub struct LayerLaneConnections {
    pub attention: HyperConnectionWeights,
    pub feed_forward: HyperConnectionWeights,
}

impl LayerLaneConnections {
    pub fn take(
        bound_tensors: &mut HashMap<String, MlxArray>,
        config: &Qwen3_5Config,
        layer_prefix: &str,
    ) -> Result<Self, TextExecutionError> {
        Ok(Self {
            attention: HyperConnectionWeights::take(
                bound_tensors,
                config,
                &format!("{layer_prefix}.attn_hyper_connection"),
                true,
            )?,
            feed_forward: HyperConnectionWeights::take(
                bound_tensors,
                config,
                &format!("{layer_prefix}.mlp_hyper_connection"),
                true,
            )?,
        })
    }

    pub fn append_array_references<'weights>(&'weights self, arrays: &mut Vec<&'weights MlxArray>) {
        self.attention.append_array_references(arrays);
        self.feed_forward.append_array_references(arrays);
    }
}

/// A full-attention layer's sparse-attention indexer.
#[derive(Debug)]
pub struct SparseAttentionIndexerWeights {
    /// Projects the query heads and the one key head together.
    pub query_key_projection: Qwen3_5DenseProjection,
    /// `1 + deviation`, float32.
    pub query_norm_scale: MlxArray,
    pub key_norm_scale: MlxArray,
}

impl SparseAttentionIndexerWeights {
    pub fn take(
        bound_tensors: &mut HashMap<String, MlxArray>,
        config: &Qwen3_5Config,
        layer_prefix: &str,
    ) -> Result<Self, TextExecutionError> {
        let indexer_prefix = format!("{layer_prefix}.self_attn.indexer");
        Ok(Self {
            query_key_projection: take_dense_projection(
                bound_tensors,
                config,
                &format!("{indexer_prefix}.index_qk_proj"),
            )?,
            query_norm_scale: take_tensor(
                bound_tensors,
                format!("{indexer_prefix}.q_layernorm.weight"),
            )?,
            key_norm_scale: take_tensor(
                bound_tensors,
                format!("{indexer_prefix}.k_layernorm.weight"),
            )?,
        })
    }

    pub fn append_array_references<'weights>(&'weights self, arrays: &mut Vec<&'weights MlxArray>) {
        self.query_key_projection.append_array_references(arrays);
        arrays.push(&self.query_norm_scale);
        arrays.push(&self.key_norm_scale);
    }
}

/// The n-gram layer's projections, norms, convolution and hash constants.
#[derive(Debug)]
pub struct NgramLayerWeights {
    pub key_projection: Qwen3_5DenseProjection,
    pub value_projection: Qwen3_5DenseProjection,
    /// `1 + deviation`, float32, one per lane feature.
    pub key_norm_scale: MlxArray,
    pub query_norm_scale: MlxArray,
    pub convolution_norm_scale: MlxArray,
    /// Depthwise kernel, `[lane features, kernel, 1]`.
    pub convolution_weight: MlxArray,
    pub hash_multipliers: MlxArray,
    pub head_vocabulary_sizes: MlxArray,
    pub head_row_offsets: MlxArray,
}

impl NgramLayerWeights {
    pub fn take(
        bound_tensors: &mut HashMap<String, MlxArray>,
        config: &Qwen3_5Config,
        layer_prefix: &str,
    ) -> Result<Self, TextExecutionError> {
        let ngram_prefix = format!("{layer_prefix}.ple");
        Ok(Self {
            key_projection: take_dense_projection(
                bound_tensors,
                config,
                &ngram_tensor_name(&ngram_prefix, "key_proj"),
            )?,
            value_projection: take_dense_projection(
                bound_tensors,
                config,
                &ngram_tensor_name(&ngram_prefix, "value_proj"),
            )?,
            key_norm_scale: take_tensor(
                bound_tensors,
                ngram_tensor_name(&ngram_prefix, "norm_key.weight"),
            )?,
            query_norm_scale: take_tensor(
                bound_tensors,
                ngram_tensor_name(&ngram_prefix, "norm_query.weight"),
            )?,
            convolution_norm_scale: take_tensor(
                bound_tensors,
                ngram_tensor_name(&ngram_prefix, "norm_conv.weight"),
            )?,
            convolution_weight: take_tensor(
                bound_tensors,
                ngram_tensor_name(&ngram_prefix, "conv1d.weight"),
            )?,
            hash_multipliers: take_tensor(
                bound_tensors,
                ngram_tensor_name(&ngram_prefix, "ple_embedding.layer_multipliers"),
            )?,
            head_vocabulary_sizes: take_tensor(
                bound_tensors,
                ngram_tensor_name(&ngram_prefix, "ple_embedding.ngram_heads_vocab_sizes"),
            )?,
            head_row_offsets: take_tensor(
                bound_tensors,
                ngram_tensor_name(&ngram_prefix, "ple_embedding.ngram_heads_offsets"),
            )?,
        })
    }

    pub fn append_array_references<'weights>(&'weights self, arrays: &mut Vec<&'weights MlxArray>) {
        self.key_projection.append_array_references(arrays);
        self.value_projection.append_array_references(arrays);
        arrays.extend([
            &self.key_norm_scale,
            &self.query_norm_scale,
            &self.convolution_norm_scale,
            &self.convolution_weight,
            &self.hash_multipliers,
            &self.head_vocabulary_sizes,
            &self.head_row_offsets,
        ]);
    }
}

fn ngram_tensor_name(ngram_prefix: &str, name: &str) -> String {
    format!("{ngram_prefix}.{name}")
}
