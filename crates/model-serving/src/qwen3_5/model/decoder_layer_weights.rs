use astronomical_runtime_integration::MlxArray;

use crate::qwen3_5::dense::mlp::Qwen3_5DenseMlpWeights;
use crate::qwen3_5_moe::model::feed_forward_weights::Qwen3_5MoEFeedForwardWeights;

/// One pre-bound affine module used directly by Qwen3.5 execution.
///
/// Each quantized module carries its own bit width and group size, enabling
/// mixed-precision models where different layers use different quantization
/// parameters (e.g., 8-bit/64-group attention, 6-bit/128-group output projections).
///
/// Also used by expert paging for prefill and decode expert weights, where the
/// weight arrays have shape `[selected_count, ...]` instead of `[256, ...]`.
#[derive(Debug)]
pub(crate) enum Qwen3_5AffineWeights {
    NativeBfloat16 {
        weight: MlxArray,
    },
    Quantized {
        packed_weight: MlxArray,
        quantization_scales: MlxArray,
        quantization_biases: MlxArray,
        quantization_bits: i32,
        quantization_group_size: i32,
    },
}

impl Qwen3_5AffineWeights {
    pub(crate) fn payload_byte_count(&self) -> u64 {
        match self {
            Self::NativeBfloat16 { weight } => weight.byte_count() as u64,
            Self::Quantized {
                packed_weight,
                quantization_scales,
                quantization_biases,
                ..
            } => packed_weight
                .byte_count()
                .saturating_add(quantization_scales.byte_count())
                .saturating_add(quantization_biases.byte_count()) as u64,
        }
    }

    pub(crate) fn append_array_references<'weights>(
        &'weights self,
        arrays: &mut Vec<&'weights MlxArray>,
    ) {
        match self {
            Self::NativeBfloat16 { weight } => arrays.push(weight),
            Self::Quantized {
                packed_weight,
                quantization_scales,
                quantization_biases,
                ..
            } => {
                arrays.push(packed_weight);
                arrays.push(quantization_scales);
                arrays.push(quantization_biases);
            }
        }
    }
}

/// Pre-bound weights for one Qwen3.5 GatedDeltaNet attention layer.
#[derive(Debug)]
pub(crate) struct Qwen3_5LinearAttentionWeights {
    pub(crate) input_queries_keys_values_projection: Qwen3_5AffineWeights,
    pub(crate) output_gate_projection: Qwen3_5AffineWeights,
    pub(crate) update_rate_projection: Qwen3_5AffineWeights,
    pub(crate) decay_interval_projection: Qwen3_5AffineWeights,
    pub(crate) convolution_weight: MlxArray,
    pub(crate) decay_interval_bias: MlxArray,
    pub(crate) decay_rate_logarithm: MlxArray,
    pub(crate) normalization_weight: MlxArray,
    pub(crate) output_projection: Qwen3_5AffineWeights,
}

impl Qwen3_5LinearAttentionWeights {
    pub(super) fn append_array_references<'weights>(
        &'weights self,
        arrays: &mut Vec<&'weights MlxArray>,
    ) {
        arrays.push(&self.convolution_weight);
        self.input_queries_keys_values_projection
            .append_array_references(arrays);
        self.output_gate_projection.append_array_references(arrays);
        self.update_rate_projection.append_array_references(arrays);
        self.decay_interval_projection
            .append_array_references(arrays);
        self.output_projection.append_array_references(arrays);
        arrays.push(&self.decay_interval_bias);
        arrays.push(&self.decay_rate_logarithm);
        arrays.push(&self.normalization_weight);
    }
}

/// Pre-bound weights for one Qwen3.5 full-attention layer.
#[derive(Debug)]
pub(crate) struct Qwen3_5FullAttentionWeights {
    pub(crate) query_projection: Qwen3_5AffineWeights,
    pub(crate) key_projection: Qwen3_5AffineWeights,
    pub(crate) value_projection: Qwen3_5AffineWeights,
    pub(crate) output_projection: Qwen3_5AffineWeights,
    pub(crate) query_normalization_weight: MlxArray,
    pub(crate) key_normalization_weight: MlxArray,
}

impl Qwen3_5FullAttentionWeights {
    pub(super) fn append_array_references<'weights>(
        &'weights self,
        arrays: &mut Vec<&'weights MlxArray>,
    ) {
        self.query_projection.append_array_references(arrays);
        self.key_projection.append_array_references(arrays);
        self.value_projection.append_array_references(arrays);
        self.output_projection.append_array_references(arrays);
        arrays.push(&self.query_normalization_weight);
        arrays.push(&self.key_normalization_weight);
    }
}

/// The pre-bound attention family for one decoder layer.
#[derive(Debug)]
pub(crate) enum Qwen3_5AttentionWeights {
    Linear(Qwen3_5LinearAttentionWeights),
    Full(Qwen3_5FullAttentionWeights),
}

impl Qwen3_5AttentionWeights {
    pub(super) fn append_array_references<'weights>(
        &'weights self,
        arrays: &mut Vec<&'weights MlxArray>,
    ) {
        match self {
            Self::Linear(linear_attention_weights) => {
                linear_attention_weights.append_array_references(arrays);
            }
            Self::Full(full_attention_weights) => {
                full_attention_weights.append_array_references(arrays);
            }
        }
    }
}

/// Mutually exclusive feed-forward weights for one Qwen3.5 decoder layer.
#[derive(Debug)]
pub(crate) enum Qwen3_5DecoderFeedForwardWeights {
    Dense(Qwen3_5DenseMlpWeights),
    MixtureOfExperts(Qwen3_5MoEFeedForwardWeights),
}

impl Qwen3_5DecoderFeedForwardWeights {
    pub(super) fn append_array_references<'weights>(
        &'weights self,
        arrays: &mut Vec<&'weights MlxArray>,
    ) {
        match self {
            Self::Dense(dense_mlp_weights) => dense_mlp_weights.append_array_references(arrays),
            Self::MixtureOfExperts(mixture_of_experts_weights) => {
                mixture_of_experts_weights.append_array_references(arrays);
            }
        }
    }
}

/// All pre-bound weights for one Qwen3.5 decoder layer.
#[derive(Debug)]
pub(crate) struct Qwen3_5DecoderLayerWeights {
    pub(crate) input_normalization_weight: MlxArray,
    pub(crate) attention_weights: Qwen3_5AttentionWeights,
    pub(crate) post_attention_normalization_weight: MlxArray,
    pub(crate) mlp_weights: Qwen3_5DecoderFeedForwardWeights,
}

impl Qwen3_5DecoderLayerWeights {
    pub(super) fn append_array_references<'weights>(
        &'weights self,
        arrays: &mut Vec<&'weights MlxArray>,
    ) {
        arrays.push(&self.input_normalization_weight);
        self.attention_weights.append_array_references(arrays);
        arrays.push(&self.post_attention_normalization_weight);
        self.mlp_weights.append_array_references(arrays);
    }
}
