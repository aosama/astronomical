use astronomical_runtime_integration::MlxRuntimeError;

use super::feed_forward_weights::Qwen3_5MoEFeedForwardWeights;
use crate::qwen3_5_core::dense::mlp::Qwen3_5DenseMlpWeights;
use mlx_c_rust::MlxArray;

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
    /// A jangtq2 codebook expert page kept in its packed on-disk form:
    /// `[experts, output_rows, words]` uint32 codes plus `[experts, output_rows]`
    /// float16 row scales. Routed experts only. The page is never decoded for
    /// retention — decoded bfloat16 is eight times larger, which is what kept
    /// the 122B bundle's expert set from fitting in RAM. Decode tokens
    /// multiply straight from the codes through the fused gathered projection
    /// kernel; multi-token prefill decodes a transient bfloat16 copy at
    /// forward time and gathers through `gather_mm`.
    Jangtq2Codebook {
        packed_code_words: MlxArray,
        row_scales: MlxArray,
        codebook_bit_width: u32,
    },
}

impl Qwen3_5AffineWeights {
    pub(crate) fn retained_reference(&self) -> Result<Self, MlxRuntimeError> {
        match self {
            Self::NativeBfloat16 { weight } => Ok(Self::NativeBfloat16 {
                weight: weight.retain()?,
            }),
            Self::Quantized {
                packed_weight,
                quantization_scales,
                quantization_biases,
                quantization_bits,
                quantization_group_size,
            } => Ok(Self::Quantized {
                packed_weight: packed_weight.retain()?,
                quantization_scales: quantization_scales.retain()?,
                quantization_biases: quantization_biases.retain()?,
                quantization_bits: *quantization_bits,
                quantization_group_size: *quantization_group_size,
            }),
            Self::Jangtq2Codebook {
                packed_code_words,
                row_scales,
                codebook_bit_width,
            } => Ok(Self::Jangtq2Codebook {
                packed_code_words: packed_code_words.retain()?,
                row_scales: row_scales.retain()?,
                codebook_bit_width: *codebook_bit_width,
            }),
        }
    }

    /// The size of the stacked expert axis (axis 0) of this projection.
    pub(crate) fn expert_count(&self) -> usize {
        let leading_dimension = match self {
            Self::NativeBfloat16 { weight } => weight.shape().first().copied(),
            Self::Quantized { packed_weight, .. } => packed_weight.shape().first().copied(),
            Self::Jangtq2Codebook {
                packed_code_words, ..
            } => packed_code_words.shape().first().copied(),
        };
        leading_dimension
            .and_then(|dimension| usize::try_from(dimension).ok())
            .unwrap_or(0)
    }

    /// Gathers the experts named by `expert_axis_indices` (axis 0 of every
    /// stacked array) into a new, smaller stacked owner, keeping the
    /// representation and its metadata. The gather is lazy; the subset
    /// materializes with whatever consumes it.
    pub(crate) fn select_experts(
        &self,
        runtime: &astronomical_runtime_integration::MlxRuntime,
        expert_axis_indices: &MlxArray,
    ) -> Result<Self, MlxRuntimeError> {
        Ok(match self {
            Self::NativeBfloat16 { weight } => Self::NativeBfloat16 {
                weight: runtime.take_axis(weight, expert_axis_indices, 0)?,
            },
            Self::Quantized {
                packed_weight,
                quantization_scales,
                quantization_biases,
                quantization_bits,
                quantization_group_size,
            } => Self::Quantized {
                packed_weight: runtime.take_axis(packed_weight, expert_axis_indices, 0)?,
                quantization_scales: runtime.take_axis(
                    quantization_scales,
                    expert_axis_indices,
                    0,
                )?,
                quantization_biases: runtime.take_axis(
                    quantization_biases,
                    expert_axis_indices,
                    0,
                )?,
                quantization_bits: *quantization_bits,
                quantization_group_size: *quantization_group_size,
            },
            Self::Jangtq2Codebook {
                packed_code_words,
                row_scales,
                codebook_bit_width,
            } => Self::Jangtq2Codebook {
                packed_code_words: runtime.take_axis(packed_code_words, expert_axis_indices, 0)?,
                row_scales: runtime.take_axis(row_scales, expert_axis_indices, 0)?,
                codebook_bit_width: *codebook_bit_width,
            },
        })
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
            Self::Jangtq2Codebook {
                packed_code_words,
                row_scales,
                ..
            } => {
                arrays.push(packed_code_words);
                arrays.push(row_scales);
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
    pub(crate) fn append_array_references<'weights>(
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
    pub(crate) fn append_array_references<'weights>(
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
    pub(crate) fn append_array_references<'weights>(
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
    pub(crate) fn append_array_references<'weights>(
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
    pub(crate) fn append_array_references<'weights>(
        &'weights self,
        arrays: &mut Vec<&'weights MlxArray>,
    ) {
        arrays.push(&self.input_normalization_weight);
        self.attention_weights.append_array_references(arrays);
        arrays.push(&self.post_attention_normalization_weight);
        self.mlp_weights.append_array_references(arrays);
    }
}
