use astronomical_runtime_integration::MlxArray;

/// One pre-bound affine module used directly by Qwen3.5-MoE execution.
///
/// Each quantized module carries its own bit width and group size, enabling
/// mixed-precision models where different layers use different quantization
/// parameters (e.g., 8-bit/64-group attention, 6-bit/128-group output projections).
///
/// Also used by expert paging for prefill and decode expert weights, where the
/// weight arrays have shape `[selected_count, ...]` instead of `[256, ...]`.
#[derive(Debug)]
pub(crate) enum Qwen3_5MoEAffineWeights {
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

impl Qwen3_5MoEAffineWeights {
    pub(in crate::qwen3_5_moe) fn append_array_references<'weights>(
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

/// Pre-bound weights for one Qwen3.5-MoE GatedDeltaNet attention layer.
#[derive(Debug)]
pub(super) struct Qwen3_5MoELinearAttentionWeights {
    pub(super) input_queries_keys_values_projection: Qwen3_5MoEAffineWeights,
    pub(super) output_gate_projection: Qwen3_5MoEAffineWeights,
    pub(super) update_rate_projection: Qwen3_5MoEAffineWeights,
    pub(super) decay_interval_projection: Qwen3_5MoEAffineWeights,
    pub(super) convolution_weight: MlxArray,
    pub(super) decay_interval_bias: MlxArray,
    pub(super) decay_rate_logarithm: MlxArray,
    pub(super) normalization_weight: MlxArray,
    pub(super) output_projection: Qwen3_5MoEAffineWeights,
}

impl Qwen3_5MoELinearAttentionWeights {
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

/// Pre-bound weights for one Qwen3.5-MoE full-attention layer.
#[derive(Debug)]
pub(super) struct Qwen3_5MoEFullAttentionWeights {
    pub(super) query_projection: Qwen3_5MoEAffineWeights,
    pub(super) key_projection: Qwen3_5MoEAffineWeights,
    pub(super) value_projection: Qwen3_5MoEAffineWeights,
    pub(super) output_projection: Qwen3_5MoEAffineWeights,
    pub(super) query_normalization_weight: MlxArray,
    pub(super) key_normalization_weight: MlxArray,
}

impl Qwen3_5MoEFullAttentionWeights {
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
pub(super) enum Qwen3_5MoEAttentionWeights {
    Linear(Qwen3_5MoELinearAttentionWeights),
    Full(Qwen3_5MoEFullAttentionWeights),
}

impl Qwen3_5MoEAttentionWeights {
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

/// Router gate weights that can be either quantized (affine) or unquantized (bfloat16).
///
/// In oQ4 models, the MoE router gate is quantized alongside other modules.
/// In oQ6e models, the gate is stored as plain bfloat16 without scales/biases.
/// This enum enables both patterns without runtime ambiguity.
#[derive(Debug)]
pub(super) enum RouterGateWeights {
    Affine(Qwen3_5MoEAffineWeights),
    Unquantized(MlxArray),
}

/// Pre-bound router and shared-expert weights for one decoder layer.
#[derive(Debug)]
pub(super) struct Qwen3_5MoEMixtureOfExpertsWeights {
    pub(super) router_projection: RouterGateWeights,
    pub(super) shared_expert_gate_projection: Qwen3_5MoEAffineWeights,
    pub(super) shared_expert_up_projection: Qwen3_5MoEAffineWeights,
    pub(super) shared_expert_down_projection: Qwen3_5MoEAffineWeights,
    pub(super) shared_expert_output_gate_projection: Qwen3_5MoEAffineWeights,
}

/// Pre-bound projections for one dense Qwen3.5 SwiGLU MLP layer.
#[derive(Debug)]
pub(super) struct Qwen3_5DenseMlpWeights {
    pub(super) gate_projection: Qwen3_5MoEAffineWeights,
    pub(super) up_projection: Qwen3_5MoEAffineWeights,
    pub(super) down_projection: Qwen3_5MoEAffineWeights,
}

impl Qwen3_5DenseMlpWeights {
    pub(super) fn append_array_references<'weights>(
        &'weights self,
        arrays: &mut Vec<&'weights MlxArray>,
    ) {
        self.gate_projection.append_array_references(arrays);
        self.up_projection.append_array_references(arrays);
        self.down_projection.append_array_references(arrays);
    }
}

/// Mutually exclusive feed-forward weights for one Qwen3.5 decoder layer.
#[derive(Debug)]
pub(super) enum Qwen3_5DecoderLayerMlpWeights {
    Dense(Qwen3_5DenseMlpWeights),
    Sparse(Qwen3_5MoEMixtureOfExpertsWeights),
}

impl Qwen3_5DecoderLayerMlpWeights {
    pub(super) fn append_array_references<'weights>(
        &'weights self,
        arrays: &mut Vec<&'weights MlxArray>,
    ) {
        match self {
            Self::Dense(dense_mlp_weights) => dense_mlp_weights.append_array_references(arrays),
            Self::Sparse(mixture_of_experts_weights) => {
                mixture_of_experts_weights.append_array_references(arrays);
            }
        }
    }
}

impl Qwen3_5MoEMixtureOfExpertsWeights {
    pub(super) fn append_array_references<'weights>(
        &'weights self,
        arrays: &mut Vec<&'weights MlxArray>,
    ) {
        match &self.router_projection {
            RouterGateWeights::Affine(affine_weights) => {
                affine_weights.append_array_references(arrays)
            }
            RouterGateWeights::Unquantized(w) => arrays.push(w),
        }
        self.shared_expert_gate_projection
            .append_array_references(arrays);
        self.shared_expert_up_projection
            .append_array_references(arrays);
        self.shared_expert_down_projection
            .append_array_references(arrays);
        self.shared_expert_output_gate_projection
            .append_array_references(arrays);
    }
}

/// All pre-bound weights for one Qwen3.5-MoE decoder layer.
#[derive(Debug)]
pub(super) struct Qwen3_5MoEDecoderLayerWeights {
    pub(super) input_normalization_weight: MlxArray,
    pub(super) attention_weights: Qwen3_5MoEAttentionWeights,
    pub(super) post_attention_normalization_weight: MlxArray,
    pub(super) mlp_weights: Qwen3_5DecoderLayerMlpWeights,
}

impl Qwen3_5MoEDecoderLayerWeights {
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
