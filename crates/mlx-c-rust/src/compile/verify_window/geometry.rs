//! Static geometry and frozen input order for the MTP verification window's
//! compiled graph.
//!
//! One verification window replays a fixed row count through the hybrid
//! decoder trunk: gated-delta layers with rolling-convolution and recurrent
//! state leaves, and full-attention layers whose key/value slabs enter as
//! fixed-shape inputs masked by a dynamic logical offset. The MLX-C closure
//! ABI carries no capture context, so the builder reads a thread-local
//! geometry while MLX traces, and every frozen weight and state leaf is a
//! positional compile input.
//!
//! Both sides of the ABI walk [`verify_window_input_slots`]: the model crate
//! assembles the input vector from its live weights and state leaves, and the
//! graph builder consumes the vector in the same order.

/// The attention family of one decoder layer position.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VerifyWindowLayerKind {
    /// Linear-attention layer with rolling convolution and recurrent state.
    GatedDelta,
    /// Append-only attention layer with a fixed-shape key/value slab.
    FullAttention,
}

/// One affine module's member arrays, in frozen order.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VerifyWindowAffineSlot {
    PackedWeight,
    Scales,
    Biases,
}

/// One gated-delta layer weight array, in frozen order.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VerifyWindowGatedDeltaWeightSlot {
    InputQueriesKeysValues(VerifyWindowAffineSlot),
    OutputGate(VerifyWindowAffineSlot),
    UpdateRate(VerifyWindowAffineSlot),
    DecayInterval(VerifyWindowAffineSlot),
    ConvolutionWeight,
    DecayIntervalBias,
    DecayRateLogarithm,
    NormalizationWeight,
    OutputProjection(VerifyWindowAffineSlot),
}

/// One full-attention layer weight array, in frozen order.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VerifyWindowFullAttentionWeightSlot {
    Query(VerifyWindowAffineSlot),
    Key(VerifyWindowAffineSlot),
    Value(VerifyWindowAffineSlot),
    Output(VerifyWindowAffineSlot),
    QueryNormalization,
    KeyNormalization,
}

/// One dense SwiGLU feed-forward weight array, in frozen order.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VerifyWindowFeedForwardWeightSlot {
    Gate(VerifyWindowAffineSlot),
    Up(VerifyWindowAffineSlot),
    Down(VerifyWindowAffineSlot),
}

/// One trunk weight array, in frozen order.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VerifyWindowTrunkWeightSlot {
    Embedding(VerifyWindowAffineSlot),
    FinalNormalization,
    LanguageModelHead(VerifyWindowAffineSlot),
}

/// The per-layer weight arrays, in frozen order.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VerifyWindowLayerWeightSlot {
    InputNormalization,
    PostAttentionNormalization,
    GatedDelta(VerifyWindowGatedDeltaWeightSlot),
    FullAttention(VerifyWindowFullAttentionWeightSlot),
    FeedForward(VerifyWindowFeedForwardWeightSlot),
}

/// One input-vector position, in frozen order.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VerifyWindowInputSlot {
    /// `[1, rows]` int32 token identifiers.
    TokenIndices,
    /// `[rows]` int32 positions, one per window row.
    PositionOffsets,
    /// Scalar int32 key/value slab logical offset for attention masking.
    KeyValueBaseOffset,
    /// Per-channel folded query RMS-norm scale for gated-delta prework.
    QueryNormalizationScale,
    /// Per-channel folded key RMS-norm scale for gated-delta prework.
    KeyNormalizationScale,
    /// `[1, kernel - 1, conv_dim]` rolling convolution state.
    GatedDeltaRollingState { layer_index: usize },
    /// `[1, value_heads, value_dim, key_dim]` float32 recurrent state.
    GatedDeltaRecurrentState { layer_index: usize },
    /// `[1, capacity, key_value_heads, head_dim]` key slab.
    FullAttentionKeysSlab { layer_index: usize },
    /// `[1, capacity, key_value_heads, head_dim]` value slab.
    FullAttentionValuesSlab { layer_index: usize },
    /// Per-layer raw normalization or weight array.
    LayerWeight {
        layer_index: usize,
        slot: VerifyWindowLayerWeightSlot,
    },
    /// Trunk weight array.
    TrunkWeight(VerifyWindowTrunkWeightSlot),
}

/// One affine module's quantization geometry.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VerifyWindowQuantizationPair {
    pub group_size: i32,
    pub bits: i32,
}

/// Every quantized module one gated-delta layer consumes, in slot order.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VerifyWindowGatedDeltaQuantization {
    pub input_queries_keys_values: VerifyWindowQuantizationPair,
    pub output_gate: VerifyWindowQuantizationPair,
    pub update_rate: VerifyWindowQuantizationPair,
    pub decay_interval: VerifyWindowQuantizationPair,
    pub output_projection: VerifyWindowQuantizationPair,
    pub feed_forward_gate: VerifyWindowQuantizationPair,
    pub feed_forward_up: VerifyWindowQuantizationPair,
    pub feed_forward_down: VerifyWindowQuantizationPair,
}

/// Every quantized module one full-attention layer consumes, in slot order.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VerifyWindowFullAttentionQuantization {
    pub query: VerifyWindowQuantizationPair,
    pub key: VerifyWindowQuantizationPair,
    pub value: VerifyWindowQuantizationPair,
    pub output: VerifyWindowQuantizationPair,
    pub feed_forward_gate: VerifyWindowQuantizationPair,
    pub feed_forward_up: VerifyWindowQuantizationPair,
    pub feed_forward_down: VerifyWindowQuantizationPair,
}

/// The trunk modules' quantization geometry.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VerifyWindowTrunkQuantization {
    pub embedding: VerifyWindowQuantizationPair,
    pub language_model_head: VerifyWindowQuantizationPair,
}

/// Static geometry for one row count's compiled verification window.
#[derive(Debug)]
pub struct VerifyWindowGeometry {
    row_count: i32,
    layer_kinds: Vec<VerifyWindowLayerKind>,
    query_head_count: i32,
    key_value_head_count: i32,
    attention_head_dimension: i32,
    rotary_dimension: i32,
    linear_key_head_count: i32,
    linear_value_head_count: i32,
    linear_head_dimension: i32,
    linear_key_dimension: i32,
    linear_convolution_dimension: i32,
    linear_convolution_kernel_dimension: i32,
    layer_quantization: Vec<VerifyWindowLayerQuantization>,
    trunk_quantization: VerifyWindowTrunkQuantization,
    rms_norm_epsilon: f32,
    rope_base: f32,
    attention_scale: f32,
}

/// One decoder layer's quantization geometry, indexed by layer position.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VerifyWindowLayerQuantization {
    pub gated_delta: Option<VerifyWindowGatedDeltaQuantization>,
    pub full_attention: Option<VerifyWindowFullAttentionQuantization>,
}

impl VerifyWindowGeometry {
    /// Creates geometry from validated model facts. Every layer position
    /// needs a quantization entry; a layer position whose entry does not
    /// match its kind declines before compilation.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        row_count: i32,
        layer_kinds: Vec<VerifyWindowLayerKind>,
        query_head_count: i32,
        key_value_head_count: i32,
        attention_head_dimension: i32,
        rotary_dimension: i32,
        linear_key_head_count: i32,
        linear_value_head_count: i32,
        linear_head_dimension: i32,
        linear_key_dimension: i32,
        linear_convolution_dimension: i32,
        linear_convolution_kernel_dimension: i32,
        layer_quantization: Vec<VerifyWindowLayerQuantization>,
        trunk_quantization: VerifyWindowTrunkQuantization,
        rms_norm_epsilon: f32,
        rope_base: f32,
    ) -> Self {
        Self {
            row_count,
            layer_kinds,
            query_head_count,
            key_value_head_count,
            attention_head_dimension,
            rotary_dimension,
            linear_key_head_count,
            linear_value_head_count,
            linear_head_dimension,
            linear_key_dimension,
            linear_convolution_dimension,
            linear_convolution_kernel_dimension,
            layer_quantization,
            trunk_quantization,
            rms_norm_epsilon,
            rope_base,
            attention_scale: (attention_head_dimension as f32).sqrt().recip(),
        }
    }

    /// The window's fixed row count (draft depth plus one).
    pub fn row_count(&self) -> i32 {
        self.row_count
    }

    /// The ordered decoder-layer attention families.
    pub fn layer_kinds(&self) -> &[VerifyWindowLayerKind] {
        &self.layer_kinds
    }

    pub(super) fn layer_quantization(
        &self,
        layer_index: usize,
    ) -> Option<&VerifyWindowLayerQuantization> {
        self.layer_quantization.get(layer_index)
    }

    pub(super) fn trunk_quantization(&self) -> &VerifyWindowTrunkQuantization {
        &self.trunk_quantization
    }

    pub(super) fn query_head_count(&self) -> i32 {
        self.query_head_count
    }

    pub(super) fn key_value_head_count(&self) -> i32 {
        self.key_value_head_count
    }

    pub(super) fn attention_head_dimension(&self) -> i32 {
        self.attention_head_dimension
    }

    pub(super) fn rotary_dimension(&self) -> i32 {
        self.rotary_dimension
    }

    pub(super) fn linear_key_head_count(&self) -> i32 {
        self.linear_key_head_count
    }

    pub(super) fn linear_value_head_count(&self) -> i32 {
        self.linear_value_head_count
    }

    pub(super) fn linear_head_dimension(&self) -> i32 {
        self.linear_head_dimension
    }

    pub(super) fn linear_key_dimension(&self) -> i32 {
        self.linear_key_dimension
    }

    pub(super) fn linear_convolution_dimension(&self) -> i32 {
        self.linear_convolution_dimension
    }

    pub(super) fn linear_convolution_kernel_dimension(&self) -> i32 {
        self.linear_convolution_kernel_dimension
    }

    pub(super) fn rms_norm_epsilon(&self) -> f32 {
        self.rms_norm_epsilon
    }

    pub(super) fn rope_base(&self) -> f32 {
        self.rope_base
    }

    pub(super) fn attention_scale(&self) -> f32 {
        self.attention_scale
    }
}

/// The frozen input order for one geometry's compiled window.
pub fn verify_window_input_slots(geometry: &VerifyWindowGeometry) -> Vec<VerifyWindowInputSlot> {
    let mut slots = vec![
        VerifyWindowInputSlot::TokenIndices,
        VerifyWindowInputSlot::PositionOffsets,
        VerifyWindowInputSlot::KeyValueBaseOffset,
        VerifyWindowInputSlot::QueryNormalizationScale,
        VerifyWindowInputSlot::KeyNormalizationScale,
    ];
    for layer_index in 0..geometry.layer_kinds.len() {
        match geometry.layer_kinds[layer_index] {
            VerifyWindowLayerKind::GatedDelta => {
                slots.push(VerifyWindowInputSlot::GatedDeltaRollingState { layer_index });
                slots.push(VerifyWindowInputSlot::GatedDeltaRecurrentState { layer_index });
                slots.push(VerifyWindowInputSlot::LayerWeight {
                    layer_index,
                    slot: VerifyWindowLayerWeightSlot::InputNormalization,
                });
                slots.extend(gated_delta_weight_slots(layer_index));
                slots.push(VerifyWindowInputSlot::LayerWeight {
                    layer_index,
                    slot: VerifyWindowLayerWeightSlot::PostAttentionNormalization,
                });
                slots.extend(feed_forward_weight_slots(layer_index));
            }
            VerifyWindowLayerKind::FullAttention => {
                slots.push(VerifyWindowInputSlot::FullAttentionKeysSlab { layer_index });
                slots.push(VerifyWindowInputSlot::FullAttentionValuesSlab { layer_index });
                slots.push(VerifyWindowInputSlot::LayerWeight {
                    layer_index,
                    slot: VerifyWindowLayerWeightSlot::InputNormalization,
                });
                slots.extend(full_attention_weight_slots(layer_index));
                slots.push(VerifyWindowInputSlot::LayerWeight {
                    layer_index,
                    slot: VerifyWindowLayerWeightSlot::PostAttentionNormalization,
                });
                slots.extend(feed_forward_weight_slots(layer_index));
            }
        }
    }
    slots.push(VerifyWindowInputSlot::TrunkWeight(
        VerifyWindowTrunkWeightSlot::Embedding(VerifyWindowAffineSlot::PackedWeight),
    ));
    slots.push(VerifyWindowInputSlot::TrunkWeight(
        VerifyWindowTrunkWeightSlot::Embedding(VerifyWindowAffineSlot::Scales),
    ));
    slots.push(VerifyWindowInputSlot::TrunkWeight(
        VerifyWindowTrunkWeightSlot::Embedding(VerifyWindowAffineSlot::Biases),
    ));
    slots.push(VerifyWindowInputSlot::TrunkWeight(
        VerifyWindowTrunkWeightSlot::FinalNormalization,
    ));
    slots.push(VerifyWindowInputSlot::TrunkWeight(
        VerifyWindowTrunkWeightSlot::LanguageModelHead(VerifyWindowAffineSlot::PackedWeight),
    ));
    slots.push(VerifyWindowInputSlot::TrunkWeight(
        VerifyWindowTrunkWeightSlot::LanguageModelHead(VerifyWindowAffineSlot::Scales),
    ));
    slots.push(VerifyWindowInputSlot::TrunkWeight(
        VerifyWindowTrunkWeightSlot::LanguageModelHead(VerifyWindowAffineSlot::Biases),
    ));
    slots
}

fn affine_slots() -> [VerifyWindowAffineSlot; 3] {
    [
        VerifyWindowAffineSlot::PackedWeight,
        VerifyWindowAffineSlot::Scales,
        VerifyWindowAffineSlot::Biases,
    ]
}

fn gated_delta_weight_slots(layer_index: usize) -> Vec<VerifyWindowInputSlot> {
    let mut slots = Vec::with_capacity(19);
    for affine_slot in affine_slots() {
        slots.push(VerifyWindowInputSlot::LayerWeight {
            layer_index,
            slot: VerifyWindowLayerWeightSlot::GatedDelta(
                VerifyWindowGatedDeltaWeightSlot::InputQueriesKeysValues(affine_slot),
            ),
        });
    }
    for affine_slot in affine_slots() {
        slots.push(VerifyWindowInputSlot::LayerWeight {
            layer_index,
            slot: VerifyWindowLayerWeightSlot::GatedDelta(
                VerifyWindowGatedDeltaWeightSlot::OutputGate(affine_slot),
            ),
        });
    }
    for affine_slot in affine_slots() {
        slots.push(VerifyWindowInputSlot::LayerWeight {
            layer_index,
            slot: VerifyWindowLayerWeightSlot::GatedDelta(
                VerifyWindowGatedDeltaWeightSlot::UpdateRate(affine_slot),
            ),
        });
    }
    for affine_slot in affine_slots() {
        slots.push(VerifyWindowInputSlot::LayerWeight {
            layer_index,
            slot: VerifyWindowLayerWeightSlot::GatedDelta(
                VerifyWindowGatedDeltaWeightSlot::DecayInterval(affine_slot),
            ),
        });
    }
    slots.push(VerifyWindowInputSlot::LayerWeight {
        layer_index,
        slot: VerifyWindowLayerWeightSlot::GatedDelta(
            VerifyWindowGatedDeltaWeightSlot::ConvolutionWeight,
        ),
    });
    slots.push(VerifyWindowInputSlot::LayerWeight {
        layer_index,
        slot: VerifyWindowLayerWeightSlot::GatedDelta(
            VerifyWindowGatedDeltaWeightSlot::DecayIntervalBias,
        ),
    });
    slots.push(VerifyWindowInputSlot::LayerWeight {
        layer_index,
        slot: VerifyWindowLayerWeightSlot::GatedDelta(
            VerifyWindowGatedDeltaWeightSlot::DecayRateLogarithm,
        ),
    });
    slots.push(VerifyWindowInputSlot::LayerWeight {
        layer_index,
        slot: VerifyWindowLayerWeightSlot::GatedDelta(
            VerifyWindowGatedDeltaWeightSlot::NormalizationWeight,
        ),
    });
    for affine_slot in affine_slots() {
        slots.push(VerifyWindowInputSlot::LayerWeight {
            layer_index,
            slot: VerifyWindowLayerWeightSlot::GatedDelta(
                VerifyWindowGatedDeltaWeightSlot::OutputProjection(affine_slot),
            ),
        });
    }
    slots
}

fn full_attention_weight_slots(layer_index: usize) -> Vec<VerifyWindowInputSlot> {
    let mut slots = Vec::with_capacity(14);
    for affine_slot in affine_slots() {
        slots.push(VerifyWindowInputSlot::LayerWeight {
            layer_index,
            slot: VerifyWindowLayerWeightSlot::FullAttention(
                VerifyWindowFullAttentionWeightSlot::Query(affine_slot),
            ),
        });
    }
    for affine_slot in affine_slots() {
        slots.push(VerifyWindowInputSlot::LayerWeight {
            layer_index,
            slot: VerifyWindowLayerWeightSlot::FullAttention(
                VerifyWindowFullAttentionWeightSlot::Key(affine_slot),
            ),
        });
    }
    for affine_slot in affine_slots() {
        slots.push(VerifyWindowInputSlot::LayerWeight {
            layer_index,
            slot: VerifyWindowLayerWeightSlot::FullAttention(
                VerifyWindowFullAttentionWeightSlot::Value(affine_slot),
            ),
        });
    }
    for affine_slot in affine_slots() {
        slots.push(VerifyWindowInputSlot::LayerWeight {
            layer_index,
            slot: VerifyWindowLayerWeightSlot::FullAttention(
                VerifyWindowFullAttentionWeightSlot::Output(affine_slot),
            ),
        });
    }
    slots.push(VerifyWindowInputSlot::LayerWeight {
        layer_index,
        slot: VerifyWindowLayerWeightSlot::FullAttention(
            VerifyWindowFullAttentionWeightSlot::QueryNormalization,
        ),
    });
    slots.push(VerifyWindowInputSlot::LayerWeight {
        layer_index,
        slot: VerifyWindowLayerWeightSlot::FullAttention(
            VerifyWindowFullAttentionWeightSlot::KeyNormalization,
        ),
    });
    slots
}

fn feed_forward_weight_slots(layer_index: usize) -> Vec<VerifyWindowInputSlot> {
    let mut slots = Vec::with_capacity(9);
    for affine_slot in affine_slots() {
        slots.push(VerifyWindowInputSlot::LayerWeight {
            layer_index,
            slot: VerifyWindowLayerWeightSlot::FeedForward(
                VerifyWindowFeedForwardWeightSlot::Gate(affine_slot),
            ),
        });
    }
    for affine_slot in affine_slots() {
        slots.push(VerifyWindowInputSlot::LayerWeight {
            layer_index,
            slot: VerifyWindowLayerWeightSlot::FeedForward(VerifyWindowFeedForwardWeightSlot::Up(
                affine_slot,
            )),
        });
    }
    for affine_slot in affine_slots() {
        slots.push(VerifyWindowInputSlot::LayerWeight {
            layer_index,
            slot: VerifyWindowLayerWeightSlot::FeedForward(
                VerifyWindowFeedForwardWeightSlot::Down(affine_slot),
            ),
        });
    }
    slots
}
