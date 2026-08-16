/// Physical member of one canonical weight or affine sidecar.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum LagunaTensorComponent {
    Weight,
    Scales,
    Biases,
    WeightGlobalScale,
    InputGlobalScale,
    LogicalShape,
    ZeroPoint,
    AttentionKeyScaleMetadata,
    AttentionValueScaleMetadata,
}

/// Model-wide tensor purpose independent of an artifact namespace.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum LagunaGlobalTensorRole {
    TokenEmbedding,
    FinalNormalization,
    OutputHead,
}

/// Canonical attention projection purpose within one decoder layer.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum LagunaAttentionProjection {
    Query,
    Key,
    Value,
    Output,
    Gate,
}

/// Canonical SwiGLU projection after artifact gate/up fusion is removed.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum LagunaExpertProjection {
    Gate,
    Up,
    Down,
}

/// Decoder-layer tensor purpose independent of source aliases and packaging.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum LagunaLayerTensorRole {
    InputNormalization,
    PostAttentionNormalization,
    Attention(LagunaAttentionProjection),
    AttentionQueryNormalization,
    AttentionKeyNormalization,
    DenseFeedForward(LagunaExpertProjection),
    Router,
    RouterCorrectionBias,
    SharedExpert(LagunaExpertProjection),
    SharedExpertGate,
    RoutedExpert(LagunaExpertProjection),
}

/// Structured downstream identity that cannot carry a raw artifact name.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum LagunaTensorId {
    Global {
        role: LagunaGlobalTensorRole,
        component: LagunaTensorComponent,
    },
    Layer {
        layer_index: usize,
        role: LagunaLayerTensorRole,
        component: LagunaTensorComponent,
    },
}
