use super::rope_descriptor::LagunaRopeDescriptor;

/// Attention implementation selected for one Laguna layer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LagunaAttentionKind {
    Full,
    Sliding,
}

/// Softplus attention output-gate granularity selected for one layer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LagunaGatingKind {
    None,
    PerHead,
    PerElement,
}

/// Canonical key/value cache topology selected from the attention kind.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LagunaCacheDescriptor {
    AppendOnly,
    Rotating { window_size: u32 },
}

/// Complete attention geometry for one ordered Laguna decoder layer.
#[derive(Clone, Debug, PartialEq)]
pub struct LagunaAttentionDescriptor {
    kind: LagunaAttentionKind,
    query_head_count: u32,
    key_value_head_count: u32,
    head_dimension: u32,
    gating_kind: LagunaGatingKind,
    rope: LagunaRopeDescriptor,
    cache: LagunaCacheDescriptor,
}

impl LagunaAttentionDescriptor {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        kind: LagunaAttentionKind,
        query_head_count: u32,
        key_value_head_count: u32,
        head_dimension: u32,
        gating_kind: LagunaGatingKind,
        rope: LagunaRopeDescriptor,
        cache: LagunaCacheDescriptor,
    ) -> Self {
        Self {
            kind,
            query_head_count,
            key_value_head_count,
            head_dimension,
            gating_kind,
            rope,
            cache,
        }
    }

    /// Returns full or sliding attention.
    #[must_use]
    pub const fn kind(&self) -> LagunaAttentionKind {
        self.kind
    }

    /// Returns this layer's query-head count.
    #[must_use]
    pub const fn query_head_count(&self) -> u32 {
        self.query_head_count
    }

    /// Returns this layer's key/value-head count.
    #[must_use]
    pub const fn key_value_head_count(&self) -> u32 {
        self.key_value_head_count
    }

    /// Returns the width of each query, key, and value head.
    #[must_use]
    pub const fn head_dimension(&self) -> u32 {
        self.head_dimension
    }

    /// Returns the layer's canonical gate granularity.
    #[must_use]
    pub const fn gating_kind(&self) -> LagunaGatingKind {
        self.gating_kind
    }

    /// Returns the rotary policy selected for this attention kind.
    #[must_use]
    pub const fn rope(&self) -> &LagunaRopeDescriptor {
        &self.rope
    }

    /// Returns append-only or bounded rotating cache topology.
    #[must_use]
    pub const fn cache(&self) -> &LagunaCacheDescriptor {
        &self.cache
    }
}

/// Dense SwiGLU geometry for one Laguna layer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LagunaDenseFeedForwardDescriptor {
    intermediate_size: u32,
}

impl LagunaDenseFeedForwardDescriptor {
    pub(super) const fn new(intermediate_size: u32) -> Self {
        Self { intermediate_size }
    }

    /// Returns the dense SwiGLU intermediate width.
    #[must_use]
    pub const fn intermediate_size(&self) -> u32 {
        self.intermediate_size
    }
}

/// Router scoring and selection behavior validated for sparse Laguna layers.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LagunaRouterKind {
    SigmoidTopK,
}

/// Sparse sigmoid top-K Mixture-of-Experts geometry for one Laguna layer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LagunaMoeDescriptor {
    router_kind: LagunaRouterKind,
    expert_count: u32,
    experts_per_token: u32,
    expert_intermediate_size: u32,
    shared_expert_intermediate_size: u32,
    normalizes_top_k_probabilities: bool,
    routed_scaling_factor: f64,
    applies_router_weight_on_input: bool,
}

impl LagunaMoeDescriptor {
    #[allow(clippy::too_many_arguments)]
    pub(super) const fn new(
        expert_count: u32,
        experts_per_token: u32,
        expert_intermediate_size: u32,
        shared_expert_intermediate_size: u32,
        normalizes_top_k_probabilities: bool,
        routed_scaling_factor: f64,
        applies_router_weight_on_input: bool,
    ) -> Self {
        Self {
            router_kind: LagunaRouterKind::SigmoidTopK,
            expert_count,
            experts_per_token,
            expert_intermediate_size,
            shared_expert_intermediate_size,
            normalizes_top_k_probabilities,
            routed_scaling_factor,
            applies_router_weight_on_input,
        }
    }

    /// Returns the validated router scoring and selection behavior.
    #[must_use]
    pub const fn router_kind(&self) -> LagunaRouterKind {
        self.router_kind
    }

    /// Returns the total routed expert count.
    #[must_use]
    pub const fn expert_count(&self) -> u32 {
        self.expert_count
    }

    /// Returns the routed top-K expert count per token.
    #[must_use]
    pub const fn experts_per_token(&self) -> u32 {
        self.experts_per_token
    }

    /// Returns each routed expert's intermediate width.
    #[must_use]
    pub const fn expert_intermediate_size(&self) -> u32 {
        self.expert_intermediate_size
    }

    /// Returns zero when no shared expert exists, otherwise its intermediate width.
    #[must_use]
    pub const fn shared_expert_intermediate_size(&self) -> u32 {
        self.shared_expert_intermediate_size
    }

    /// Returns whether top-K router probabilities are normalized.
    #[must_use]
    pub const fn normalizes_top_k_probabilities(&self) -> bool {
        self.normalizes_top_k_probabilities
    }

    /// Returns the routed expert output multiplier.
    #[must_use]
    pub const fn routed_scaling_factor(&self) -> f64 {
        self.routed_scaling_factor
    }

    /// Returns whether router weights multiply expert input rather than output.
    #[must_use]
    pub const fn applies_router_weight_on_input(&self) -> bool {
        self.applies_router_weight_on_input
    }
}

/// Canonical dense or sparse feed-forward implementation for one layer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LagunaFeedForwardDescriptor {
    Dense(LagunaDenseFeedForwardDescriptor),
    Moe(LagunaMoeDescriptor),
}

/// One fully normalized Laguna decoder layer in model order.
#[derive(Clone, Debug, PartialEq)]
pub struct LagunaLayerDescriptor {
    layer_index: usize,
    attention: LagunaAttentionDescriptor,
    feed_forward: LagunaFeedForwardDescriptor,
}

impl LagunaLayerDescriptor {
    pub(super) fn new(
        layer_index: usize,
        attention: LagunaAttentionDescriptor,
        feed_forward: LagunaFeedForwardDescriptor,
    ) -> Self {
        Self {
            layer_index,
            attention,
            feed_forward,
        }
    }

    /// Returns the zero-based decoder-layer index.
    #[must_use]
    pub const fn layer_index(&self) -> usize {
        self.layer_index
    }

    /// Returns the complete canonical attention descriptor.
    #[must_use]
    pub const fn attention(&self) -> &LagunaAttentionDescriptor {
        &self.attention
    }

    /// Returns the complete dense or Mixture-of-Experts descriptor.
    #[must_use]
    pub const fn feed_forward(&self) -> &LagunaFeedForwardDescriptor {
        &self.feed_forward
    }
}
