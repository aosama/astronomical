//! Stacked affine versus unstacked per-expert on-disk dialects.
//!
//! The first executable dialect is stacked MLX affine. Unstacked
//! `experts.{i}` tensors are rejected until a later normalizer exists.

/// Classifies expert tensor naming in a safetensors index.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum K2HorizonMoVAWeightDialect {
    StackedAffine,
    UnstackedPerExpert,
    Unknown,
}

impl K2HorizonMoVAWeightDialect {
    /// Inspects weight-map keys without reading payloads.
    #[must_use]
    pub fn from_tensor_names<'a, Names>(tensor_names: Names) -> Self
    where
        Names: IntoIterator<Item = &'a str>,
    {
        let mut saw_stacked_feed_forward = false;
        let mut saw_stacked_value_experts = false;
        let mut saw_unstacked = false;
        for tensor_name in tensor_names {
            if tensor_name.contains(".mlp.switch_mlp.") {
                saw_stacked_feed_forward = true;
            }
            if tensor_name.contains(".self_attn.v_experts.weight")
                || tensor_name.contains(".self_attn.v_experts.scales")
            {
                saw_stacked_value_experts = true;
            }
            if tensor_name.contains(".mlp.experts.0.")
                || tensor_name.contains(".self_attn.v_experts.0.")
            {
                saw_unstacked = true;
            }
        }
        if saw_unstacked && !saw_stacked_feed_forward && !saw_stacked_value_experts {
            return Self::UnstackedPerExpert;
        }
        if saw_stacked_feed_forward || saw_stacked_value_experts {
            return Self::StackedAffine;
        }
        Self::Unknown
    }
}
