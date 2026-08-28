use std::collections::BTreeSet;

use super::storage_descriptor::{
    LagunaBlockFp8Profile, LagunaNvfp4Profile, LagunaSymmetricPackedAffineProfile,
};

/// Canonical module categories selected by evidenced compressed-tensors patterns.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum LagunaCompressedModuleScope {
    AllMatrices,
    AllLinear,
    DenseFeedForward,
    RoutedExperts,
    SharedExpert,
}

/// Canonical exclusions retained from an evidenced compressed-tensors declaration.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum LagunaCompressedIgnoreScope {
    OutputHead,
    AttentionQuery,
    AttentionKey,
    AttentionValue,
    AttentionOutput,
    AttentionGate,
    Router,
    SharedExpert,
    DenseFeedForward {
        layer_index: usize,
        projection: LagunaCompressedFeedForwardProjection,
    },
}

/// One canonical feed-forward projection used by exact compressed exclusions.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum LagunaCompressedFeedForwardProjection {
    Gate,
    Up,
    Down,
}

/// Exact physical weight encoding selected for targeted modules.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LagunaCompressedWeightEncoding {
    SymmetricPackedAffine(LagunaSymmetricPackedAffineProfile),
    TwoLevelNvfp4(LagunaNvfp4Profile),
    BlockFp8(LagunaBlockFp8Profile),
}

/// Retained input-activation quantization required by compressed execution.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LagunaCompressedInputActivationDescriptor {
    Nvfp4TensorGroup(LagunaNvfp4InputActivationDescriptor),
    Fp8Group(LagunaFp8InputActivationDescriptor),
}

impl LagunaCompressedInputActivationDescriptor {
    #[must_use]
    pub const fn bits(&self) -> u32 {
        match self {
            Self::Nvfp4TensorGroup(descriptor) => descriptor.bits(),
            Self::Fp8Group(descriptor) => descriptor.bits(),
        }
    }

    #[must_use]
    pub const fn group_size(&self) -> u32 {
        match self {
            Self::Nvfp4TensorGroup(descriptor) => descriptor.group_size(),
            Self::Fp8Group(descriptor) => descriptor.group_size(),
        }
    }
}

/// Exact local dynamic NVFP4 activation profile with E4M3 scales.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LagunaNvfp4InputActivationDescriptor;

impl LagunaNvfp4InputActivationDescriptor {
    pub(super) const fn evidenced() -> Self {
        Self
    }

    #[must_use]
    pub const fn bits(&self) -> u32 {
        4
    }

    #[must_use]
    pub const fn group_size(&self) -> u32 {
        16
    }
}

/// Exact dynamic FP8 activation profile grouped by 128 values.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LagunaFp8InputActivationDescriptor;

impl LagunaFp8InputActivationDescriptor {
    pub(super) const fn evidenced() -> Self {
        Self
    }

    #[must_use]
    pub const fn bits(&self) -> u32 {
        8
    }

    #[must_use]
    pub const fn group_size(&self) -> u32 {
        128
    }
}

/// Exact symmetric per-tensor FP8 key/value-cache declaration.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LagunaFp8KvCacheDescriptor;

impl LagunaFp8KvCacheDescriptor {
    pub(super) const fn evidenced() -> Self {
        Self
    }

    #[must_use]
    pub const fn bits(&self) -> u32 {
        8
    }
}

/// Strict canonical compressed-tensors semantics used by artifact binding.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LagunaCompressedStorageDescriptor {
    weight_encoding: LagunaCompressedWeightEncoding,
    target_scopes: BTreeSet<LagunaCompressedModuleScope>,
    ignored_scopes: BTreeSet<LagunaCompressedIgnoreScope>,
    input_activations: Option<LagunaCompressedInputActivationDescriptor>,
    kv_cache: Option<LagunaFp8KvCacheDescriptor>,
}

impl LagunaCompressedStorageDescriptor {
    pub(super) fn new(
        weight_encoding: LagunaCompressedWeightEncoding,
        target_scopes: BTreeSet<LagunaCompressedModuleScope>,
        ignored_scopes: BTreeSet<LagunaCompressedIgnoreScope>,
        input_activations: Option<LagunaCompressedInputActivationDescriptor>,
        kv_cache: Option<LagunaFp8KvCacheDescriptor>,
    ) -> Self {
        Self {
            weight_encoding,
            target_scopes,
            ignored_scopes,
            input_activations,
            kv_cache,
        }
    }

    #[must_use]
    pub const fn weight_encoding(&self) -> LagunaCompressedWeightEncoding {
        self.weight_encoding
    }

    #[must_use]
    pub const fn input_activations(&self) -> Option<LagunaCompressedInputActivationDescriptor> {
        self.input_activations
    }

    #[must_use]
    pub const fn kv_cache(&self) -> Option<LagunaFp8KvCacheDescriptor> {
        self.kv_cache
    }

    /// Applies structured selector semantics without executing arbitrary regular expressions.
    #[must_use]
    pub fn applies_to_module(&self, canonical_module_name: &str) -> bool {
        let is_targeted = self
            .target_scopes
            .contains(&LagunaCompressedModuleScope::AllMatrices)
            || self
                .target_scopes
                .contains(&LagunaCompressedModuleScope::AllLinear)
                && canonical_module_name != "model.embed_tokens"
                && !is_router_module(canonical_module_name)
            || module_scope(canonical_module_name)
                .is_some_and(|scope| self.target_scopes.contains(&scope));
        is_targeted && !self.is_ignored(canonical_module_name)
    }

    fn is_ignored(&self, canonical_module_name: &str) -> bool {
        self.ignored_scopes
            .iter()
            .any(|scope| scope.matches(canonical_module_name))
    }
}

fn is_router_module(canonical_module_name: &str) -> bool {
    canonical_module_name.ends_with(".mlp.gate")
        || canonical_module_name.ends_with(".mlp.gate.proj")
}

fn module_scope(canonical_module_name: &str) -> Option<LagunaCompressedModuleScope> {
    if canonical_module_name.contains(".mlp.shared_expert.") {
        return Some(LagunaCompressedModuleScope::SharedExpert);
    }
    if canonical_module_name.contains(".mlp.switch_mlp.") {
        return Some(LagunaCompressedModuleScope::RoutedExperts);
    }
    if canonical_module_name.contains(".mlp.")
        && canonical_module_name
            .rsplit('.')
            .next()
            .is_some_and(|name| matches!(name, "gate_proj" | "up_proj" | "down_proj"))
    {
        return Some(LagunaCompressedModuleScope::DenseFeedForward);
    }
    None
}

impl LagunaCompressedIgnoreScope {
    fn matches(self, canonical_module_name: &str) -> bool {
        let suffix = canonical_module_name.rsplit('.').next();
        match self {
            Self::OutputHead => canonical_module_name == "lm_head",
            Self::AttentionQuery => {
                suffix == Some("q_proj") && canonical_module_name.contains(".self_attn.")
            }
            Self::AttentionKey => {
                suffix == Some("k_proj") && canonical_module_name.contains(".self_attn.")
            }
            Self::AttentionValue => {
                suffix == Some("v_proj") && canonical_module_name.contains(".self_attn.")
            }
            Self::AttentionOutput => {
                suffix == Some("o_proj") && canonical_module_name.contains(".self_attn.")
            }
            Self::AttentionGate => {
                suffix == Some("g_proj") && canonical_module_name.contains(".self_attn.")
            }
            Self::Router => is_router_module(canonical_module_name),
            Self::SharedExpert => canonical_module_name.contains(".mlp.shared_expert."),
            Self::DenseFeedForward {
                layer_index,
                projection,
            } => {
                canonical_module_name
                    == format!(
                        "model.layers.{layer_index}.mlp.{}",
                        projection.module_suffix()
                    )
            }
        }
    }
}

impl LagunaCompressedFeedForwardProjection {
    const fn module_suffix(self) -> &'static str {
        match self {
            Self::Gate => "gate_proj",
            Self::Up => "up_proj",
            Self::Down => "down_proj",
        }
    }
}
