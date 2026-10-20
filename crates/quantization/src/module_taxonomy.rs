//! Name-based classification of model modules.
//!
//! OptiQ decides which modules are protected, which are attention, which are
//! routers, and which are routed experts purely from the module path, with no
//! model config injected. That keeps the allocator architecture-agnostic, and
//! the same rules are needed in four places: the allocator's protected set,
//! the block-aware floor's attention carve-out, the static structural scorer,
//! and the shard-wise structural bit policy. This module is the single owner
//! of those rules, so a new architecture's naming is taught once.
//!
//! The initial vocabulary is OptiQ's, which covers every family OptiQ ships
//! for (Qwen3.5/3.6 dense and mixture-of-experts, Gemma-4, DeepSeek-V3/V4,
//! LFM2, Nemotron-H, Laguna). Astronomical serves a subset of these, and the
//! port keeps the full table because the rules are cheap and dropping one is
//! how a family's routers end up quantized.

use serde::{Deserialize, Serialize};

#[cfg(test)]
#[path = "module_taxonomy_tests.rs"]
mod module_taxonomy_tests;

/// What a module does in the network, as far as quantization policy cares.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModuleRole {
    /// The token embedding table or the output head (`embed_tokens`,
    /// `lm_head`, `wte`, `wpe`, `embed_out`, `output_proj`, `head`). Always
    /// protected: cheap in parameters and pathological to quantize.
    EmbeddingOrOutputHead,
    /// Any module not inside a `layers.N` block that is not an embedding or
    /// head (final norm scales, multi-token-prediction heads). Also protected.
    OutsideTransformerBlocks,
    /// A full-attention projection (`q_proj`, `k_proj`, `v_proj`, `o_proj`,
    /// fused `qkv_proj`/`wqkv`/`Wqkv`/`attn_qkv`).
    FullAttentionProjection,
    /// A linear-attention / GatedDeltaNet projection (`in_proj_qkv`,
    /// `in_proj_a`, `in_proj_b`, `in_proj_z`, `out_proj`).
    LinearAttentionProjection,
    /// The mixture-of-experts router: `*.mlp.gate` (Qwen3, where the SwiGLU
    /// gate is `gate_proj` and bare `gate` is the router), `*.router`,
    /// `*.router.proj` (Gemma-4), `*.ffn.gate.*`, `expert_bias`. Never
    /// quantized by the requantizer; protected by the allocator.
    MixtureOfExpertsRouter,
    /// A shared-expert dense sub-MLP that fires for every token
    /// (`.shared_experts.` or `.shared_expert.` segment). Protected.
    SharedExpert,
    /// A routed expert projection, fused (`switch_mlp`, `switch_glu`) or
    /// unfused (`experts.N`). Holds most of a mixture-of-experts model's
    /// parameters and tolerates the least precision per byte.
    RoutedExpert,
    /// A dense feed-forward projection (`gate_proj`, `up_proj`, `down_proj`,
    /// `fc1`, `fc2`) inside a transformer block.
    DenseFeedForward,
    /// Anything else inside a transformer block (hyper-connections, n-gram
    /// shards, convolution weights that happen to be 2-D).
    OtherInsideBlock,
}

/// The attention projection suffixes OptiQ reserves floor slots for.
pub const FULL_ATTENTION_PROJECTION_SUFFIXES: &[&str] = &[
    "q_proj", "k_proj", "v_proj", "o_proj", "qkv_proj", "wqkv", "Wqkv", "attn_qkv",
];

/// The linear-attention (GatedDeltaNet) projection suffixes.
pub const LINEAR_ATTENTION_PROJECTION_SUFFIXES: &[&str] = &[
    "in_proj_qkv",
    "in_proj_a",
    "in_proj_b",
    "in_proj_z",
    "out_proj",
];

/// Extracts the transformer block index from `...layers.7.mlp.up_proj`.
///
/// Returns `None` for modules outside any block. The regex OptiQ uses is
/// `layers\.(\d+)`; the port matches the first `layers` path segment followed
/// by an all-digit segment, which is equivalent for every name in the field.
#[must_use]
pub fn transformer_block_index(module_name: &str) -> Option<usize> {
    let _ = module_name;
    todo!(
        "split on '.', find the first segment equal to \"layers\" whose successor parses as usize"
    )
}

/// Classifies a module by its path alone.
///
/// Work to port, in precedence order (first match wins, exactly as OptiQ's
/// separate predicates compose):
/// 1. Router patterns (`is_router`): `last == "gate" && previous == "mlp"`,
///    `last == "router"`, `last == "proj" && previous == "router"`, the key
///    ends in `.gate.weight`/`.gate`, contains `.ffn.gate.`, `expert_bias`,
///    or `.router.`.
/// 2. Shared expert: any `.shared_experts.` or `.shared_expert.` segment, and
///    the shardwise `.shared_` prefix form.
/// 3. Routed expert (`is_expert`): `.experts.`, `.switch_mlp.`, `.switch_glu.`
///    with the shared-expert case already excluded above.
/// 4. Embedding or head (`_identify_output_layers`): one of the output
///    keywords in the last or second-to-last segment, matched case-insensitively.
/// 5. Outside blocks: no `layers` segment at all.
/// 6. Attention suffix tables, dense feed-forward suffixes, else other.
#[must_use]
pub fn classify_module(module_name: &str) -> ModuleRole {
    let _ = module_name;
    todo!("apply the precedence-ordered predicates documented above")
}

/// True for the roles the allocator always boosts: embeddings and heads,
/// modules outside blocks, routers, and shared experts.
#[must_use]
pub fn is_always_protected(role: ModuleRole) -> bool {
    matches!(
        role,
        ModuleRole::EmbeddingOrOutputHead
            | ModuleRole::OutsideTransformerBlocks
            | ModuleRole::MixtureOfExpertsRouter
            | ModuleRole::SharedExpert
    )
}

/// True for either attention family; the block-aware floor reserves its first
/// slots for the most-sensitive attention components of each block.
#[must_use]
pub fn is_attention_component(role: ModuleRole) -> bool {
    matches!(
        role,
        ModuleRole::FullAttentionProjection | ModuleRole::LinearAttentionProjection
    )
}

/// Returns every module in the first `boundary_block_count` and the last
/// `boundary_block_count` transformer blocks, the `protect_first_last` set.
///
/// Work to port: group names by [`transformer_block_index`], sort the block
/// indices, take the first and last `boundary_block_count` of them.
#[must_use]
pub fn boundary_block_modules<'names>(
    module_names: &[&'names str],
    boundary_block_count: usize,
) -> Vec<&'names str> {
    let _ = (module_names, boundary_block_count);
    todo!("group by block index and collect the modules of the edge blocks")
}
