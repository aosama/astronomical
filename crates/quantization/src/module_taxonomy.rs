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

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

#[cfg(test)]
#[path = "module_taxonomy_tests.rs"]
mod module_taxonomy_tests;

/// The output-projection / embedding keywords OptiQ protects in
/// `_identify_output_layers`, matched case-insensitively as a substring of the
/// last segment or as a whole second-to-last segment.
const OUTPUT_KEYWORDS: &[&str] = &[
    "lm_head",
    "embed_tokens",
    "wte",
    "wpe",
    "output_proj",
    "embed_out",
    "head",
];

/// The dense feed-forward projection suffixes inside a transformer block
/// (`gate_proj`/`up_proj`/`down_proj` are the SwiGLU MLP; `fc1`/`fc2` the plain
/// MLP). Anything else still inside a block falls through to
/// [`ModuleRole::OtherInsideBlock`].
pub const DENSE_FEEDFORWARD_PROJECTION_SUFFIXES: &[&str] =
    &["gate_proj", "up_proj", "down_proj", "fc1", "fc2"];

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
    // Equivalent to OptiQ's regex `layers\.(\d+)` for every name in the field,
    // where the block index is always its own all-digit segment: scan adjacent
    // segment pairs and take the first whose left is the literal `layers` and
    // whose right segment is all digits, so a stray non-numeric segment after
    // one `layers` does not shadow a later real block index.
    let segments: Vec<&str> = module_name.split('.').collect();
    segments.windows(2).find_map(|pair| {
        // Only a `layers` left segment counts; parse its right segment and let
        // a non-numeric one fall through as `None` so the scan keeps going.
        if pair[0] == "layers" {
            pair[1].parse::<usize>().ok()
        } else {
            None
        }
    })
}

/// Classifies a module by its path alone, first match wins.
///
/// Precedence mirrors how OptiQ's separate name predicates compose, using the
/// allocator's tight taxonomy rather than the looser shardwise requantizer
/// rule:
/// 1. Router (`_identify_moe_protected_layers`): `last == "gate" && prev == "mlp"`,
///    `last == "router"`, or `last == "proj" && prev == "router"`. Deliberately
///    narrower than the shardwise `is_router_tensor_key`: `expert_bias`,
///    `.ffn.gate.`, and `.router.` are NOT routers here.
/// 2. Shared expert (`_identify_moe_protected_layers`): a `.shared_experts.`
///    or `.shared_expert.` segment (DeepSeek's singular, Qwen/Gemma's plural).
/// 3. Routed expert (`shardwise.is_expert`): `.experts.`, `.switch_mlp.`, or
///    `.switch_glu.`, with the shared-expert case already excluded above.
/// 4. Embedding or head (`_identify_output_layers`): an output keyword in the
///    last or second-to-last segment, matched case-insensitively.
/// 5. Outside blocks: no `layers` segment at all.
/// 6. Attention suffix tables, dense feed-forward suffixes, else other.
#[must_use]
pub fn classify_module(module_name: &str) -> ModuleRole {
    let segments: Vec<&str> = module_name.split('.').collect();
    let last = segments.last().copied().unwrap_or("");
    // `prev` is the segment before `last`; empty when the name has one segment.
    let prev = segments
        .len()
        .checked_sub(2)
        .map(|index| segments[index])
        .unwrap_or("");

    // 1. Router (OptiQ's `_identify_moe_protected_layers` router branch). Note
    //    this is the tight allocator predicate: `expert_bias` and `.ffn.gate.`
    //    are NOT routers here, only the looser shardwise requantizer rule
    //    (`is_router_tensor_key`) treats them as such.
    if (last == "gate" && prev == "mlp") || last == "router" || (last == "proj" && prev == "router")
    {
        return ModuleRole::MixtureOfExpertsRouter;
    }

    // 2. Shared-expert sub-MLP (OptiQ's `_identify_moe_protected_layers` shared
    //    branch). The singular `shared_expert` covers DeepSeek-V3, the plural
    //    `shared_experts` the Qwen/Gemma naming.
    if module_name.contains(".shared_experts.") || module_name.contains(".shared_expert.") {
        return ModuleRole::SharedExpert;
    }

    // 3. Routed expert (OptiQ's `shardwise.is_expert`): `.experts.`,
    //    `.switch_mlp.`, or `.switch_glu.`, with the shared-expert case already
    //    excluded by the earlier branch. The shardwise `is_expert` also rejects
    //    any broader `.shared_` prefix, but no field module nests a shared
    //    sub-tensor under an expert index, so the narrower step-2 patterns are
    //    sufficient and more precise here.
    if module_name.contains(".experts.")
        || module_name.contains(".switch_mlp.")
        || module_name.contains(".switch_glu.")
    {
        return ModuleRole::RoutedExpert;
    }

    // 4. Embedding or output head (OptiQ's `_identify_output_layers` keyword
    //    branch), matched case-insensitively.
    if is_embedding_or_output_head(module_name) {
        return ModuleRole::EmbeddingOrOutputHead;
    }

    // 5. Outside transformer blocks: no `layers` segment at all.
    if !segments.contains(&"layers") {
        return ModuleRole::OutsideTransformerBlocks;
    }

    // 6. Attention families, then dense feed-forward, else other inside block.
    if FULL_ATTENTION_PROJECTION_SUFFIXES.contains(&last) {
        return ModuleRole::FullAttentionProjection;
    }
    if LINEAR_ATTENTION_PROJECTION_SUFFIXES.contains(&last) {
        return ModuleRole::LinearAttentionProjection;
    }
    if DENSE_FEEDFORWARD_PROJECTION_SUFFIXES.contains(&last) {
        return ModuleRole::DenseFeedForward;
    }
    ModuleRole::OtherInsideBlock
}

/// True when the name carries an [`OUTPUT_KEYWORDS`] embedding/output marker,
/// mirroring OptiQ's `_identify_output_layers`: a keyword appearing as a
/// substring of the last segment, or as the whole second-to-last segment.
fn is_embedding_or_output_head(module_name: &str) -> bool {
    let lowered = module_name.to_ascii_lowercase();
    let segments: Vec<&str> = lowered.split('.').collect();
    let Some(last) = segments.last().copied() else {
        return false;
    };
    let second_to_last = segments.len().checked_sub(2).map(|index| segments[index]);
    OUTPUT_KEYWORDS
        .iter()
        .any(|&keyword| last.contains(keyword) || second_to_last.is_some_and(|seg| seg == keyword))
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
/// Names are grouped by [`transformer_block_index`]; the first and last
/// `boundary_block_count` distinct block indices form the edge set (a single
/// block, or a count at or above the block total, is not double-counted), and
/// the matching modules are returned in the caller's original order so the
/// result is deterministic. A `boundary_block_count` of zero returns nothing;
/// that is a deliberate departure from OptiQ's `_identify_boundary_layers`,
/// whose `sorted[-0:]` slice would select every block.
#[must_use]
pub fn boundary_block_modules<'names>(
    module_names: &[&'names str],
    boundary_block_count: usize,
) -> Vec<&'names str> {
    if boundary_block_count == 0 {
        return Vec::new();
    }

    // One block index per name; names with no `layers` segment are excluded,
    // exactly as OptiQ's `_identify_boundary_layers` skips them.
    let block_indices: Vec<Option<usize>> = module_names
        .iter()
        .map(|name| transformer_block_index(name))
        .collect();

    // Sorted distinct block indices, then the first and last
    // `boundary_block_count` of them as the protected edge set. A `BTreeSet`
    // both deduplicates (when the first and last windows overlap on few
    // blocks) and keeps membership checks cheap.
    let distinct_indices: Vec<usize> = block_indices
        .iter()
        .copied()
        .flatten()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let edge_count = boundary_block_count.min(distinct_indices.len());
    let mut edge_blocks: BTreeSet<usize> = BTreeSet::new();
    for index in distinct_indices
        .iter()
        .take(edge_count)
        .chain(distinct_indices.iter().rev().take(edge_count))
    {
        edge_blocks.insert(*index);
    }

    // Emit edge-block members in the caller's original order so the result is
    // deterministic (block 0's modules first, then the last block's).
    module_names
        .iter()
        .zip(block_indices.iter())
        .filter(|(_, index)| index.is_some_and(|block| edge_blocks.contains(&block)))
        .map(|(name, _)| *name)
        .collect()
}
