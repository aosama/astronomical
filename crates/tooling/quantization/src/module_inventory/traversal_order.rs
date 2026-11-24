//! The order mlx-lm visits a model's modules in, reproduced from names.
//!
//! OptiQ builds its sensitivity table by walking `model.named_modules()`,
//! and the allocator's block-aware floor breaks ties by that order: its sort
//! is stable, so among equally sensitive modules of a block the first one
//! visited is promoted. Ties are the rule, not the exception, for the static
//! scorer (every projection of a linear-attention block scores the same)
//! and for a uniform-baseline sweep (every module reads zero at the
//! baseline's own width). A table in any other order produces a different,
//! equally defensible bit map, which is exactly what a port must not do.
//!
//! mlx's walk is a depth-first traversal over a stack, so at every level
//! the children come out in the reverse of the order the model class
//! assigns them in `__init__`, and list items come out highest index first.
//! The assignment orders below are read from mlx-lm's `qwen3_5`,
//! `qwen3_5_moe` and `qwen3_next` model classes.

use std::cmp::Ordering;

#[cfg(test)]
#[path = "traversal_order_tests.rs"]
mod traversal_order_tests;

/// Children of each parent segment in the order the model class assigns
/// them. Dense and mixture-of-experts feed-forward children share the `mlp`
/// parent; the two sets are disjoint, so one list serves both.
const ASSIGNMENT_ORDER_BY_PARENT: [(&str, &[&str]); 7] = [
    ("language_model", &["model", "lm_head"]),
    ("model", &["embed_tokens", "layers", "norm"]),
    ("self_attn", &["q_proj", "k_proj", "v_proj", "o_proj"]),
    (
        "linear_attn",
        &[
            "in_proj_qkv",
            "in_proj_z",
            "in_proj_b",
            "in_proj_a",
            "out_proj",
        ],
    ),
    (
        "mlp",
        &[
            "gate",
            "switch_mlp",
            "shared_expert",
            "shared_expert_gate",
            "gate_proj",
            "down_proj",
            "up_proj",
        ],
    ),
    ("switch_mlp", &["gate_proj", "up_proj", "down_proj"]),
    ("shared_expert", &["gate_proj", "down_proj", "up_proj"]),
];

/// Children of a decoder layer (a numeric segment under `layers`).
const DECODER_LAYER_ASSIGNMENT_ORDER: [&str; 5] = [
    "linear_attn",
    "self_attn",
    "input_layernorm",
    "post_attention_layernorm",
    "mlp",
];

/// Orders two module names the way mlx-lm's `named_modules()` yields them.
///
/// Names are compared segment by segment. At the first differing segment
/// the one assigned later (or the higher list index) comes first. A segment
/// the tables do not know sorts after the known ones, in reverse name order,
/// so the result stays total and deterministic for other families.
#[must_use]
pub fn compare_in_mlx_lm_traversal_order(left_name: &str, right_name: &str) -> Ordering {
    let left_segments: Vec<&str> = left_name.split('.').collect();
    let right_segments: Vec<&str> = right_name.split('.').collect();
    let shared_segment_count = left_segments.len().min(right_segments.len());
    for segment_index in 0..shared_segment_count {
        let (left_segment, right_segment) =
            (left_segments[segment_index], right_segments[segment_index]);
        if left_segment == right_segment {
            continue;
        }
        let parent_segment = if segment_index == 0 {
            ""
        } else {
            left_segments[segment_index - 1]
        };
        return compare_sibling_segments(parent_segment, left_segment, right_segment);
    }
    left_segments.len().cmp(&right_segments.len())
}

/// Two different children of one parent: later assignment first.
fn compare_sibling_segments(
    parent_segment: &str,
    left_segment: &str,
    right_segment: &str,
) -> Ordering {
    if let (Ok(left_index), Ok(right_index)) =
        (left_segment.parse::<u64>(), right_segment.parse::<u64>())
    {
        return right_index.cmp(&left_index);
    }
    let left_rank = assignment_rank(parent_segment, left_segment);
    let right_rank = assignment_rank(parent_segment, right_segment);
    match (left_rank, right_rank) {
        (Some(left_rank), Some(right_rank)) => right_rank.cmp(&left_rank),
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (None, None) => right_segment.cmp(left_segment),
    }
}

/// Where `segment` falls in its parent's assignment order, if known.
fn assignment_rank(parent_segment: &str, segment: &str) -> Option<usize> {
    let assignment_order: &[&str] = if parent_segment.parse::<u64>().is_ok() {
        &DECODER_LAYER_ASSIGNMENT_ORDER
    } else {
        let mut known_order: Option<&[&str]> = None;
        for (known_parent, parent_order) in ASSIGNMENT_ORDER_BY_PARENT {
            if known_parent == parent_segment {
                known_order = Some(parent_order);
            }
        }
        known_order?
    };
    for (rank, assigned_segment) in assignment_order.iter().enumerate() {
        if *assigned_segment == segment {
            return Some(rank);
        }
    }
    None
}
