//! How many nodes a draft tree verifies past its anchor, chosen once per
//! target when its drafter loads.
//!
//! A round pays the target's forward over every tree row, and the tokens
//! it keeps level off near three, so the best size is where one more row
//! costs more than the tokens it adds. The two target kinds land far apart:
//!
//! - A dense target reads every weight once a forward whatever its rows,
//!   so a plain step is already expensive and extra rows are nearly free.
//!   TensorFold's 15-node tree serves the 27B.
//! - A mixture-of-experts target reads each row's own experts, so every row
//!   adds about 1.6 ms of expert reads to a plain step of about 14 ms on the
//!   35B-A3B. Measured with chat at temperature 1.0
//!   (`cargo measure-dflash2-tree-rounds`, the tree-size ladder): 3 nodes
//!   decoded 91 tok/s against 70 plain, 7 nodes 78 and 15 nodes 52. TensorFold
//!   likewise verifies 4 to 5 rows a round on this family.
//!
//! The 3 nodes stay free to branch. Over 16 interleaved seeds of the same
//! chat (`cargo measure-dflash2-tree-rounds`, the depth comparisons):
//!
//! | Drafting | Kept a round | ms a round | Decode |
//! |---|---|---|---|
//! | 3-node tree (this rule) | 2.45 | 26.7-27.3 | 89.7-91.8 tok/s |
//! | 3-node chain, always 3 drafts deep | 2.27 | 26.7 | 84.9 tok/s |
//! | 4-node tree, up to 4 deep | 2.56 | 30.4 | 84.2 tok/s |
//! | 4-node chain, always 4 deep | 2.35 | 30.5 | 77.0 tok/s |
//!
//! A chain wastes its later nodes whenever its first guess is rejected,
//! where a tree spends them on alternatives for the next token; the tree kept
//! more on every seed at equal cost. A fourth node keeps 0.11 more tokens a
//! round but adds about 3 ms of expert reads, and its fifth row passes the
//! row-exact tree attention kernel's four.

use engine_core::speculative_decoding::dflash_drafter::DraftTreeOptions;

/// Nodes past the anchor for a mixture-of-experts target: a 4-row round.
pub const MIXTURE_OF_EXPERTS_NODE_BUDGET: usize = 3;

/// The tree options a target drafts with: the default (TensorFold's 15-node
/// tree) for a dense target, and the small tree for a target with experts.
#[must_use]
pub fn draft_tree_options_for_target(expert_count: u32) -> DraftTreeOptions {
    let default_options = DraftTreeOptions::default();
    if expert_count == 0 {
        return default_options;
    }
    DraftTreeOptions {
        node_budget: MIXTURE_OF_EXPERTS_NODE_BUDGET,
        ..default_options
    }
}
