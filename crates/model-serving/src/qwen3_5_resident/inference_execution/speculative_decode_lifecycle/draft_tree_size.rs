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

use crate::speculative_decoding::dflash2_drafter::DraftTreeOptions;

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
