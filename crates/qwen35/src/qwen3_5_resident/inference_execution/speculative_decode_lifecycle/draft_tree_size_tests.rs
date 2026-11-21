use super::draft_tree_size::{MIXTURE_OF_EXPERTS_NODE_BUDGET, draft_tree_options_for_target};
use crate::speculative_decoding::dflash_drafter::DraftTreeOptions;

#[test]
fn should_keep_the_default_fifteen_node_tree_for_a_dense_target() {
    assert_eq!(
        draft_tree_options_for_target(0),
        DraftTreeOptions::default()
    );
    assert_eq!(draft_tree_options_for_target(0).node_budget, 15);
}

#[test]
fn should_verify_a_four_row_tree_for_a_mixture_of_experts_target() {
    let options = draft_tree_options_for_target(256);
    assert_eq!(options.node_budget, MIXTURE_OF_EXPERTS_NODE_BUDGET);
    assert_eq!(options.node_budget + 1, 4);
}

/// Only the size changes: the search keeps TensorFold's scoring, so a
/// smaller tree still follows the same best paths first.
#[test]
fn should_change_only_the_node_budget_for_a_mixture_of_experts_target() {
    let options = draft_tree_options_for_target(1);
    let default_options = DraftTreeOptions::default();
    assert_eq!(
        DraftTreeOptions {
            node_budget: default_options.node_budget,
            ..options
        },
        default_options
    );
}
