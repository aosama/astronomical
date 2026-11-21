//! A draft tree verified in one pass must give every row what its own chain
//! alone gives, and committing one branch must leave the target exactly
//! where feeding only that branch would.
//!
//! On the small dense Qwen3.5, which mixes DeltaNet and attention layers as
//! the 27B does. The tree branches twice under the anchor and once below, so
//! sibling rows share a position, read different convolution windows and
//! start their recurrence from different parents.

use mlx_bridge::MlxArray;
use model_serving::speculative_decoding::dflash_drafter::DraftTree;
use model_serving::speculative_decoding::tree_rows::TreeRows;
use serial_test::serial;

use super::drafted_rewind::{
    BLOCK, MAXIMUM_LOGIT_DIFFERENCE, SmallTarget, forward, largest_difference, load_small_target,
    prefilled_state,
};

/// Under the anchor `BLOCK[0]`: the block's own chain three deep, a sibling
/// at the first slot, and a sibling at the second.
fn branching_tree() -> TreeRows {
    TreeRows::from_draft_tree(
        BLOCK[0],
        &DraftTree {
            token_ids: vec![BLOCK[1], 1_000, BLOCK[2], 2_000, BLOCK[3]],
            parent_indices: vec![None, None, Some(0), Some(0), Some(2)],
            depths: vec![0, 0, 1, 1, 2],
        },
    )
}

/// Each row's chain from the anchor, as the tokens feeding it alone would be.
fn chain_tokens(tree: &TreeRows, row: usize) -> Vec<u32> {
    let mut reversed = Vec::new();
    let mut current = Some(row);
    while let Some(current_row) = current {
        reversed.push(tree.token_ids[current_row]);
        current = tree.parent_rows[current_row];
    }
    reversed.reverse();
    reversed
}

fn tree_target() -> SmallTarget {
    let mut target = load_small_target();
    target
        .model
        .enable_draft_tree_verification()
        .expect("the tree kernel declares");
    target
}

fn row_logits(target: &SmallTarget, per_row_logits: &MlxArray, row: usize) -> MlxArray {
    let vocabulary = per_row_logits.shape()[2];
    target
        .model
        .runtime()
        .slice(
            per_row_logits,
            &[0, row as i32, 0],
            &[1, row as i32 + 1, vocabulary],
            &[1, 1, 1],
        )
        .expect("one row slices")
}

#[test]
#[ignore = "loads the small dense Qwen3.5 on the GPU"]
#[serial]
fn should_give_every_tree_row_the_logits_its_chain_alone_gives() {
    let target = tree_target();
    let tree = branching_tree();
    let anchor_position = target.prompt_token_ids.len() as u32;
    let mut state = prefilled_state(&target);
    let verification = target
        .model
        .verify_draft_tree(&tree, anchor_position, &mut state, &[])
        .expect("the tree verifies");
    for row in 0..tree.row_count() {
        let mut chain_state = prefilled_state(&target);
        let alone = forward(
            &target,
            &chain_tokens(&tree, row),
            anchor_position,
            &mut chain_state,
        );
        let in_tree = row_logits(&target, &verification.per_row_logits, row);
        let difference = largest_difference(&target, &in_tree, &alone);
        eprintln!("[draft-tree] row {row}: largest logit difference {difference:.4}");
        assert!(
            difference <= MAXIMUM_LOGIT_DIFFERENCE,
            "row {row}: logits differ by {difference}"
        );
    }
}

/// Verifies the tree, commits `kept_rows`, and returns the logits of feeding
/// `next_token` after them.
fn next_logits_after_commit(
    target: &SmallTarget,
    kept_rows: &[usize],
    next_token: u32,
) -> MlxArray {
    let tree = branching_tree();
    let anchor_position = target.prompt_token_ids.len() as u32;
    let mut state = prefilled_state(target);
    let verification = target
        .model
        .verify_draft_tree(&tree, anchor_position, &mut state, &[])
        .expect("the tree verifies");
    target
        .model
        .commit_draft_tree(&mut state, verification, kept_rows)
        .expect("the kept chain commits");
    forward(
        target,
        &[next_token],
        anchor_position + kept_rows.len() as u32,
        &mut state,
    )
}

fn next_logits_after_feeding(target: &SmallTarget, chain: &[u32], next_token: u32) -> MlxArray {
    let anchor_position = target.prompt_token_ids.len() as u32;
    let mut state = prefilled_state(target);
    forward(target, chain, anchor_position, &mut state);
    forward(
        target,
        &[next_token],
        anchor_position + chain.len() as u32,
        &mut state,
    )
}

#[test]
#[ignore = "loads the small dense Qwen3.5 on the GPU"]
#[serial]
fn should_leave_the_state_feeding_the_kept_branch_alone_leaves() {
    let target = tree_target();
    let tree = branching_tree();
    // The deep branch (rows 0, 1, 3, 5) and the sibling branch (rows 0, 1, 4).
    for kept_rows in [vec![0, 1, 3, 5], vec![0, 1, 4], vec![0, 2], vec![0]] {
        let chain: Vec<u32> = kept_rows.iter().map(|row| tree.token_ids[*row]).collect();
        let committed = next_logits_after_commit(&target, &kept_rows, BLOCK[6]);
        let fed = next_logits_after_feeding(&target, &chain, BLOCK[6]);
        let difference = largest_difference(&target, &committed, &fed);
        eprintln!("[draft-tree] kept rows {kept_rows:?}: largest logit difference {difference:.4}");
        assert!(
            difference <= MAXIMUM_LOGIT_DIFFERENCE,
            "kept rows {kept_rows:?}: logits differ by {difference}"
        );
    }
}

#[test]
#[ignore = "loads the small dense Qwen3.5 on the GPU"]
#[serial]
fn should_differ_from_feeding_a_branch_it_did_not_commit() {
    let target = tree_target();
    let tree = branching_tree();
    let committed = next_logits_after_commit(&target, &[0, 1, 3], BLOCK[6]);
    let other_branch = [tree.token_ids[0], tree.token_ids[1], tree.token_ids[4]];
    let fed = next_logits_after_feeding(&target, &other_branch, BLOCK[6]);
    let difference = largest_difference(&target, &committed, &fed);
    eprintln!("[draft-tree] control, other branch: largest logit difference {difference:.4}");
    assert!(
        difference > MAXIMUM_LOGIT_DIFFERENCE * 4.0,
        "the control must expose a commit of the wrong branch"
    );
}
