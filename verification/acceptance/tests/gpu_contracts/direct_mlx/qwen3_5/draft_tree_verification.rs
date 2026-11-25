//! A draft tree verified in one pass must give every row what its own chain
//! alone gives, and committing one branch must leave the target exactly
//! where feeding only that branch would.
//!
//! On the small dense Qwen3.5, which mixes DeltaNet and attention layers as
//! the 27B does. The tree branches twice under the anchor and once below, so
//! sibling rows share a position, read different convolution windows and
//! start their recurrence from different parents.
//!
//! The contract is written once, over `TreeContractTarget`, and run on the
//! resident model here and on the streaming model in
//! `streaming_draft_tree_verification`: the pass is the shared core's, and
//! each engine brings its own layer step and evaluation to it.

use mlx_bridge::MlxArray;
use model_serving::speculative_decoding::dflash_drafter::DraftTree;
use model_serving::speculative_decoding::tree_rows::TreeRows;
use model_serving::{DraftTreeTarget, Qwen3_5DraftTreeVerification, RequestDecoderStateStack};
use runtime_integration::MlxRuntime;
use serial_test::serial;

use super::drafted_rewind::{
    BLOCK, MAXIMUM_LOGIT_DIFFERENCE, SmallTarget, forward, load_small_target, prefilled_state,
};

/// A request's decoder state, and a verified tree, as the contract names them.
pub(super) type State = RequestDecoderStateStack;
pub(super) type Verified = Qwen3_5DraftTreeVerification;

/// What the contract needs from a loaded target that the two engines do
/// their own way. The runtime and the commit come from `DraftTreeTarget`.
pub(super) trait TreeContractTarget {
    fn prefilled_state(&self) -> State;
    /// `[1, 1, vocabulary]`: the logits after the last of `token_ids`.
    fn logits_after(&self, token_ids: &[u32], first_position: u32, state: &mut State) -> MlxArray;
    fn verified_tree(&self, tree: &TreeRows, state: &mut State) -> Verified;
}

impl TreeContractTarget for SmallTarget {
    fn prefilled_state(&self) -> State {
        prefilled_state(self)
    }

    fn logits_after(&self, token_ids: &[u32], first_position: u32, state: &mut State) -> MlxArray {
        forward(self, token_ids, first_position, state)
    }

    fn verified_tree(&self, tree: &TreeRows, state: &mut State) -> Verified {
        self.model
            .verify_draft_tree(tree, anchor_position(self), state, &[])
            .expect("the tree verifies")
    }
}

/// The position after the target's prompt, where a tree's anchor sits.
pub(super) fn anchor_position<Model>(target: &SmallTarget<Model>) -> u32 {
    target.prompt_token_ids.len() as u32
}

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

fn float32_values(runtime: &MlxRuntime, logits: &MlxArray) -> Vec<f32> {
    runtime
        .array_to_vec_f32(
            &runtime
                .astype(logits, mlx_bridge::MlxDtype::Float32)
                .expect("the logits widen"),
        )
        .expect("the logits read back")
}

fn largest_logit_difference(runtime: &MlxRuntime, left: &MlxArray, right: &MlxArray) -> f32 {
    let mut largest = 0.0_f32;
    for (left_value, right_value) in float32_values(runtime, left)
        .iter()
        .zip(&float32_values(runtime, right))
    {
        largest = largest.max((left_value - right_value).abs());
    }
    largest
}

fn row_logits(runtime: &MlxRuntime, per_row_logits: &MlxArray, row: usize) -> MlxArray {
    let vocabulary = per_row_logits.shape()[2];
    runtime
        .slice(
            per_row_logits,
            &[0, row as i32, 0],
            &[1, row as i32 + 1, vocabulary],
            &[1, 1, 1],
        )
        .expect("one row slices")
}

/// Every row of the tree must have the logits feeding its chain alone gives.
pub(super) fn assert_every_tree_row_matches_its_chain<Model: DraftTreeTarget>(
    target: &SmallTarget<Model>,
) where
    SmallTarget<Model>: TreeContractTarget,
{
    let tree = branching_tree();
    let mut state = target.prefilled_state();
    let hand_offs_before = graph_host_step::evaluation_counts();
    let verification = target.verified_tree(&tree, &mut state);
    let pass_hand_offs = graph_host_step::evaluation_counts().since(hand_offs_before);
    // A pass wants at most one blocking evaluation, its own; every further
    // one is a wait a round pays on every layer that makes it.
    eprintln!(
        "[draft-tree] one pass: blocking_evaluations={} asynchronous_evaluations={} cross_stream_waits={}",
        pass_hand_offs.blocking_evaluations,
        pass_hand_offs.asynchronous_evaluations,
        pass_hand_offs.cross_stream_waits
    );
    for row in 0..tree.row_count() {
        let mut chain_state = target.prefilled_state();
        let alone = target.logits_after(
            &chain_tokens(&tree, row),
            anchor_position(target),
            &mut chain_state,
        );
        let in_tree = row_logits(
            target.model.draft_tree_model_base().runtime(),
            &verification.per_row_logits,
            row,
        );
        let difference = largest_logit_difference(
            target.model.draft_tree_model_base().runtime(),
            &in_tree,
            &alone,
        );
        eprintln!("[draft-tree] row {row}: largest logit difference {difference:.4}");
        assert!(
            difference <= MAXIMUM_LOGIT_DIFFERENCE,
            "row {row}: logits differ by {difference}"
        );
    }
}

/// Verifies the tree, commits `kept_rows`, and returns the logits of feeding
/// `next_token` after them.
fn next_logits_after_commit<Model: DraftTreeTarget>(
    target: &SmallTarget<Model>,
    kept_rows: &[usize],
    next_token: u32,
) -> MlxArray
where
    SmallTarget<Model>: TreeContractTarget,
{
    let tree = branching_tree();
    let mut state = target.prefilled_state();
    let verification = target.verified_tree(&tree, &mut state);
    target
        .model
        .commit_draft_tree(&mut state, verification, kept_rows)
        .expect("the kept chain commits");
    target.logits_after(
        &[next_token],
        anchor_position(target) + kept_rows.len() as u32,
        &mut state,
    )
}

fn next_logits_after_feeding<Model: DraftTreeTarget>(
    target: &SmallTarget<Model>,
    chain: &[u32],
    next_token: u32,
) -> MlxArray
where
    SmallTarget<Model>: TreeContractTarget,
{
    let mut state = target.prefilled_state();
    target.logits_after(chain, anchor_position(target), &mut state);
    target.logits_after(
        &[next_token],
        anchor_position(target) + chain.len() as u32,
        &mut state,
    )
}

/// Committing a branch must leave the state feeding that branch leaves.
pub(super) fn assert_a_commit_leaves_the_state_feeding_the_branch_leaves<Model: DraftTreeTarget>(
    target: &SmallTarget<Model>,
) where
    SmallTarget<Model>: TreeContractTarget,
{
    let tree = branching_tree();
    // The deep branch (rows 0, 1, 3, 5) and the sibling branch (rows 0, 1, 4).
    for kept_rows in [vec![0, 1, 3, 5], vec![0, 1, 4], vec![0, 2], vec![0]] {
        let chain: Vec<u32> = kept_rows.iter().map(|row| tree.token_ids[*row]).collect();
        let committed = next_logits_after_commit(target, &kept_rows, BLOCK[6]);
        let fed = next_logits_after_feeding(target, &chain, BLOCK[6]);
        let difference = largest_logit_difference(
            target.model.draft_tree_model_base().runtime(),
            &committed,
            &fed,
        );
        eprintln!("[draft-tree] kept rows {kept_rows:?}: largest logit difference {difference:.4}");
        assert!(
            difference <= MAXIMUM_LOGIT_DIFFERENCE,
            "kept rows {kept_rows:?}: logits differ by {difference}"
        );
    }
}

/// The control: a commit of one branch must not look like feeding another.
pub(super) fn assert_a_commit_differs_from_feeding_another_branch<Model: DraftTreeTarget>(
    target: &SmallTarget<Model>,
) where
    SmallTarget<Model>: TreeContractTarget,
{
    let tree = branching_tree();
    let committed = next_logits_after_commit(target, &[0, 1, 3], BLOCK[6]);
    let other_branch = [tree.token_ids[0], tree.token_ids[1], tree.token_ids[4]];
    let fed = next_logits_after_feeding(target, &other_branch, BLOCK[6]);
    let difference = largest_logit_difference(
        target.model.draft_tree_model_base().runtime(),
        &committed,
        &fed,
    );
    eprintln!("[draft-tree] control, other branch: largest logit difference {difference:.4}");
    assert!(
        difference > MAXIMUM_LOGIT_DIFFERENCE * 4.0,
        "the control must expose a commit of the wrong branch"
    );
}

fn tree_target() -> SmallTarget {
    let mut target = load_small_target();
    target
        .model
        .enable_draft_tree_verification()
        .expect("the tree kernel declares");
    target
}

#[test]
#[ignore = "loads the small dense Qwen3.5 on the GPU"]
#[serial]
fn should_give_every_tree_row_the_logits_its_chain_alone_gives() {
    assert_every_tree_row_matches_its_chain(&tree_target());
}

#[test]
#[ignore = "loads the small dense Qwen3.5 on the GPU"]
#[serial]
fn should_leave_the_state_feeding_the_kept_branch_alone_leaves() {
    assert_a_commit_leaves_the_state_feeding_the_branch_leaves(&tree_target());
}

#[test]
#[ignore = "loads the small dense Qwen3.5 on the GPU"]
#[serial]
fn should_differ_from_feeding_a_branch_it_did_not_commit() {
    assert_a_commit_differs_from_feeding_another_branch(&tree_target());
}
