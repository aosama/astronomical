//! The draft tree contract of `draft_tree_verification`, run on the
//! streaming model, the one text model: every row matches feeding its chain
//! alone there, a commit leaves the streamed state feeding that branch
//! leaves, and a commit of another branch differs.
//!
//! On the same small dense Qwen3.5 and the same branching tree. The pass is
//! the shared core's; this proves the streaming model's own layer step and
//! evaluation under it. Layers of experts held in part are proven through
//! the engine's reply loop, where their tables exist.
//!
//! Run with `cargo test-streaming-draft-tree-verification`, seconds; the
//! same contract on another model with
//! `cargo test-configured-streaming-draft-tree-verification --configuration PATH`.

use mlx_bridge::MlxArray;
use model_serving::speculative_decoding::tree_rows::TreeRows;
use model_serving::{PerformanceAttribution, Qwen3_5TextModel};
use serial_test::serial;

use super::draft_tree_verification::{
    State, TreeContractTarget, Verified, anchor_position,
    assert_a_commit_differs_from_feeding_another_branch,
    assert_a_commit_leaves_the_state_feeding_the_branch_leaves,
    assert_every_tree_row_matches_its_chain, chunk_against_step_logit_difference,
};
use super::drafted_rewind::{
    MAXIMUM_LOGIT_DIFFERENCE, SmallTarget, load_target, small_target_sources, target_sources,
};

type SmallStreamedTarget = SmallTarget<Qwen3_5TextModel>;

fn load_small_streamed_target() -> SmallStreamedTarget {
    load_streamed_target(small_target_sources())
}

/// The test run's own model (`--configuration`'s `model_directory`), all of
/// it in memory: the contract then runs over that model's expert layers.
fn load_configured_streamed_target() -> SmallStreamedTarget {
    let model_directory = test_harness::test_run::TestRunContext::current()
        .configuration
        .model_directory
        .expect("the configured-model contract needs --configuration with a model_directory");
    load_streamed_target(target_sources(model_directory, 40))
}

fn load_streamed_target(sources: super::drafted_rewind::SmallTargetSources) -> SmallStreamedTarget {
    let mut target = load_target(sources);
    target
        .model
        .enable_draft_tree_verification()
        .expect("the tree kernel declares");
    target
}

impl TreeContractTarget for SmallStreamedTarget {
    fn prefilled_state(&self) -> State {
        let mut state = State::empty_from_astro_decoder_state_cache_layout_with_full_attention_kv_state_growth_tokens(
            &self.model.astro_decoder_state_cache_layout(),
            256,
        )
        .expect("the decoder state allocates");
        self.logits_after(&self.prompt_token_ids, 0, &mut state);
        state
    }

    fn logits_after(&self, token_ids: &[u32], first_position: u32, state: &mut State) -> MlxArray {
        self.model
            .forward_chunk(token_ids, first_position, state)
            .expect("the forward runs")
    }

    /// The streamed pass is evaluated where it is verified, as the engine
    /// runs it, so a refused lookup would be served again here too.
    fn verified_tree(&self, tree: &TreeRows, state: &mut State) -> Verified {
        self.model
            .verify_draft_tree_resolving_expert_routes(
                tree,
                anchor_position(self),
                state,
                &[],
                &mut PerformanceAttribution::disabled(),
            )
            .expect("the tree verifies")
    }
}

#[test]
#[ignore = "loads the small dense Qwen3.5 on the streaming engine"]
#[serial]
fn should_give_every_row_of_the_small_model_s_streamed_tree_the_logits_its_chain_alone_gives() {
    assert_every_tree_row_matches_its_chain(
        &load_small_streamed_target(),
        MAXIMUM_LOGIT_DIFFERENCE,
    );
}

#[test]
#[ignore = "loads the small dense Qwen3.5 on the streaming engine"]
#[serial]
fn should_leave_the_small_model_s_streamed_state_feeding_the_kept_branch_alone_leaves() {
    assert_a_commit_leaves_the_state_feeding_the_branch_leaves(
        &load_small_streamed_target(),
        MAXIMUM_LOGIT_DIFFERENCE,
    );
}

#[test]
#[ignore = "loads the small dense Qwen3.5 on the streaming engine"]
#[serial]
fn should_differ_from_feeding_a_branch_the_small_model_s_streamed_commit_did_not_keep() {
    assert_a_commit_differs_from_feeding_another_branch(
        &load_small_streamed_target(),
        MAXIMUM_LOGIT_DIFFERENCE * 4.0,
    );
}

/// One chunk's rounding is a sample, not a bound: on a 35B-A3B the sample
/// is 1.62, tree rows differ from their chains by 1.06 to 1.62 and a commit
/// by up to 1.78, where a wrong branch moves the logits by 3.55.
const ROUNDING_SAMPLE_SPREAD: f32 = 1.5;

/// A configured model's rows are held to that model's own rounding between
/// a chunk and single tokens, never to less than the small model's.
fn configured_target_and_its_rounding() -> (SmallStreamedTarget, f32) {
    let target = load_configured_streamed_target();
    let rounding = (chunk_against_step_logit_difference(&target) * ROUNDING_SAMPLE_SPREAD)
        .max(MAXIMUM_LOGIT_DIFFERENCE);
    (target, rounding)
}

#[test]
#[ignore = "loads the configured Qwen3.5 model on the streaming engine"]
#[serial]
fn should_give_every_row_of_the_configured_model_s_streamed_tree_the_logits_its_chain_gives() {
    let (target, rounding) = configured_target_and_its_rounding();
    assert_every_tree_row_matches_its_chain(&target, rounding);
}

#[test]
#[ignore = "loads the configured Qwen3.5 model on the streaming engine"]
#[serial]
fn should_leave_the_configured_model_s_streamed_state_feeding_the_kept_branch_leaves() {
    let (target, rounding) = configured_target_and_its_rounding();
    assert_a_commit_leaves_the_state_feeding_the_branch_leaves(&target, rounding);
}

#[test]
#[ignore = "loads the configured Qwen3.5 model on the streaming engine"]
#[serial]
fn should_differ_from_feeding_a_branch_the_configured_model_s_streamed_commit_did_not_keep() {
    let (target, rounding) = configured_target_and_its_rounding();
    assert_a_commit_differs_from_feeding_another_branch(&target, rounding);
}
