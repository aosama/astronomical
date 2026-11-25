//! The draft tree contract of `draft_tree_verification`, run on the
//! streaming model: every row matches feeding its chain alone there, a
//! commit leaves the streamed state feeding that branch leaves, and a
//! commit of another branch differs.
//!
//! On the same small dense Qwen3.5 and the same branching tree. The pass is
//! the shared core's; this proves the streaming model's own layer step and
//! evaluation under it. Layers of experts held in part are proven through
//! the engine's reply loop, where their tables exist.
//!
//! Run with `cargo test-streaming-draft-tree-verification`. Seconds.

use mlx_bridge::MlxArray;
use model_serving::speculative_decoding::tree_rows::TreeRows;
use model_serving::{PerformanceAttribution, Qwen3_5StreamingModel};
use serial_test::serial;

use super::draft_tree_verification::{
    State, TreeContractTarget, Verified, anchor_position,
    assert_a_commit_differs_from_feeding_another_branch,
    assert_a_commit_leaves_the_state_feeding_the_branch_leaves,
    assert_every_tree_row_matches_its_chain,
};
use super::drafted_rewind::{SmallTarget, small_target_sources};

type SmallStreamedTarget = SmallTarget<Qwen3_5StreamingModel>;

fn load_small_streamed_target() -> SmallStreamedTarget {
    let sources = small_target_sources();
    let mut model = Qwen3_5StreamingModel::load(
        sources.runtime,
        sources.artifact,
        &sources.model_directory,
        crate::common::standard_qwen3_5_model_chunking_configuration(),
    )
    .expect("the model loads on the streaming engine");
    model
        .materialize_target_weights()
        .expect("the weights materialize");
    model
        .enable_draft_tree_verification()
        .expect("the tree kernel declares");
    SmallTarget {
        model,
        prompt_token_ids: sources.prompt_token_ids,
    }
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
fn should_give_every_streamed_tree_row_the_logits_its_chain_alone_gives() {
    assert_every_tree_row_matches_its_chain(&load_small_streamed_target());
}

#[test]
#[ignore = "loads the small dense Qwen3.5 on the streaming engine"]
#[serial]
fn should_leave_the_streamed_state_feeding_the_kept_branch_alone_leaves() {
    assert_a_commit_leaves_the_state_feeding_the_branch_leaves(&load_small_streamed_target());
}

#[test]
#[ignore = "loads the small dense Qwen3.5 on the streaming engine"]
#[serial]
fn should_differ_from_feeding_a_branch_the_streamed_commit_did_not_keep() {
    assert_a_commit_differs_from_feeding_another_branch(&load_small_streamed_target());
}
