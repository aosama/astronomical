//! What a drafted round needs from the model it drafts for.

use text_generation::execution_error::TextExecutionError;

use crate::qwen3_5_core::model::draft_tree_verification::{
    DraftTreeTarget, Qwen3_5DraftTreeVerification,
};
use crate::{PerformanceAttribution, RequestDecoderStateStack};
use engine_core::speculative_decoding::tree_rows::TreeRows;

/// The tree one round verifies.
pub struct RoundTree<'round> {
    pub tree_rows: &'round TreeRows,
    /// Where the tree's anchor sits: every token before it is in the state.
    pub anchor_position: u32,
    /// The layers whose hidden rows the drafter reads next round.
    pub tap_layer_ids: &'round [i32],
}

/// A model a session can decode in drafted rounds: a draft tree's target,
/// plus the two steps each engine does its own way.
pub trait DraftRoundTarget: DraftTreeTarget + Sized {
    /// Verifies one round's tree as this engine serves a forward. The
    /// resident model leaves the pass lazy for the round's first readback;
    /// the streaming model evaluates it, because its routes are resolved
    /// in that evaluation and a refused pass has to be run again.
    fn verify_round_tree(
        &self,
        round_tree: RoundTree<'_>,
        request_decoder_state: &mut RequestDecoderStateStack,
    ) -> Result<Qwen3_5DraftTreeVerification, TextExecutionError>;

    /// Declares the kernels a drafted round needs and converts what this
    /// engine converts for it: the count of projections converted, or
    /// `None` on a GPU that cannot run those kernels.
    fn prepare_for_draft_trees(
        &mut self,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<Option<usize>, TextExecutionError>;

    /// Frees `drafter_load_bytes` under the memory limit for a drafter
    /// about to load, and says whether this engine did. An engine that
    /// holds every weight cannot, and its drafter is refused when the room
    /// is not already there; one that keeps streamed weights only while
    /// there is room gives some of them back.
    fn make_room_for_a_drafter(&mut self, _drafter_load_bytes: u64) -> bool {
        false
    }
}
