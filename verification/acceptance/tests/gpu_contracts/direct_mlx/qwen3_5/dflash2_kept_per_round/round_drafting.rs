//! The drafting stages of one round: running the drafter over the masked block
//! that starts at the round's last token and walking one path through the
//! candidate selector, then reading the drafter's sampled distribution of that
//! path back to the host for the decision of how many drafted tokens to keep.

use mlx_bridge::MlxArray;
use model_serving::speculative_decoding::dflash_drafter::{
    DFlashDraftTarget, PathChoice, ProposedPath, drafted_hidden_rows, select_path,
};
use runtime_integration::MlxRuntime;

use super::super::dflash2_journey_support::host_values;
use super::{Cell, DrafterDistribution, JourneyPair, RoundState};

/// Runs the drafter over the masked block that starts at the round's last
/// token, then walks one path through the candidate selector.
pub(super) fn propose_path(
    pair: &JourneyPair,
    cell: &Cell,
    round: &mut RoundState,
    random_state: &mut MlxArray,
) -> ProposedPath {
    let runtime = pair.target.runtime();
    let configuration = &pair.drafter_configuration;
    let mut block_ids = vec![round.last_token_id];
    block_ids.extend(std::iter::repeat_n(
        configuration.mask_token_id,
        configuration.proposal_count() as usize,
    ));
    let block = runtime
        .array_from_u32(&block_ids, &[1, configuration.block_size])
        .expect("the block builds");
    let target: &dyn DFlashDraftTarget = &*pair.target;
    let hidden = drafted_hidden_rows(
        runtime,
        &pair.drafter_weights,
        configuration,
        target,
        &mut round.context_states,
        &block,
        &round.pending_tap_rows,
    )
    .expect("the drafter runs");
    let draft_logits = target
        .head_logits(runtime, &hidden)
        .expect("the head scores the block");
    let anchor = runtime
        .array_from_u32(&[round.last_token_id], &[1])
        .expect("the anchor builds");
    let mut path_choice = if cell.temperature_thousandths == 0 {
        PathChoice::Greedy
    } else {
        PathChoice::Sampled {
            temperature: f32::from(cell.temperature_thousandths) / 1_000.0,
            random_state,
        }
    };
    select_path(
        runtime,
        pair.drafter_weights
            .candidate_selector
            .as_ref()
            .expect("a DFlash2 drafter has a candidate selector"),
        configuration,
        &hidden,
        &draft_logits,
        &anchor,
        &mut path_choice,
    )
    .expect("the selector walks a path")
}

/// The drafter's sampled `q` for a path, read back to the host; a greedy path
/// carries none.
pub(super) fn drafter_distribution_of_path(
    runtime: &MlxRuntime,
    path: &ProposedPath,
) -> Option<DrafterDistribution> {
    match &path.candidate_probabilities {
        Some(probabilities) => Some(DrafterDistribution {
            candidate_ids: host_values(runtime, &path.candidate_ids)
                .iter()
                .map(|value| *value as u32)
                .collect(),
            candidate_probabilities: host_values(runtime, probabilities),
        }),
        None => None,
    }
}
