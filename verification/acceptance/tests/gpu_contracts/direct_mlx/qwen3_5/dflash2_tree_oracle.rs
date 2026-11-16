//! Gate 4's first measurement: how many tokens a round would keep if it
//! verified TensorFold's draft tree instead of one path, before any tree
//! forward exists.
//!
//! Greedy output does not depend on drafting, so each cell first decodes its
//! greedy reference with the single-path loop. The oracle then replays it:
//! every round drafts a block, builds the tree from its lattice, and verifies
//! the reference's own continuation as an ordinary path; the target's picks
//! along it are the truth, and the round keeps as deep as that truth runs
//! through the tree, plus the target's next token. The kept count is exactly
//! what a tree verify of the same rows would give. TensorFold's own rounds
//! on this pair kept 5.95 (code completion, greedy) and 4.2 (chat, greedy).

use astronomical_model_serving::speculative_decoding::dflash2_drafter::{
    DFlash2DraftTarget, DraftTree, DraftTreeOptions, DrafterContextStates, best_first_draft_tree,
    draft_lattice, drafted_hidden_rows,
};
use astronomical_model_serving::speculative_decoding::verification_decision::VerificationDecision;
use astronomical_model_serving::{
    PerformanceAttribution, RequestDecoderStateStack, VerificationLogitsMode,
};
use serial_test::serial;

use super::dflash2_journey_support::{host_values, tap_rows};
use super::dflash2_kept_per_round::{
    CHAT_PROMPT, CODE_PROMPT, Cell, JourneyPair, OUTPUT_TOKENS, RoundState, completion_token_ids,
    load_pair, prompt_token_ids, run_cell, settle_round,
};

const ORACLE_CELLS: [Cell; 2] = [
    Cell {
        label: "code completion greedy",
        is_raw_completion: true,
        prompt: CODE_PROMPT,
        temperature_thousandths: 0,
        minimum_kept_per_round: 0.0,
    },
    Cell {
        label: "chat greedy",
        is_raw_completion: false,
        prompt: CHAT_PROMPT,
        temperature_thousandths: 0,
        minimum_kept_per_round: 0.0,
    },
];

#[test]
#[ignore = "loads the real 27B and its DFlash2 drafter; run through cargo measure-dflash2-tree-oracle"]
#[serial]
fn should_measure_how_many_tokens_a_draft_tree_keeps_per_round_on_greedy_references() {
    let pair = load_pair();
    for (cell_index, cell) in ORACLE_CELLS.iter().enumerate() {
        let reference = run_cell(&pair, cell, 0);
        let path_kept = reference.emitted_tokens as f64 / reference.rounds as f64;
        let oracle = replay_with_tree(&pair, cell, &reference.emitted_token_ids);
        eprintln!(
            "[dflash2-tree-oracle {}/{}] {}: path {path_kept:.2} kept per round, tree {:.2} \
             ({} tokens in {} rounds, {} reference divergences)",
            cell_index + 1,
            ORACLE_CELLS.len(),
            cell.label,
            oracle.kept_tokens as f64 / oracle.rounds as f64,
            oracle.kept_tokens,
            oracle.rounds,
            oracle.reference_divergences,
        );
    }
}

struct OracleOutcome {
    rounds: usize,
    kept_tokens: usize,
    /// Rows where the target's pick differed from the recorded reference;
    /// numerically identical runs give none.
    reference_divergences: usize,
}

fn replay_with_tree(pair: &JourneyPair, cell: &Cell, reference: &[u32]) -> OracleOutcome {
    let tap_layer_ids = &pair.drafter_configuration.target_tap_layer_ids;
    let prompt = if cell.is_raw_completion {
        completion_token_ids(pair, cell.prompt)
    } else {
        prompt_token_ids(pair, cell.prompt)
    };
    let mut decoder_state = RequestDecoderStateStack::empty_from_astro_decoder_state_cache_layout_with_full_attention_kv_state_growth_tokens(
        &pair.target.astro_decoder_state_cache_layout(),
        256,
    )
    .expect("the decoder state allocates");
    let prefill = pair
        .target
        .build_speculative_verification_forward(
            &prompt,
            0,
            &mut decoder_state,
            tap_layer_ids,
            VerificationLogitsMode::LastRowOnly,
            &mut PerformanceAttribution::disabled(),
        )
        .expect("the prompt prefills with taps");
    let mut round = RoundState {
        last_token_id: reference[0],
        fed_token_count: prompt.len() as u32,
        pending_tap_rows: tap_rows(
            pair.target.runtime(),
            &prefill.hidden_state_taps,
            prompt.len() as i32,
        ),
        context_states: DrafterContextStates::new(&pair.drafter_configuration, 0),
    };
    let mut outcome = OracleOutcome {
        rounds: 0,
        kept_tokens: 0,
        reference_divergences: 0,
    };
    let block_reach = pair.drafter_configuration.proposal_count() as usize;
    let mut reference_index = 0;
    while outcome.kept_tokens < OUTPUT_TOKENS && reference_index + block_reach < reference.len() {
        let kept = oracle_round(
            pair,
            &mut round,
            &mut decoder_state,
            &reference[reference_index + 1..=reference_index + block_reach],
            &mut outcome,
        );
        reference_index += kept;
        outcome.kept_tokens += kept;
        outcome.rounds += 1;
    }
    outcome
}

/// One replayed round: draft, build the tree, verify `continuation` (the
/// reference's next tokens after the anchor), keep what the tree reaches.
fn oracle_round(
    pair: &JourneyPair,
    round: &mut RoundState,
    decoder_state: &mut RequestDecoderStateStack,
    continuation: &[u32],
    outcome: &mut OracleOutcome,
) -> usize {
    let runtime = pair.target.runtime();
    let tree = draft_tree(pair, round);
    let mut verified_ids = vec![round.last_token_id];
    verified_ids.extend_from_slice(continuation);
    let verification = pair
        .target
        .verify_drafted_rows(
            &verified_ids,
            round.fed_token_count,
            decoder_state,
            &pair.drafter_configuration.target_tap_layer_ids,
            &mut PerformanceAttribution::disabled(),
        )
        .expect("the target verifies the continuation");
    let target_picks: Vec<u32> = host_values(
        runtime,
        &runtime
            .argmax_axis(&verification.per_position_logits, -1)
            .expect("argmax builds"),
    )
    .iter()
    .map(|value| *value as u32)
    .collect();
    let kept_draft_count = depth_reached(&tree, &target_picks, continuation, outcome);
    let decision = VerificationDecision {
        kept_draft_count,
        next_token_id: target_picks[kept_draft_count],
    };
    settle_round(pair, round, decoder_state, verification, decision);
    kept_draft_count + 1
}

fn draft_tree(pair: &JourneyPair, round: &mut RoundState) -> DraftTree {
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
    let target: &dyn DFlash2DraftTarget = &pair.target;
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
    let lattice = draft_lattice(
        runtime,
        &pair.drafter_weights.candidate_selector,
        configuration,
        &hidden,
        &draft_logits,
        round.last_token_id,
    )
    .expect("the lattice reads back");
    best_first_draft_tree(
        &lattice,
        u64::from(round.fed_token_count),
        &DraftTreeOptions::default(),
        None,
    )
}

/// How many slots deep the target's picks run through the tree from its
/// root. A pick that differs from the reference ends the walk too: the rows
/// after it were computed on the reference, not on that pick.
fn depth_reached(
    tree: &DraftTree,
    target_picks: &[u32],
    continuation: &[u32],
    outcome: &mut OracleOutcome,
) -> usize {
    let mut parent: Option<usize> = None;
    for (depth, target_pick) in target_picks.iter().take(continuation.len()).enumerate() {
        if *target_pick != continuation[depth] {
            outcome.reference_divergences += 1;
            return depth;
        }
        let mut matching_child = None;
        for (node_index, node_token) in tree.token_ids.iter().enumerate() {
            if tree.parent_indices[node_index] == parent && *node_token == *target_pick {
                matching_child = Some(node_index);
            }
        }
        match matching_child {
            Some(node_index) => parent = Some(node_index),
            None => return depth,
        }
    }
    continuation.len()
}
