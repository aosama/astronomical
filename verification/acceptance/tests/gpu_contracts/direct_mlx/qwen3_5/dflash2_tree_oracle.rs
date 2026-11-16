//! Gate 4's first measurement: how many tokens a round would keep if it
//! verified TensorFold's draft tree instead of one path, before any tree
//! forward exists.
//!
//! Neither greedy nor keyed sampling depends on drafting: a keyed sample is
//! fixed by (seed, position, token), as greedy is by the logits. So each cell
//! first decodes its reference with a single drafted path, then the oracle
//! replays it: every round drafts a block, builds the tree from its lattice,
//! and verifies the reference's own continuation as an ordinary path. The
//! target's picks along it are the truth, and the round keeps as deep as
//! that truth runs through the tree, plus the target's next token: exactly
//! what a tree verify of the same rows keeps. TensorFold's rounds on this
//! pair kept 5.95 and 6.2 (code completion, greedy and 0.7) and 4.2 and 3.46
//! (chat, greedy and 1.0).

use astronomical_model_serving::speculative_decoding::dflash2_drafter::{
    DFlash2DraftTarget, DraftTree, DraftTreeOptions, DrafterContextStates, KeyedNoise,
    best_first_draft_tree, draft_lattice, drafted_hidden_rows,
};
use astronomical_model_serving::speculative_decoding::keyed_sampling::{
    RequestKeyedNoise, keyed_sample,
};
use astronomical_model_serving::speculative_decoding::verification_decision::{
    TargetDistribution, VerificationDecision,
};
use astronomical_model_serving::{
    PerformanceAttribution, RequestDecoderStateStack, VerificationLogitsMode,
};
use mlx_bridge::MlxArray;
use serial_test::serial;

use super::dflash2_journey_support::{host_values, tap_rows};
use super::dflash2_kept_per_round::{
    CHAT_PROMPT, CODE_PROMPT, Cell, JourneyPair, OUTPUT_TOKENS, RoundState, TOP_K,
    TOP_P_THOUSANDTHS, completion_token_ids, load_pair, prompt_token_ids, settle_round,
};

const KEYED_SEED: u64 = 1234;

pub(super) const ORACLE_CELLS: [Cell; 4] = [
    oracle_cell("code completion greedy", true, CODE_PROMPT, 0),
    oracle_cell("code completion 0.7", true, CODE_PROMPT, 700),
    oracle_cell("chat greedy", false, CHAT_PROMPT, 0),
    oracle_cell("chat 1.0", false, CHAT_PROMPT, 1_000),
];

const fn oracle_cell(
    label: &'static str,
    is_raw_completion: bool,
    prompt: &'static str,
    temperature_thousandths: u16,
) -> Cell {
    Cell {
        label,
        is_raw_completion,
        prompt,
        temperature_thousandths,
        minimum_kept_per_round: 0.0,
    }
}

#[test]
#[ignore = "loads the real 27B and its DFlash2 drafter; run through cargo measure-dflash2-tree-oracle"]
#[serial]
fn should_measure_how_many_tokens_a_draft_tree_keeps_per_round_on_recorded_references() {
    let pair = load_pair();
    for (cell_index, cell) in ORACLE_CELLS.iter().enumerate() {
        let reference = decode(&pair, cell, Drafting::SinglePath, &[]);
        let replay = decode(&pair, cell, Drafting::Tree, &reference.emitted_token_ids);
        eprintln!(
            "[dflash2-tree-oracle {}/{}] {}: path {:.2} kept per round, tree {:.2} \
             ({} tokens in {} rounds, {} reference divergences)",
            cell_index + 1,
            ORACLE_CELLS.len(),
            cell.label,
            reference.kept_per_round(),
            replay.kept_per_round(),
            replay.emitted_token_ids.len() - 1,
            replay.rounds,
            replay.reference_divergences,
        );
    }
}

/// How each round drafts: one chain verified as drafted, or the tree with the
/// recorded reference's continuation verified in its place.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Drafting {
    SinglePath,
    Tree,
}

struct DecodeOutcome {
    rounds: usize,
    /// The prefill's pick, then every token the rounds kept.
    emitted_token_ids: Vec<u32>,
    reference_divergences: usize,
}

impl DecodeOutcome {
    fn kept_per_round(&self) -> f64 {
        (self.emitted_token_ids.len() - 1) as f64 / self.rounds as f64
    }
}

/// Decodes one cell. With `Drafting::Tree`, `reference` is the single-path
/// run's output, whose continuation each round verifies.
fn decode(pair: &JourneyPair, cell: &Cell, drafting: Drafting, reference: &[u32]) -> DecodeOutcome {
    let (mut round, mut decoder_state, first_token_id) = prefill(pair, cell);
    round.last_token_id = first_token_id;
    let mut outcome = DecodeOutcome {
        rounds: 0,
        emitted_token_ids: vec![first_token_id],
        reference_divergences: 0,
    };
    let reach = pair.drafter_configuration.proposal_count() as usize;
    while outcome.emitted_token_ids.len() <= OUTPUT_TOKENS {
        let emitted_count = outcome.emitted_token_ids.len();
        if drafting == Drafting::Tree && emitted_count + reach > reference.len() {
            break;
        }
        let tree = draft_tree(pair, cell, &mut round, drafting);
        let continuation = match drafting {
            Drafting::SinglePath => tree.token_ids.clone(),
            Drafting::Tree => reference[emitted_count..emitted_count + reach].to_vec(),
        };
        let kept = verify_round(
            pair,
            cell,
            &mut round,
            &mut decoder_state,
            &tree,
            &continuation,
        );
        if drafting == Drafting::Tree
            && kept != reference[emitted_count..emitted_count + kept.len()]
        {
            outcome.reference_divergences += 1;
        }
        outcome.emitted_token_ids.extend(kept);
        outcome.rounds += 1;
    }
    outcome
}

/// Prefills the prompt: the drafter's starting state, the target's decoder
/// state, and the target's first pick.
pub(super) fn prefill(
    pair: &JourneyPair,
    cell: &Cell,
) -> (RoundState, RequestDecoderStateStack, u32) {
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
            &pair.drafter_configuration.target_tap_layer_ids,
            VerificationLogitsMode::LastRowOnly,
            &mut PerformanceAttribution::disabled(),
        )
        .expect("the prompt prefills with taps");
    let last_logits = prefill
        .per_position_logits
        .expect("the last row has logits");
    let first_token_id = target_picks(pair, cell, &last_logits, &[prompt.len() as u64])[0];
    let round = RoundState {
        last_token_id: first_token_id,
        fed_token_count: prompt.len() as u32,
        pending_tap_rows: tap_rows(
            pair.target.runtime(),
            &prefill.hidden_state_taps,
            prompt.len() as i32,
        ),
        context_states: DrafterContextStates::new(&pair.drafter_configuration, 0),
    };
    (round, decoder_state, first_token_id)
}

/// Verifies `continuation` after the anchor, keeps as deep as the target's
/// picks run through `tree`, and returns the kept tokens.
fn verify_round(
    pair: &JourneyPair,
    cell: &Cell,
    round: &mut RoundState,
    decoder_state: &mut RequestDecoderStateStack,
    tree: &DraftTree,
    continuation: &[u32],
) -> Vec<u32> {
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
    // Row `i` holds the anchor's position plus `i`; its pick takes the next.
    let mut pick_positions = Vec::with_capacity(verified_ids.len());
    for row_index in 0..verified_ids.len() as u64 {
        pick_positions.push(u64::from(round.fed_token_count) + 1 + row_index);
    }
    let picks = target_picks(
        pair,
        cell,
        &verification.per_position_logits,
        &pick_positions,
    );
    let kept_draft_count = depth_reached(tree, &picks, continuation);
    let mut kept = continuation[..kept_draft_count].to_vec();
    kept.push(picks[kept_draft_count]);
    let decision = VerificationDecision {
        kept_draft_count,
        next_token_id: picks[kept_draft_count],
    };
    settle_round(pair, round, decoder_state, verification, decision);
    kept
}

/// The target's pick from every row of `logits` [1, rows, vocabulary]: the
/// argmax for greedy cells, else the keyed sample at the position each
/// row's pick takes, `pick_positions[row]`.
pub(super) fn target_picks(
    pair: &JourneyPair,
    cell: &Cell,
    logits: &MlxArray,
    pick_positions: &[u64],
) -> Vec<u32> {
    let runtime = pair.target.runtime();
    if cell.temperature_thousandths == 0 {
        let argmax = runtime.argmax_axis(logits, -1).expect("argmax builds");
        return host_values(runtime, &argmax)
            .iter()
            .map(|value| *value as u32)
            .collect();
    }
    let (distribution_ids, distribution_probabilities) = pair
        .target
        .sampling_distribution_rows(
            logits,
            cell.temperature_thousandths,
            TOP_P_THOUSANDTHS,
            TOP_K,
        )
        .expect("the target's sampling law applies");
    let kept_ids: Vec<u32> = host_values(runtime, &distribution_ids)
        .iter()
        .map(|value| *value as u32)
        .collect();
    let kept_probabilities = host_values(runtime, &distribution_probabilities);
    let row_count = logits.shape()[1] as usize;
    let kept_per_row = kept_ids.len() / row_count;
    let mut picks = Vec::with_capacity(row_count);
    for row_index in 0..row_count {
        let row_range = row_index * kept_per_row..(row_index + 1) * kept_per_row;
        let distribution = TargetDistribution {
            token_ids: &kept_ids[row_range.clone()],
            probabilities: &kept_probabilities[row_range],
        };
        picks.push(
            keyed_sample(KEYED_SEED, pick_positions[row_index], &distribution)
                .expect("a row keeps at least one token"),
        );
    }
    picks
}

/// Drafts a block and searches its lattice: TensorFold's tree, or a single
/// chain (one child a node) as deep as the block. Sampled cells add the
/// request's keyed draws, as the target samples with them.
pub(super) fn draft_tree(
    pair: &JourneyPair,
    cell: &Cell,
    round: &mut RoundState,
    drafting: Drafting,
) -> DraftTree {
    let (hidden, draft_logits) = drafted_block(pair, round);
    let lattice = draft_lattice(
        pair.target.runtime(),
        &pair.drafter_weights.candidate_selector,
        &pair.drafter_configuration,
        &hidden,
        &draft_logits,
        round.last_token_id,
    )
    .expect("the lattice reads back");
    search_tree(pair, cell, round, drafting, &lattice)
}

/// A drafted tree with each stage's milliseconds, every stage synced.
pub(super) struct StagedDraftTree {
    pub(super) tree: DraftTree,
    pub(super) drafter_milliseconds: f64,
    pub(super) lattice_milliseconds: f64,
    pub(super) search_milliseconds: f64,
}

pub(super) fn staged_draft_tree(
    pair: &JourneyPair,
    cell: &Cell,
    round: &mut RoundState,
) -> StagedDraftTree {
    let runtime = pair.target.runtime();
    let started_at = std::time::Instant::now();
    let (hidden, draft_logits) = drafted_block(pair, round);
    runtime
        .evaluate_arrays(&[&hidden])
        .expect("the drafter evaluates");
    let drafter_milliseconds = started_at.elapsed().as_secs_f64() * 1000.0;
    let started_at = std::time::Instant::now();
    let lattice = draft_lattice(
        runtime,
        &pair.drafter_weights.candidate_selector,
        &pair.drafter_configuration,
        &hidden,
        &draft_logits,
        round.last_token_id,
    )
    .expect("the lattice reads back");
    let lattice_milliseconds = started_at.elapsed().as_secs_f64() * 1000.0;
    let started_at = std::time::Instant::now();
    let tree = search_tree(pair, cell, round, Drafting::Tree, &lattice);
    StagedDraftTree {
        tree,
        drafter_milliseconds,
        lattice_milliseconds,
        search_milliseconds: started_at.elapsed().as_secs_f64() * 1000.0,
    }
}

/// The drafter's hidden rows for one block and the target head's logits.
fn drafted_block(pair: &JourneyPair, round: &mut RoundState) -> (MlxArray, MlxArray) {
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
    (hidden, draft_logits)
}

/// TensorFold's tree over `lattice`, or a single chain (one child a node) as
/// deep as the block. Sampled cells add the request's keyed draws, as the
/// target samples with them.
fn search_tree(
    pair: &JourneyPair,
    cell: &Cell,
    round: &RoundState,
    drafting: Drafting,
    lattice: &astronomical_model_serving::speculative_decoding::dflash2_drafter::DraftLattice,
) -> DraftTree {
    let mut options = DraftTreeOptions {
        sampling_temperature: f64::from(cell.temperature_thousandths) / 1_000.0,
        ..DraftTreeOptions::default()
    };
    if drafting == Drafting::SinglePath {
        options.node_budget = pair.drafter_configuration.proposal_count() as usize;
        options.children_per_node = 1;
    }
    let keyed_noise = RequestKeyedNoise { seed: KEYED_SEED };
    let noise: Option<&dyn KeyedNoise> = if cell.temperature_thousandths == 0 {
        None
    } else {
        Some(&keyed_noise)
    };
    best_first_draft_tree(lattice, u64::from(round.fed_token_count), &options, noise)
}

/// How many slots deep the target's picks run through the tree from its
/// root, never past the verified `continuation`: rows after a pick that
/// differs from it were computed on the continuation, not on that pick.
fn depth_reached(tree: &DraftTree, target_picks: &[u32], continuation: &[u32]) -> usize {
    let mut parent: Option<usize> = None;
    for (depth, target_pick) in target_picks.iter().take(continuation.len()).enumerate() {
        if *target_pick != continuation[depth] {
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
