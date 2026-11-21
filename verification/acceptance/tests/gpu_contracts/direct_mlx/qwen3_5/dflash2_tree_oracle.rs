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

use mlx_bridge::MlxArray;
use model_serving::speculative_decoding::dflash_drafter::{
    DFlashDraftTarget, DraftTree, DraftTreeOptions, DrafterContextStates, KeyedNoise,
    TENSORFOLD_TREE_BLOCK_ROW_COUNT, best_first_draft_tree, draft_lattice, drafted_hidden_rows,
    tree_block_row_count,
};
use model_serving::speculative_decoding::keyed_sampling::{RequestKeyedNoise, keyed_sample};
use model_serving::speculative_decoding::verification_decision::{
    TargetDistribution, VerificationDecision,
};
use model_serving::{PerformanceAttribution, RequestDecoderStateStack, VerificationLogitsMode};
use serial_test::serial;

use super::dflash2_journey_support::{host_values, tap_rows};
use super::dflash2_kept_per_round::{
    CHAT_PROMPT, CODE_PROMPT, Cell, ENGINE_CHILDREN_PER_NODE, ENGINE_NODE_BUDGET, JourneyPair,
    OUTPUT_TOKENS, RoundState, TOP_K, TOP_P_THOUSANDTHS, completion_token_ids, load_pair,
    prompt_token_ids, settle_round,
};

/// TensorFold's community benchmark seeds its repetitions from 1234.
pub(super) const FIRST_KEYED_SEED: u64 = 1234;

pub(super) const ORACLE_CELLS: [Cell; 5] = [
    oracle_cell("code completion greedy", true, CODE_PROMPT, 0),
    oracle_cell("code completion 0.7", true, CODE_PROMPT, 700),
    oracle_cell("code completion 1.0", true, CODE_PROMPT, 1_000),
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
        keyed_seed: FIRST_KEYED_SEED,
        node_budget: ENGINE_NODE_BUDGET,
        children_per_node: ENGINE_CHILDREN_PER_NODE,
    }
}

#[test]
#[ignore = "loads a real target and its DFlash2 drafter (the 27B pair unless configured); run through cargo measure-dflash2-tree-oracle"]
#[serial]
fn should_measure_how_many_tokens_a_draft_tree_keeps_per_round_on_recorded_references() {
    let pair = load_pair();
    for (cell_index, cell) in ORACLE_CELLS.iter().enumerate() {
        let reference = decode(&pair, cell, Drafting::SinglePath, &[]);
        eprintln!(
            "[dflash2-tree-oracle] {} reference text: {:?}",
            cell.label,
            reference_text(&pair, &reference.emitted_token_ids)
        );
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
    /// The tree against a text another engine wrote: the reference's tokens
    /// stand in for the target's picks, so the count is how many rounds this
    /// drafter and tree need for exactly that text.
    TreeOnForcedText,
}

pub(super) struct DecodeOutcome {
    pub(super) rounds: usize,
    /// The prefill's pick, then every token the rounds kept.
    pub(super) emitted_token_ids: Vec<u32>,
    /// Tokens each round kept, the target's own pick included.
    kept_by_round: Vec<usize>,
    reference_divergences: usize,
}

impl DecodeOutcome {
    fn kept_per_round(&self) -> f64 {
        (self.emitted_token_ids.len() - 1) as f64 / self.rounds as f64
    }
}

/// Decodes one cell. With `Drafting::Tree`, `reference` is the single-path
/// run's output, whose continuation each round verifies.
pub(super) fn decode(
    pair: &JourneyPair,
    cell: &Cell,
    drafting: Drafting,
    reference: &[u32],
) -> DecodeOutcome {
    let (mut round, mut decoder_state, first_token_id) = prefill(pair, cell);
    let first_token_id = if drafting == Drafting::TreeOnForcedText {
        reference[0]
    } else {
        first_token_id
    };
    round.last_token_id = first_token_id;
    let mut outcome = DecodeOutcome {
        rounds: 0,
        emitted_token_ids: vec![first_token_id],
        kept_by_round: Vec::new(),
        reference_divergences: 0,
    };
    let reach = block_row_count(pair, drafting, cell.node_budget) - 1;
    while outcome.emitted_token_ids.len() <= OUTPUT_TOKENS {
        let emitted_count = outcome.emitted_token_ids.len();
        if drafting != Drafting::SinglePath && emitted_count + reach + 1 > reference.len() {
            break;
        }
        let tree = draft_tree(pair, cell, &mut round, drafting);
        let continuation = match drafting {
            Drafting::SinglePath => tree.token_ids.clone(),
            _ => reference[emitted_count..emitted_count + reach].to_vec(),
        };
        let forced_picks = if drafting == Drafting::TreeOnForcedText {
            Some(&reference[emitted_count..=emitted_count + reach])
        } else {
            None
        };
        let kept = verify_round(
            pair,
            cell,
            (&mut round, &mut decoder_state),
            &tree,
            &continuation,
            forced_picks,
        );
        if drafting == Drafting::Tree
            && kept != reference[emitted_count..emitted_count + kept.len()]
        {
            outcome.reference_divergences += 1;
        }
        outcome.kept_by_round.push(kept.len());
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
    (round, decoder_state): (&mut RoundState, &mut RequestDecoderStateStack),
    tree: &DraftTree,
    continuation: &[u32],
    forced_picks: Option<&[u32]>,
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
    let picks = match forced_picks {
        Some(forced_picks) => forced_picks.to_vec(),
        None => picks,
    };
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
            keyed_sample(cell.keyed_seed, pick_positions[row_index], &distribution)
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
    let (hidden, draft_logits) = drafted_block(pair, round, drafting, cell.node_budget);
    let lattice = draft_lattice(
        pair.target.runtime(),
        pair.drafter_weights
            .candidate_selector
            .as_ref()
            .expect("a DFlash2 drafter has a candidate selector"),
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
    let (hidden, draft_logits) = drafted_block(pair, round, Drafting::Tree, cell.node_budget);
    runtime
        .evaluate_arrays(&[&hidden])
        .expect("the drafter evaluates");
    let drafter_milliseconds = started_at.elapsed().as_secs_f64() * 1000.0;
    let started_at = std::time::Instant::now();
    let lattice = draft_lattice(
        runtime,
        pair.drafter_weights
            .candidate_selector
            .as_ref()
            .expect("a DFlash2 drafter has a candidate selector"),
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

/// Rows the drafted block holds: the checkpoint's trained block, as the
/// engine serves, or TensorFold's 16 rows when its own text is replayed, so
/// the rounds are compared like for like.
fn block_row_count(pair: &JourneyPair, drafting: Drafting, node_budget: usize) -> usize {
    let longest_block_row_count = if drafting == Drafting::TreeOnForcedText {
        TENSORFOLD_TREE_BLOCK_ROW_COUNT
    } else {
        pair.drafter_configuration.block_size as usize
    };
    tree_block_row_count(node_budget, longest_block_row_count)
}

/// The drafter's hidden rows for one block and the target head's logits.
fn drafted_block(
    pair: &JourneyPair,
    round: &mut RoundState,
    drafting: Drafting,
    node_budget: usize,
) -> (MlxArray, MlxArray) {
    let runtime = pair.target.runtime();
    let configuration = &pair.drafter_configuration;
    let block_row_count = block_row_count(pair, drafting, node_budget);
    let mut block_ids = vec![configuration.mask_token_id; block_row_count];
    block_ids[0] = round.last_token_id;
    let block = runtime
        .array_from_u32(&block_ids, &[1, block_row_count as i32])
        .expect("the block builds");
    let target: &dyn DFlashDraftTarget = &pair.target;
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
    lattice: &model_serving::speculative_decoding::dflash_drafter::DraftLattice,
) -> DraftTree {
    let mut options = DraftTreeOptions {
        sampling_temperature: f64::from(cell.temperature_thousandths) / 1_000.0,
        node_budget: cell.node_budget,
        children_per_node: cell.children_per_node,
        ..DraftTreeOptions::default()
    };
    if drafting == Drafting::SinglePath {
        options.node_budget = pair.drafter_configuration.proposal_count() as usize;
        options.children_per_node = 1;
    }
    let keyed_noise = RequestKeyedNoise {
        seed: cell.keyed_seed,
    };
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

/// The decoded reference, to see what the cell writes and compare it with
/// another engine's output for the same prompt.
fn reference_text(pair: &JourneyPair, token_ids: &[u32]) -> String {
    let mut decoder = pair.tokenizer.incremental_decoder();
    let mut text = String::new();
    for token_id in token_ids {
        if let Ok(Some(piece)) = decoder.push_token(*token_id) {
            text.push_str(&piece);
        }
    }
    text
}

/// TensorFold's own greedy code completion (engine 1.2.0, this prompt, 256
/// tokens), which it drafted in 43 rounds: 5.95 kept a round.
const TENSORFOLD_GREEDY_CODE_COMPLETION: &str =
    include_str!("../../../fixtures/dflash_reference/tensorfold_greedy_code_completion.txt");

/// The two engines' greedy outputs part after about a hundred tokens, where
/// they round a near-tie differently, so their round counts are not on the
/// same text. Forcing TensorFold's text through this drafter and tree is.
#[test]
#[ignore = "loads a real target and its DFlash2 drafter (the 27B pair unless configured); run through cargo measure-dflash2-tree-oracle"]
#[serial]
fn should_count_the_rounds_this_tree_needs_for_tensorfolds_own_greedy_text() {
    let pair = load_pair();
    let cell = &ORACLE_CELLS[0];
    let reference = pair
        .tokenizer
        .encode_prompt(TENSORFOLD_GREEDY_CODE_COMPLETION)
        .expect("the reference text tokenizes");
    let forced = decode(&pair, cell, Drafting::TreeOnForcedText, &reference);
    eprintln!(
        "[dflash2-tree-oracle] TensorFold's greedy code text ({} tokens): {:.2} kept per round \
         over {} rounds here; TensorFold drafted it at 5.95",
        reference.len(),
        forced.kept_per_round(),
        forced.rounds,
    );
    eprintln!(
        "[dflash2-tree-oracle] kept by round: {:?}",
        forced.kept_by_round
    );
}
