//! Gate 4 end to end on the real 27B pair: every round drafts TensorFold's
//! tree, verifies all of it in one target pass, picks with the keyed sampler
//! (or greedily), and commits the kept chain. Prints kept per round and the
//! decode speed (prefill excluded) for TensorFold's code-completion and chat
//! cells, where its engine measured 78.7 and 44.0 tok/s on this Mac.
//!
//! Each stage still waits for the GPU (the lattice and the picks are read
//! back), so this is the speed a straightforward tree loop reaches, not a
//! pipelined serving loop's.

use std::time::Instant;

use astronomical_model_serving::RequestDecoderStateStack;
use astronomical_model_serving::speculative_decoding::tree_rows::TreeRows;
use mlx_bridge::MlxArray;
use serial_test::serial;

use astronomical_model_serving::speculative_decoding::context_copy::{
    ContextCopier, ContextCopySettings,
};

use super::dflash2_kept_per_round::{
    Cell, JourneyPair, OUTPUT_TOKENS, RoundState, completion_token_ids, load_pair, prompt_token_ids,
};
use super::dflash2_tree_oracle::{
    Drafting, FIRST_KEYED_SEED, ORACLE_CELLS, draft_tree, prefill, target_picks,
};

/// Seeds pooled for a sampled cell, as TensorFold's three repetitions.
const SAMPLED_SEED_COUNT: u64 = 3;
/// TensorFold's copied-row budget: it starts at 16 rows past the tree,
/// doubles while copies land whole, halves when most miss, and stops at 48.
const FIRST_COPY_ROWS: usize = 16;
const MAXIMUM_COPY_ROWS: usize = 48;

#[test]
#[ignore = "loads the real 27B and its DFlash2 drafter; run through cargo measure-dflash2-tree-rounds"]
#[serial]
fn should_decode_with_draft_trees_verified_in_one_target_pass() {
    let mut pair = load_pair();
    pair.target
        .enable_draft_tree_verification()
        .expect("the tree kernel declares");
    for (cell_index, cell) in ORACLE_CELLS.iter().enumerate() {
        let seed_count = if cell.temperature_thousandths == 0 {
            1
        } else {
            SAMPLED_SEED_COUNT
        };
        // Alternating the two in one process keeps drift out of the comparison.
        let mut pooled = [PooledTreeDecode::default(), PooledTreeDecode::default()];
        for seed_offset in 0..seed_count {
            let seeded_cell = Cell {
                keyed_seed: FIRST_KEYED_SEED + seed_offset,
                ..*cell
            };
            for (pooled_decode, copies_context) in pooled.iter_mut().zip([false, true]) {
                pooled_decode.add(&decode_with_trees(&pair, &seeded_cell, copies_context));
            }
        }
        for (pooled_decode, label) in pooled.iter().zip(["tree", "tree + context copies"]) {
            eprintln!(
                "[dflash2-tree-rounds {}/{}] {} ({seed_count} seeds), {label}: {:.2} kept per \
                 round, {:.1} tok/s decode ({} tokens in {} rounds, {:.1} ms a round)",
                cell_index + 1,
                ORACLE_CELLS.len(),
                cell.label,
                pooled_decode.emitted as f64 / pooled_decode.rounds as f64,
                pooled_decode.emitted as f64 / pooled_decode.decode_seconds,
                pooled_decode.emitted,
                pooled_decode.rounds,
                pooled_decode.decode_seconds * 1000.0 / pooled_decode.rounds as f64,
            );
        }
    }
}

/// Tree decodes pooled over seeds.
#[derive(Default)]
struct PooledTreeDecode {
    rounds: usize,
    emitted: usize,
    decode_seconds: f64,
}

impl PooledTreeDecode {
    fn add(&mut self, tree: &TreeDecode) {
        self.rounds += tree.rounds;
        self.emitted += tree.emitted_token_ids.len() - 1;
        self.decode_seconds += tree.decode_seconds;
    }
}

struct TreeDecode {
    rounds: usize,
    /// The prefill's pick, then every token the rounds kept.
    emitted_token_ids: Vec<u32>,
    decode_seconds: f64,
}

/// The request's text so far and its copier, when the round adds copies.
struct CopyContext {
    history: Vec<u32>,
    copier: ContextCopier,
    copy_row_budget: usize,
}

fn decode_with_trees(pair: &JourneyPair, cell: &Cell, copies_context: bool) -> TreeDecode {
    let (mut round, mut decoder_state, first_token_id) = prefill(pair, cell);
    let mut copy_context = if copies_context {
        let mut history = if cell.is_raw_completion {
            completion_token_ids(pair, cell.prompt)
        } else {
            prompt_token_ids(pair, cell.prompt)
        };
        history.push(first_token_id);
        Some(CopyContext {
            history,
            copier: ContextCopier::new(ContextCopySettings::default()),
            copy_row_budget: FIRST_COPY_ROWS,
        })
    } else {
        None
    };
    let decode_started_at = Instant::now();
    let mut emitted_token_ids = vec![first_token_id];
    let mut rounds = 0;
    while emitted_token_ids.len() <= OUTPUT_TOKENS {
        let emitted = tree_round(
            pair,
            cell,
            &mut round,
            &mut decoder_state,
            copy_context.as_mut(),
        );
        if let Some(copy_context) = copy_context.as_mut() {
            copy_context.history.extend_from_slice(&emitted);
        }
        emitted_token_ids.extend(emitted);
        rounds += 1;
    }
    TreeDecode {
        rounds,
        emitted_token_ids,
        decode_seconds: decode_started_at.elapsed().as_secs_f64(),
    }
}

/// Drafts a tree, hangs a context copy below it when given a copier,
/// verifies every row, keeps the chain the target's picks follow and
/// commits it. Returns the tokens the round emitted.
fn tree_round(
    pair: &JourneyPair,
    cell: &Cell,
    round: &mut RoundState,
    decoder_state: &mut RequestDecoderStateStack,
    mut copy_context: Option<&mut CopyContext>,
) -> Vec<u32> {
    let tree = draft_tree(pair, cell, round, Drafting::Tree);
    let mut rows = TreeRows::from_draft_tree(round.last_token_id, &tree);
    let first_copied_row = rows.row_count();
    let mut copied_row_count = 0;
    if let Some(copy_context) = copy_context.as_deref_mut() {
        let copied = copy_context.copier.propose(
            &copy_context.history,
            copy_context.copy_row_budget + tree.token_ids.len(),
        );
        copied_row_count = rows.add_copied_chain(&copied);
    }
    let verification = pair
        .target
        .verify_draft_tree(
            &rows,
            round.fed_token_count,
            decoder_state,
            &pair.drafter_configuration.target_tap_layer_ids,
        )
        .expect("the tree verifies");
    // A row's pick takes the position after the row's own.
    let mut pick_positions = Vec::with_capacity(rows.row_count());
    for row_position in rows.positions(round.fed_token_count as i32) {
        pick_positions.push(row_position as u64 + 1);
    }
    let picks = target_picks(pair, cell, &verification.per_row_logits, &pick_positions);
    let kept = rows.kept_chain(&picks);
    if let Some(copy_context) = copy_context {
        let mut landed = 0;
        for kept_row in &kept.rows {
            if *kept_row >= first_copied_row {
                landed += 1;
            }
        }
        copy_context.copier.observe(copied_row_count, landed);
        adapt_copy_row_budget(copy_context, copied_row_count, landed);
    }
    round.pending_tap_rows = kept_tap_rows(pair, &verification.hidden_state_taps, &kept.rows);
    pair.target
        .commit_draft_tree(decoder_state, verification, &kept.rows)
        .expect("the kept chain commits");
    let mut emitted = Vec::with_capacity(kept.rows.len());
    for kept_row in &kept.rows[1..] {
        emitted.push(rows.token_ids[*kept_row]);
    }
    emitted.push(kept.next_token_id);
    round.fed_token_count += kept.rows.len() as u32;
    round.last_token_id = kept.next_token_id;
    emitted
}

/// TensorFold's rule: double the copied rows while copies land whole, halve
/// them (not below the first budget) when fewer than half land.
fn adapt_copy_row_budget(copy_context: &mut CopyContext, copied_row_count: usize, landed: usize) {
    if copied_row_count == 0 {
        return;
    }
    if landed == copied_row_count {
        copy_context.copy_row_budget = (copy_context.copy_row_budget * 2).min(MAXIMUM_COPY_ROWS);
    } else if 2 * landed < copied_row_count {
        copy_context.copy_row_budget = (copy_context.copy_row_budget / 2).max(FIRST_COPY_ROWS);
    }
}

/// The drafter's next input: every tap's kept rows, in chain order, joined.
fn kept_tap_rows(pair: &JourneyPair, layer_taps: &[MlxArray], kept_rows: &[usize]) -> MlxArray {
    let runtime = pair.target.runtime();
    let mut kept_row_ids = Vec::with_capacity(kept_rows.len());
    for kept_row in kept_rows {
        kept_row_ids.push(*kept_row as i32);
    }
    let kept_row_array = runtime
        .array_from_i32(&kept_row_ids, &[kept_rows.len() as i32])
        .expect("the kept rows build");
    let mut kept_taps = Vec::with_capacity(layer_taps.len());
    for layer_tap in layer_taps {
        kept_taps.push(
            runtime
                .take_axis(layer_tap, &kept_row_array, 1)
                .expect("the kept tap rows gather"),
        );
    }
    let references: Vec<&MlxArray> = kept_taps.iter().collect();
    runtime
        .concatenate_axis(&references, -1)
        .expect("the taps join")
}

const BREAKDOWN_ROUND_COUNT: usize = 24;
const BREAKDOWN_WARM_UP_ROUNDS: usize = 4;
const BREAKDOWN_STAGES: [&str; 7] = [
    "drafter pass (4-bit, 8 rows)",
    "draft head + lattice readback",
    "tree search (CPU)",
    "tree verify graph build (CPU)",
    "tree verify GPU wait",
    "picks readback (keyed sample)",
    "commit kept chain",
];

#[test]
#[ignore = "loads the real 27B and its DFlash2 drafter; run through cargo measure-dflash2-tree-rounds"]
#[serial]
fn should_measure_where_a_tree_round_spends_its_time() {
    let mut pair = load_pair();
    pair.target
        .enable_draft_tree_verification()
        .expect("the tree kernel declares");
    let cell = &ORACLE_CELLS[4];
    let (mut round, mut decoder_state, _first_token_id) = prefill(&pair, cell);
    let mut stage_times: Vec<Vec<f64>> = vec![Vec::new(); BREAKDOWN_STAGES.len()];
    for round_index in 0..BREAKDOWN_ROUND_COUNT {
        let round_stage_times = timed_tree_round(&pair, cell, &mut round, &mut decoder_state);
        if round_index >= BREAKDOWN_WARM_UP_ROUNDS {
            for (stage_index, stage_time) in round_stage_times.iter().enumerate() {
                stage_times[stage_index].push(*stage_time);
            }
        }
    }
    let mut round_total = 0.0;
    for (label, times) in BREAKDOWN_STAGES.iter().zip(stage_times.iter_mut()) {
        times.sort_by(f64::total_cmp);
        let median = times[times.len() / 2];
        round_total += median;
        eprintln!("[dflash2-tree-round-cost] {label:<34} {median:7.1} ms");
    }
    eprintln!(
        "[dflash2-tree-round-cost] {:<34} {round_total:7.1} ms (chat 1.0, every stage synced)",
        "tree round total"
    );
}

/// One tree round with every stage evaluated on its own: milliseconds per
/// stage in `BREAKDOWN_STAGES` order.
fn timed_tree_round(
    pair: &JourneyPair,
    cell: &Cell,
    round: &mut RoundState,
    decoder_state: &mut RequestDecoderStateStack,
) -> Vec<f64> {
    let runtime = pair.target.runtime();
    let mut stage_times = Vec::with_capacity(BREAKDOWN_STAGES.len());
    let mut stage_started_at = Instant::now();
    let staged = super::dflash2_tree_oracle::staged_draft_tree(pair, cell, round);
    stage_times.push(staged.drafter_milliseconds);
    stage_times.push(staged.lattice_milliseconds);
    stage_times.push(staged.search_milliseconds);
    let rows = TreeRows::from_draft_tree(round.last_token_id, &staged.tree);
    stage_started_at = restart(&mut stage_times, stage_started_at, false);
    let verification = pair
        .target
        .verify_draft_tree(
            &rows,
            round.fed_token_count,
            decoder_state,
            &pair.drafter_configuration.target_tap_layer_ids,
        )
        .expect("the tree verifies");
    stage_started_at = restart(&mut stage_times, stage_started_at, true);
    let mut verified_arrays: Vec<&MlxArray> = verification.hidden_state_taps.iter().collect();
    verified_arrays.push(&verification.per_row_logits);
    runtime
        .evaluate_arrays(&verified_arrays)
        .expect("the verify evaluates");
    stage_started_at = restart(&mut stage_times, stage_started_at, true);
    let mut pick_positions = Vec::with_capacity(rows.row_count());
    for row_position in rows.positions(round.fed_token_count as i32) {
        pick_positions.push(row_position as u64 + 1);
    }
    let picks = target_picks(pair, cell, &verification.per_row_logits, &pick_positions);
    let kept = rows.kept_chain(&picks);
    stage_started_at = restart(&mut stage_times, stage_started_at, true);
    round.pending_tap_rows = kept_tap_rows(pair, &verification.hidden_state_taps, &kept.rows);
    pair.target
        .commit_draft_tree(decoder_state, verification, &kept.rows)
        .expect("the kept chain commits");
    evaluate_decoder_state(pair, decoder_state, &round.pending_tap_rows);
    restart(&mut stage_times, stage_started_at, true);
    round.fed_token_count += kept.rows.len() as u32;
    round.last_token_id = kept.next_token_id;
    stage_times
}

/// Records the time since `stage_started_at` when `should_record`, and
/// returns a fresh start.
fn restart(stage_times: &mut Vec<f64>, stage_started_at: Instant, should_record: bool) -> Instant {
    if should_record {
        stage_times.push(stage_started_at.elapsed().as_secs_f64() * 1000.0);
    }
    Instant::now()
}

/// Waits for every layer's state, so the commit's GPU work is timed.
fn evaluate_decoder_state(
    pair: &JourneyPair,
    decoder_state: &mut RequestDecoderStateStack,
    pending_tap_rows: &MlxArray,
) {
    use astronomical_model_serving::AstroDecoderStateCacheState;
    let mut state_arrays: Vec<MlxArray> = Vec::new();
    for layer_index in 0..decoder_state.layer_count() {
        match decoder_state.layer_mut(layer_index) {
            Some(AstroDecoderStateCacheState::Composite {
                convolution,
                recurrent,
            }) => {
                for array in [convolution.state(), recurrent.state()]
                    .into_iter()
                    .flatten()
                {
                    state_arrays.push(array.retain().expect("the state retains"));
                }
            }
            Some(AstroDecoderStateCacheState::AppendOnlyAttention { attention }) => {
                for array in [attention.keys_state(), attention.values_state()]
                    .into_iter()
                    .flatten()
                {
                    state_arrays.push(array.retain().expect("the state retains"));
                }
            }
            None => {}
        }
    }
    let mut references: Vec<&MlxArray> = state_arrays.iter().collect();
    references.push(pending_tap_rows);
    pair.target
        .runtime()
        .evaluate_arrays(&references)
        .expect("the committed state evaluates");
}

const CHAIN_COMPARISON_ROUNDS: usize = 12;

/// Whether the tree pass costs more than a plain forward over as many rows:
/// a chain-shaped 16-row tree against the plain 16-row forward, alternating
/// in one process so drift hits both alike, each from a fresh prefill.
#[test]
#[ignore = "loads the real 27B and its DFlash2 drafter; run through cargo measure-dflash2-tree-rounds"]
#[serial]
fn should_compare_a_chain_tree_verify_with_the_plain_forward_over_as_many_rows() {
    use astronomical_model_serving::speculative_decoding::dflash2_drafter::DraftTree;
    use astronomical_model_serving::{PerformanceAttribution, VerificationLogitsMode};
    let mut pair = load_pair();
    pair.target
        .enable_draft_tree_verification()
        .expect("the tree kernel declares");
    let cell = &ORACLE_CELLS[3];
    let mut row_tokens = Vec::with_capacity(16);
    let mut chain = DraftTree::default();
    for row in 0..16_usize {
        row_tokens.push(1_000 + row as u32);
        if row > 0 {
            chain.token_ids.push(1_000 + row as u32);
            chain.parent_indices.push((row - 1).checked_sub(1));
            chain.depths.push(row - 1);
        }
    }
    let rows = TreeRows::from_draft_tree(row_tokens[0], &chain);
    let (mut plain_times, mut tree_times) = (Vec::new(), Vec::new());
    for _ in 0..CHAIN_COMPARISON_ROUNDS {
        let (round, mut state, _) = prefill(&pair, cell);
        let started_at = Instant::now();
        let plain = pair
            .target
            .build_speculative_verification_forward(
                &row_tokens,
                round.fed_token_count,
                &mut state,
                &pair.drafter_configuration.target_tap_layer_ids,
                VerificationLogitsMode::AllPositions,
                &mut PerformanceAttribution::disabled(),
            )
            .expect("the plain forward runs");
        let mut plain_arrays: Vec<&MlxArray> = plain.hidden_state_taps.iter().collect();
        let plain_logits = plain.per_position_logits.expect("the rows have logits");
        plain_arrays.push(&plain_logits);
        pair.target
            .runtime()
            .evaluate_arrays(&plain_arrays)
            .expect("evaluates");
        plain_times.push(started_at.elapsed().as_secs_f64() * 1000.0);
        let (round, mut state, _) = prefill(&pair, cell);
        let started_at = Instant::now();
        let tree = pair
            .target
            .verify_draft_tree(
                &rows,
                round.fed_token_count,
                &mut state,
                &pair.drafter_configuration.target_tap_layer_ids,
            )
            .expect("the chain tree verifies");
        let mut tree_arrays: Vec<&MlxArray> = tree.hidden_state_taps.iter().collect();
        tree_arrays.push(&tree.per_row_logits);
        pair.target
            .runtime()
            .evaluate_arrays(&tree_arrays)
            .expect("evaluates");
        tree_times.push(started_at.elapsed().as_secs_f64() * 1000.0);
    }
    plain_times.sort_by(f64::total_cmp);
    tree_times.sort_by(f64::total_cmp);
    eprintln!(
        "[dflash2-tree-round-cost] 16 rows: plain forward {:.1} ms, chain tree {:.1} ms (medians of {CHAIN_COMPARISON_ROUNDS})",
        plain_times[plain_times.len() / 2],
        tree_times[tree_times.len() / 2],
    );
}

/// Where a plain 16-row forward spends its GPU time, stage by stage, from
/// the engine's own per-stage evaluation boundaries: the rungs between the
/// projection ladder (`cargo measure-k-row-verify-cost`) and a whole pass.
/// Every boundary is a GPU wait, so the stages sum to more than a fused pass.
#[test]
#[ignore = "loads the real 27B and its DFlash2 drafter; run through cargo measure-dflash2-tree-rounds"]
#[serial]
fn should_attribute_a_sixteen_row_forward_to_its_layer_stages() {
    use astronomical_model_serving::{
        PerformanceAttribution, PerformanceOperation, VerificationLogitsMode,
    };
    let pair = load_pair();
    let cell = &ORACLE_CELLS[3];
    let row_tokens: Vec<u32> = (1_000..1_016).collect();
    let stages = [
        ("DeltaNet input projections", PerformanceOperation::PrefillLinearAttentionProjectionsGraphicsProcessorCompletionWait),
        ("DeltaNet convolution + SiLU", PerformanceOperation::PrefillLinearAttentionConvolutionGraphicsProcessorCompletionWait),
        ("DeltaNet norms, decay, rates", PerformanceOperation::PrefillLinearAttentionNormalizationGraphicsProcessorCompletionWait),
        ("DeltaNet recurrence", PerformanceOperation::PrefillLinearAttentionRecurrenceGraphicsProcessorCompletionWait),
        ("DeltaNet gate + output projection", PerformanceOperation::PrefillLinearAttentionEpilogueGraphicsProcessorCompletionWait),
        ("DeltaNet layer remainder", PerformanceOperation::PrefillLinearAttentionGraphicsProcessorCompletionWait),
        ("full attention layers", PerformanceOperation::PrefillFullAttentionGraphicsProcessorCompletionWait),
        ("feed-forward (gate, up, down)", PerformanceOperation::PrefillFeedForwardGraphicsProcessorCompletionWait),
    ];
    let mut stage_totals = vec![Vec::new(); stages.len()];
    let mut pass_totals = Vec::new();
    for _ in 0..6 {
        let (round, mut state, _) = prefill(&pair, cell);
        let mut attribution = PerformanceAttribution::enabled();
        let started_at = Instant::now();
        let forward = pair
            .target
            .build_speculative_verification_forward(
                &row_tokens,
                round.fed_token_count,
                &mut state,
                &[],
                VerificationLogitsMode::AllPositions,
                &mut attribution,
            )
            .expect("the forward runs");
        let logits = forward.per_position_logits.expect("the rows have logits");
        pair.target
            .runtime()
            .evaluate_arrays(&[&logits])
            .expect("evaluates");
        pass_totals.push(started_at.elapsed().as_secs_f64() * 1000.0);
        for (stage_index, (_, operation)) in stages.iter().enumerate() {
            let milliseconds = match attribution.operation_measurement(*operation) {
                Some(measurement) => measurement.total_elapsed_nanoseconds() as f64 / 1e6,
                None => 0.0,
            };
            stage_totals[stage_index].push(milliseconds);
        }
    }
    for ((label, _), totals) in stages.iter().zip(stage_totals.iter_mut()) {
        totals.sort_by(f64::total_cmp);
        eprintln!(
            "[dflash2-layer-stages] {label:<36} {:7.1} ms",
            totals[totals.len() / 2]
        );
    }
    pass_totals.sort_by(f64::total_cmp);
    eprintln!(
        "[dflash2-layer-stages] {:<36} {:7.1} ms (16 rows, every stage synced)",
        "whole pass with the head",
        pass_totals[pass_totals.len() / 2]
    );
}
