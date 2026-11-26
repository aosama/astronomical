//! Where a draft-tree round's time goes on a real target and its DFlash2
//! drafter (the 27B pair unless the test run configures another target):
//! each stage synced on its own, after a short and a long prompt, against a
//! plain forward over as many rows, and by layer stage.
//!
//! These explain the decode speeds `dflash2_tree_rounds` reports; they
//! assert nothing.

use model_serving::DraftTreeTarget;
use std::time::Instant;

use mlx_bridge::MlxArray;
use model_serving::RequestDecoderStateStack;
use model_serving::speculative_decoding::tree_rows::TreeRows;
use serial_test::serial;

use super::dflash2_journey_support::{
    TargetLogitRows, forward_with_taps, story_cell, story_prompt,
};
use super::dflash2_kept_per_round::{Cell, JourneyPair, RoundState, load_pair};
use super::dflash2_tree_oracle::{ORACLE_CELLS, prefill, target_picks};
use super::dflash2_tree_rounds::kept_tap_rows;

const BREAKDOWN_ROUND_COUNT: usize = 24;
const BREAKDOWN_WARM_UP_ROUNDS: usize = 4;
const BREAKDOWN_STAGES: [&str; 7] = [
    "drafter pass (4-bit, 16 rows)",
    "draft head + lattice readback",
    "tree search (CPU)",
    "tree verify graph build (CPU)",
    "tree verify GPU wait",
    "picks readback (keyed sample)",
    "commit kept chain",
];

#[test]
#[ignore = "loads a real target and its DFlash2 drafter (the 27B pair unless configured); run through cargo measure-dflash2-tree-rounds"]
#[serial]
fn should_measure_where_a_tree_round_spends_its_time() {
    let mut pair = load_pair();
    pair.target
        .enable_draft_tree_verification()
        .expect("the tree kernel declares");
    print_stage_breakdown(&pair, &story_cell("story 1.0", 1_000), "story 1.0");
}

/// The Romeo and Juliet words a long prompt is cut from.
const LONG_PROMPT_SOURCE: &str =
    include_str!("../../../fixtures/model_metrics_10000_romeo_and_juliet_words.txt");
/// About 6,000 tokens: long enough that anything a round copies or reads
/// per context row shows, short enough to prefill in one pass.
const LONG_PROMPT_WORD_COUNT: usize = 4_000;

/// A round's stages with a long prompt already in the target's state. A
/// round's cost should not grow with the context beyond the attention over
/// it; the 27B served a 10,011-token prompt at 35.7 tok/s against 46 on a
/// short one, and this is where the difference shows.
#[test]
#[ignore = "loads a real target and its DFlash2 drafter (the 27B pair unless configured); run through cargo measure-dflash2-tree-rounds"]
#[serial]
fn should_measure_where_a_tree_round_spends_its_time_after_a_long_prompt() {
    let mut pair = load_pair();
    pair.target
        .enable_draft_tree_verification()
        .expect("the tree kernel declares");
    let long_prompt: &'static str = Box::leak(
        LONG_PROMPT_SOURCE
            .split_whitespace()
            .take(LONG_PROMPT_WORD_COUNT)
            .collect::<Vec<_>>()
            .join(" ")
            .into_boxed_str(),
    );
    let mut cell = ORACLE_CELLS[2];
    cell.prompt = long_prompt;
    print_stage_breakdown(&pair, &cell, "long raw prompt, 1.0");
}

/// The same breakdown after the story prompt (about 2,000 tokens on a dense
/// target, 10,000 on a mixture-of-experts one),
/// with the 4-row tree a mixture-of-experts target drafts with, so a
/// round's growth with the prompt is placed on its stage.
#[test]
#[ignore = "loads a real target and its DFlash2 drafter; run through cargo measure-dflash2-tree-rounds"]
#[serial]
fn should_measure_where_a_small_tree_round_spends_its_time_after_a_journey_prompt() {
    let mut pair = load_pair();
    pair.target
        .enable_draft_tree_verification()
        .expect("the tree kernel declares");
    let journey_prompt = story_prompt();
    let mut prompts = vec![("short prompt".to_owned(), ORACLE_CELLS[2].prompt)];
    for word_count in [750, 2_200, LONG_PROMPT_WORD_COUNT] {
        let prompt: &'static str = Box::leak(
            LONG_PROMPT_SOURCE
                .split_whitespace()
                .take(word_count)
                .collect::<Vec<_>>()
                .join(" ")
                .into_boxed_str(),
        );
        prompts.push((format!("{word_count}-word prompt"), prompt));
    }
    prompts.push(("story prompt".to_owned(), journey_prompt));
    // Each context through both tree attention paths: where the row-exact
    // kernel's round starts costing less than serving's MLX attention.
    for (prompt_label, prompt) in prompts {
        for uses_kernel in [true, false] {
            if uses_kernel {
                pair.target
                    .enable_row_exact_tree_attention()
                    .expect("the tree attention kernel declares");
            } else {
                pair.target.set_row_exact_tree_attention_aside();
            }
            let mut cell = ORACLE_CELLS[2];
            cell.prompt = prompt;
            cell.node_budget = 3;
            let path_label = if uses_kernel {
                "row-exact kernel"
            } else {
                "MLX attention"
            };
            print_stage_breakdown(
                &pair,
                &cell,
                &format!("{prompt_label}, 3-node tree, {path_label}"),
            );
        }
    }
}

/// Whether a verify's cost after a long prompt follows the rows it attends
/// with (attention work) or only the context (a copy of the stored keys and
/// values): one row and sixteen rows, after a short and a long prompt. Each
/// verify keeps only its anchor, so the context grows a token a round.
#[test]
#[ignore = "loads a real target and its DFlash2 drafter (the 27B pair unless configured); run through cargo measure-dflash2-tree-rounds"]
#[serial]
fn should_separate_a_long_contexts_attention_work_from_copies_of_its_keys() {
    use model_serving::speculative_decoding::dflash_drafter::DraftTree;
    let mut pair = load_pair();
    pair.target
        .enable_draft_tree_verification()
        .expect("the tree kernel declares");
    let long_prompt: &'static str = Box::leak(
        LONG_PROMPT_SOURCE
            .split_whitespace()
            .take(LONG_PROMPT_WORD_COUNT)
            .collect::<Vec<_>>()
            .join(" ")
            .into_boxed_str(),
    );
    let mut long_cell = ORACLE_CELLS[2];
    long_cell.prompt = long_prompt;
    let journey_prompt = story_prompt();
    let mut journey_cell = ORACLE_CELLS[2];
    journey_cell.prompt = journey_prompt;
    for (context_label, cell) in [
        ("short prompt", ORACLE_CELLS[2]),
        ("long prompt", long_cell),
        ("story prompt", journey_cell),
    ] {
        // 4 rows is the round a mixture-of-experts target verifies.
        for row_count in [1_usize, 4, 16] {
            let mut chain = DraftTree::default();
            for row in 1..row_count {
                chain.token_ids.push(1_000 + row as u32);
                chain.parent_indices.push((row - 1).checked_sub(1));
                chain.depths.push(row - 1);
            }
            let rows = TreeRows::from_draft_tree(1_000, &chain);
            let (mut round, mut state, _) = prefill(&pair, &cell);
            let mut verify_times = Vec::new();
            for _ in 0..10 {
                let started_at = Instant::now();
                let verification = pair
                    .target
                    .verify_draft_tree(
                        &rows,
                        round.fed_token_count,
                        &mut state,
                        &pair.drafter_configuration.target_tap_layer_ids,
                    )
                    .expect("the tree verifies");
                pair.target
                    .runtime()
                    .evaluate_arrays(&[&verification.per_row_logits])
                    .expect("the verify evaluates");
                verify_times.push(started_at.elapsed().as_secs_f64() * 1000.0);
                round.pending_tap_rows =
                    kept_tap_rows(&pair, &verification.hidden_state_taps, &[0]);
                pair.target
                    .commit_draft_tree(&mut state, verification, &[0])
                    .expect("the anchor commits");
                evaluate_decoder_state(&pair, &mut state, &round.pending_tap_rows);
                round.fed_token_count += 1;
            }
            verify_times.sort_by(f64::total_cmp);
            eprintln!(
                "[dflash2-context-cost] {context_label}, {row_count:>2} rows: verify {:6.1} ms",
                verify_times[verify_times.len() / 2]
            );
        }
    }
}

/// Runs tree rounds on `cell` with every stage synced and prints each
/// stage's median milliseconds.
fn print_stage_breakdown(pair: &JourneyPair, cell: &Cell, label: &str) {
    let (mut round, mut decoder_state, _first_token_id) = prefill(pair, cell);
    eprintln!(
        "[dflash2-tree-round-cost] {label}: {} prompt tokens",
        round.fed_token_count
    );
    let mut stage_times: Vec<Vec<f64>> = vec![Vec::new(); BREAKDOWN_STAGES.len()];
    for round_index in 0..BREAKDOWN_ROUND_COUNT {
        let round_stage_times = timed_tree_round(pair, cell, &mut round, &mut decoder_state);
        if round_index >= BREAKDOWN_WARM_UP_ROUNDS {
            for (stage_index, stage_time) in round_stage_times.iter().enumerate() {
                stage_times[stage_index].push(*stage_time);
            }
        }
    }
    let mut round_total = 0.0;
    for (stage_label, times) in BREAKDOWN_STAGES.iter().zip(stage_times.iter_mut()) {
        times.sort_by(f64::total_cmp);
        let median = times[times.len() / 2];
        round_total += median;
        eprintln!("[dflash2-tree-round-cost] {stage_label:<34} {median:7.1} ms");
    }
    eprintln!(
        "[dflash2-tree-round-cost] {:<34} {round_total:7.1} ms ({label}, every stage synced)",
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
    use model_serving::AstroDecoderStateCacheState;
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
#[ignore = "loads a real target and its DFlash2 drafter (the 27B pair unless configured); run through cargo measure-dflash2-tree-rounds"]
#[serial]
fn should_compare_a_chain_tree_verify_with_the_plain_forward_over_as_many_rows() {
    use model_serving::speculative_decoding::dflash_drafter::DraftTree;

    let mut pair = load_pair();
    pair.target
        .enable_draft_tree_verification()
        .expect("the tree kernel declares");
    let cell = &story_cell("story greedy", 0);
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
        let plain = forward_with_taps(
            &pair.target,
            &row_tokens,
            round.fed_token_count,
            &mut state,
            &pair.drafter_configuration.target_tap_layer_ids,
            TargetLogitRows::EveryRow,
        );
        let mut plain_arrays: Vec<&MlxArray> = plain.hidden_state_taps.iter().collect();
        let plain_logits = plain.logits;
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
#[ignore = "loads a real target and its DFlash2 drafter (the 27B pair unless configured); run through cargo measure-dflash2-tree-rounds"]
#[serial]
fn should_attribute_a_sixteen_row_forward_to_its_layer_stages() {
    use model_serving::{PerformanceAttribution, PerformanceOperation};
    let pair = load_pair();
    let cell = &story_cell("story greedy", 0);
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
    // Two rows beside sixteen: a stage's difference is what fourteen rows
    // cost it, free of the wait every stage boundary adds to both. (One row
    // takes the decode path, which has no stage boundaries.)
    let mut medians_by_row_count = Vec::new();
    for row_count in [2_usize, 16] {
        let mut stage_totals = vec![Vec::new(); stages.len() + 1];
        for _ in 0..6 {
            let (round, mut state, _) = prefill(&pair, cell);
            let mut attribution = PerformanceAttribution::enabled();
            let started_at = Instant::now();
            let logits = pair
                .target
                .forward_chunk_with_performance_attribution(
                    &row_tokens[..row_count],
                    round.fed_token_count,
                    &mut state,
                    &mut attribution,
                )
                .expect("the rows forward");
            pair.target
                .runtime()
                .evaluate_arrays(&[&logits])
                .expect("evaluates");
            stage_totals[stages.len()].push(started_at.elapsed().as_secs_f64() * 1000.0);
            for (stage_index, (_, operation)) in stages.iter().enumerate() {
                let milliseconds = match attribution.operation_measurement(*operation) {
                    Some(measurement) => measurement.total_elapsed_nanoseconds() as f64 / 1e6,
                    None => 0.0,
                };
                stage_totals[stage_index].push(milliseconds);
            }
        }
        let mut medians = Vec::with_capacity(stage_totals.len());
        for totals in &mut stage_totals {
            totals.sort_by(f64::total_cmp);
            medians.push(totals[totals.len() / 2]);
        }
        medians_by_row_count.push(medians);
    }
    let labels = stages
        .iter()
        .map(|(label, _)| *label)
        .chain(["whole pass with the last row's head"]);
    for (stage_index, label) in labels.enumerate() {
        let (two_rows, sixteen_rows) = (
            medians_by_row_count[0][stage_index],
            medians_by_row_count[1][stage_index],
        );
        eprintln!(
            "[dflash2-layer-stages] {label:<36} 2 rows {two_rows:6.1} ms, 16 rows \
             {sixteen_rows:6.1} ms, 14 rows cost {:5.1} ms",
            sixteen_rows - two_rows
        );
    }
}

/// A tree round drafts a block twice as long as the checkpoint trained on.
/// The trained rows must come out exactly as the trained block gives them,
/// after a short prompt and after one longer than the drafter's window:
/// only then is the longer block a pure addition.
#[test]
#[ignore = "loads a real target and its DFlash2 drafter (the 27B pair unless configured); run through cargo measure-dflash2-tree-rounds"]
#[serial]
fn should_leave_the_trained_rows_unchanged_when_the_block_is_longer() {
    use model_serving::speculative_decoding::dflash_drafter::{
        DFlashDraftTarget, DrafterContextStates, drafted_hidden_rows,
    };
    let pair = load_pair();
    let runtime = pair.target.runtime();
    let configuration = &pair.drafter_configuration;
    let long_prompt: &'static str = Box::leak(
        LONG_PROMPT_SOURCE
            .split_whitespace()
            .take(2_000)
            .collect::<Vec<_>>()
            .join(" ")
            .into_boxed_str(),
    );
    let mut long_cell = ORACLE_CELLS[2];
    long_cell.prompt = long_prompt;
    for (label, cell) in [
        ("short prompt", ORACLE_CELLS[2]),
        ("long prompt", long_cell),
    ] {
        let (round, _state, first_token_id) = prefill(&pair, &cell);
        let trained_row_count = configuration.block_size as usize;
        let mut trained_rows_by_block = Vec::new();
        for block_row_count in [trained_row_count, 2 * trained_row_count] {
            let mut block_ids = vec![configuration.mask_token_id; block_row_count];
            block_ids[0] = first_token_id;
            let block = runtime
                .array_from_u32(&block_ids, &[1, block_row_count as i32])
                .expect("the block builds");
            let mut context_states = DrafterContextStates::new(configuration, 0);
            let target: &dyn DFlashDraftTarget = &*pair.target;
            let hidden_rows = drafted_hidden_rows(
                runtime,
                &pair.drafter_weights,
                configuration,
                target,
                &mut context_states,
                &block,
                &round.pending_tap_rows,
            )
            .expect("the drafter runs");
            let trained_rows = runtime
                .slice(
                    &hidden_rows,
                    &[0, 0, 0],
                    &[1, trained_row_count as i32 - 1, configuration.hidden_size],
                    &[1, 1, 1],
                )
                .expect("the trained rows slice");
            trained_rows_by_block.push(
                runtime
                    .array_to_vec_f32(
                        &runtime
                            .astype(&trained_rows, mlx_bridge::MlxDtype::Float32)
                            .expect("widen"),
                    )
                    .expect("the rows read back"),
            );
        }
        // The attention and convolution kernels round differently for a
        // different row count, so equal means a difference far below the
        // rows' own size: root mean square against root mean square.
        let (mut squared_values, mut squared_differences) = (0.0_f64, 0.0_f64);
        for (trained, longer) in trained_rows_by_block[0]
            .iter()
            .zip(trained_rows_by_block[1].iter())
        {
            squared_values += f64::from(*trained).powi(2);
            squared_differences += f64::from(trained - longer).powi(2);
        }
        let relative_difference = (squared_differences / squared_values).sqrt();
        eprintln!(
            "[dflash2-tree-block] {label} ({} context rows): the trained rows differ by \
             {relative_difference:.4} of their size",
            round.fed_token_count
        );
        assert!(
            relative_difference <= 0.02,
            "{label}: the trained rows changed when the block grew"
        );
    }
}
