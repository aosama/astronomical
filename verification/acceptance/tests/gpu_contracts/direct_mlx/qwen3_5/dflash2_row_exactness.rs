//! Whether a verified draft row gives the target's own one-token step's
//! logits on a real pair. TensorFold holds every (row, expert) pair to the
//! same bits in any window, so a drafted reply equals the plain one; a row
//! that drifts can flip a near-tied greedy pick, which is how the 35B-A3B
//! mixture-of-experts build first wrote "live in Verona" where plain
//! decoding wrote "live in fair Verona".
//!
//! The target's greedy continuation is fed three ways from the same prompt:
//! one token a step (the reference), as one 16-row chain through the tree
//! verify, and as one plain 16-row forward. Each row reports its largest
//! logit difference from the reference, whether its pick agrees, and the
//! reference's margin between its two best tokens, so a flip on a near tie
//! is told apart from drift on a clear pick. It reports; it asserts nothing.

use mlx_bridge::MlxArray;
use model_serving::DraftTreeTarget;
use model_serving::speculative_decoding::dflash_drafter::DraftTree;
use model_serving::speculative_decoding::tree_rows::TreeRows;
use serial_test::serial;

use super::dflash2_journey_support::{TargetLogitRows, forward_with_taps, host_values};
use super::dflash2_kept_per_round::{
    Cell, JourneyPair, load_pair, load_pair_verifying_trees_as_served,
};
use super::dflash2_tree_oracle::{Drafting, ORACLE_CELLS, draft_tree, prefill};

const COMPARED_ROW_COUNT: usize = 16;
/// The oracle's greedy chat cell: the prompt plain decoding was compared on.
const GREEDY_CHAT_CELL_INDEX: usize = 3;
/// About 10,000 prompt tokens, the serving journey's size.
const JOURNEY_PROMPT_WORD_COUNT: usize = 7_400;

#[test]
#[ignore = "loads a real target and its DFlash2 drafter; run through cargo measure-dflash2-row-exactness"]
#[serial]
fn should_report_how_far_verified_rows_drift_from_the_one_token_step() {
    let mut pair = load_pair();
    pair.target
        .enable_draft_tree_verification()
        .expect("the tree kernel declares");
    report_row_drift(
        &pair,
        &ORACLE_CELLS[GREEDY_CHAT_CELL_INDEX],
        COMPARED_ROW_COUNT,
        "16-row chain",
    );
}

/// The same comparison for the 4-row round a mixture-of-experts target
/// drafts, whose attention runs the row-exact tree kernel, after a short
/// prompt and after the serving journey's 10,000 tokens.
#[test]
#[ignore = "loads a real target and its DFlash2 drafter; run through cargo measure-dflash2-row-exactness"]
#[serial]
fn should_report_how_far_four_row_rounds_drift_after_a_short_and_a_journey_prompt() {
    let mut pair = load_pair();
    pair.target
        .enable_draft_tree_verification()
        .expect("the tree kernel declares");
    let journey_prompt: &'static str = Box::leak(
        include_str!("../../../fixtures/model_metrics_10000_romeo_and_juliet_words.txt")
            .split_whitespace()
            .take(JOURNEY_PROMPT_WORD_COUNT)
            .collect::<Vec<_>>()
            .join(" ")
            .into_boxed_str(),
    );
    let mut journey_cell = ORACLE_CELLS[GREEDY_CHAT_CELL_INDEX];
    journey_cell.prompt = journey_prompt;
    report_row_drift(
        &pair,
        &ORACLE_CELLS[GREEDY_CHAT_CELL_INDEX],
        4,
        "4-row chain, short prompt",
    );
    if labelled_prompts_the_target_reads(&pair, journey_prompt).len() > 1 {
        report_row_drift(&pair, &journey_cell, 4, "4-row chain, journey prompt");
    }
}

/// One real drafted tree after the serving journey's prompt, verified twice
/// from the same state: through the row-exact tree attention kernel and
/// through MLX's masked attention. The two attentions differ by rounding,
/// so the rows' logits should too; a row far apart means the kernel reads a
/// row's keys wrongly in the real model.
#[test]
#[ignore = "loads a real target and its DFlash2 drafter; run through cargo measure-dflash2-row-exactness"]
#[serial]
fn should_give_a_real_tree_the_same_logits_through_the_kernel_and_mlx_attention() {
    let mut pair = load_pair();
    let journey_prompt: &'static str = Box::leak(
        include_str!("../../../fixtures/model_metrics_10000_romeo_and_juliet_words.txt")
            .split_whitespace()
            .take(JOURNEY_PROMPT_WORD_COUNT)
            .collect::<Vec<_>>()
            .join(" ")
            .into_boxed_str(),
    );
    for (_, prompt) in labelled_prompts_the_target_reads(&pair, journey_prompt) {
        let mut cell = ORACLE_CELLS[4];
        cell.prompt = prompt;
        cell.node_budget = 3;
        let mut per_path_logits = Vec::new();
        for uses_kernel in [true, false] {
            pair.target
                .enable_draft_tree_verification()
                .expect("the tree kernels declare");
            if uses_kernel {
                pair.target
                    .enable_row_exact_tree_attention()
                    .expect("the tree attention kernel declares");
            } else {
                pair.target.set_row_exact_tree_attention_aside();
            }
            let (mut round, mut state, _) = prefill(&pair, &cell);
            let tree = draft_tree(&pair, &cell, &mut round, Drafting::Tree);
            let rows = TreeRows::from_draft_tree(round.last_token_id, &tree);
            let verified = pair
                .target
                .verify_draft_tree(
                    &rows,
                    round.fed_token_count,
                    &mut state,
                    &pair.drafter_configuration.target_tap_layer_ids,
                )
                .expect("the tree verifies");
            per_path_logits.push((
                split_rows(&pair, &verified.per_row_logits),
                tree.parent_indices.clone(),
            ));
        }
        let (kernel_rows, parents) = &per_path_logits[0];
        let (mlx_rows, _) = &per_path_logits[1];
        for (row_index, (kernel_row, mlx_row)) in kernel_rows.iter().zip(mlx_rows).enumerate() {
            eprintln!(
                "[dflash2-row-exactness] real tree ({} prompt words, parents {parents:?}) row {row_index}: kernel against MLX max diff {:.4}, pick {}",
                prompt.split_whitespace().count(),
                largest_difference(mlx_row, kernel_row),
                agreement(mlx_row, kernel_row),
            );
        }
    }
}

/// On real model data, how often each tree attention path's bf16 output
/// differs from float32 attention rounded to bf16: the row-exact kernel and
/// MLX's bf16 attention, which serving uses, on the same inputs in every
/// full-attention layer of one real drafted tree's verify.
#[test]
#[ignore = "loads a real target and its DFlash2 drafter; run through cargo measure-dflash2-row-exactness"]
#[serial]
fn should_count_where_tree_attention_paths_leave_float32_on_real_data() {
    let mut pair = load_pair_verifying_trees_as_served();
    pair.target
        .start_recording_tree_attention_errors_for_measurement();
    let journey_prompt: &'static str = Box::leak(
        include_str!("../../../fixtures/model_metrics_10000_romeo_and_juliet_words.txt")
            .split_whitespace()
            .take(JOURNEY_PROMPT_WORD_COUNT)
            .collect::<Vec<_>>()
            .join(" ")
            .into_boxed_str(),
    );
    for (label, prompt) in labelled_prompts_the_target_reads(&pair, journey_prompt) {
        let mut cell = ORACLE_CELLS[4];
        cell.prompt = prompt;
        cell.node_budget = 3;
        let (mut round, mut state, _) = prefill(&pair, &cell);
        let tree = draft_tree(&pair, &cell, &mut round, Drafting::Tree);
        let rows = TreeRows::from_draft_tree(round.last_token_id, &tree);
        let verified = pair
            .target
            .verify_draft_tree(
                &rows,
                round.fed_token_count,
                &mut state,
                &pair.drafter_configuration.target_tap_layer_ids,
            )
            .expect("the tree verifies");
        split_rows(&pair, &verified.per_row_logits);
        for (layer_order, count) in pair
            .target
            .take_tree_attention_errors_for_measurement()
            .iter()
            .enumerate()
        {
            eprintln!(
                "[dflash2-row-exactness] {label}, full-attention layer {layer_order}: of {} elements, kernel differs from float32 in {} (largest {:.5}; by row {:?}; by half {:?}; smaller {}), MLX bf16 in {} (largest {:.5})",
                count.element_count,
                count.kernel_differing_elements,
                count.kernel_largest_difference,
                count.kernel_differing_by_row,
                count.kernel_differing_by_half,
                count.kernel_smaller_elements,
                count.mlx_differing_elements,
                count.mlx_largest_difference,
            );
        }
    }
}

/// The short prompt, and the journey prompt for a target that can read it
/// here. These journeys read a prompt in one direct prefill under a 30 GB
/// limit; the dense 27B's 64 layers outgrow that on the journey prompt's
/// 10,000 tokens (it fails the same way before a tree is ever verified), so
/// a dense target is measured after the short prompt only.
fn labelled_prompts_the_target_reads(
    pair: &JourneyPair,
    journey_prompt: &'static str,
) -> Vec<(&'static str, &'static str)> {
    let mut labelled_prompts = vec![("short prompt", ORACLE_CELLS[4].prompt)];
    if pair.target.config.feed_forward_architecture()
        != model_serving::Qwen3_5FeedForwardArchitecture::Dense
    {
        labelled_prompts.push(("journey prompt", journey_prompt));
    }
    labelled_prompts
}

fn report_row_drift(pair: &JourneyPair, cell: &Cell, row_count: usize, label: &str) {
    let (reference_tokens, reference_rows) = one_token_steps(pair, cell, row_count);
    let tree_rows = chain_tree_rows(pair, cell, &reference_tokens);
    let forward_rows = plain_forward_rows(pair, cell, &reference_tokens);
    for row_index in 0..row_count {
        let reference = &reference_rows[row_index];
        eprintln!(
            "[dflash2-row-exactness] {label} row {row_index:>2}: top-two margin {:>7.3} | tree max diff {:>8.4} pick {} | plain forward max diff {:>8.4} pick {}",
            top_two_margin(reference),
            largest_difference(reference, &tree_rows[row_index]),
            agreement(reference, &tree_rows[row_index]),
            largest_difference(reference, &forward_rows[row_index]),
            agreement(reference, &forward_rows[row_index]),
        );
    }
}

/// The greedy continuation fed one token a step, with each step's logits.
/// Row `r` is the logits after feeding token `r`, where token 0 is the
/// prompt's own greedy pick.
fn one_token_steps(pair: &JourneyPair, cell: &Cell, row_count: usize) -> (Vec<u32>, Vec<Vec<f32>>) {
    let (round, mut state, first_token_id) = prefill(pair, cell);
    let mut fed_token_count = round.fed_token_count;
    let mut tokens = vec![first_token_id];
    let mut rows = Vec::with_capacity(row_count);
    for step_index in 0..row_count {
        let step = forward_with_taps(
            &pair.target,
            &tokens[step_index..=step_index],
            fed_token_count,
            &mut state,
            &pair.drafter_configuration.target_tap_layer_ids,
            TargetLogitRows::EveryRow,
        );
        let logits = step.logits;
        let row = host_values(pair.target.runtime(), &logits);
        tokens.push(highest_logit_token(&row));
        rows.push(row);
        fed_token_count += 1;
    }
    tokens.truncate(row_count);
    (tokens, rows)
}

/// The same tokens as one chain through the tree verify, split into rows.
fn chain_tree_rows(pair: &JourneyPair, cell: &Cell, tokens: &[u32]) -> Vec<Vec<f32>> {
    let verified = verify_chain(
        pair,
        cell,
        tokens,
        &pair.drafter_configuration.target_tap_layer_ids,
    );
    split_rows(pair, &verified.per_row_logits)
}

/// `tokens` as one chain through the tree verify after `cell`'s prompt:
/// every row's parent is the row before it.
pub(super) fn verify_chain(
    pair: &JourneyPair,
    cell: &Cell,
    tokens: &[u32],
    tap_layer_ids: &[i32],
) -> model_serving::Qwen3_5DraftTreeVerification {
    let (round, mut state, _) = prefill(pair, cell);
    let mut chain = DraftTree::default();
    for (row_index, token_id) in tokens.iter().enumerate().skip(1) {
        chain.token_ids.push(*token_id);
        chain.parent_indices.push((row_index - 1).checked_sub(1));
        chain.depths.push(row_index - 1);
    }
    let rows = TreeRows::from_draft_tree(tokens[0], &chain);
    pair.target
        .verify_draft_tree(&rows, round.fed_token_count, &mut state, tap_layer_ids)
        .expect("the chain tree verifies")
}

/// The same tokens as one plain 16-row forward, split into rows.
fn plain_forward_rows(pair: &JourneyPair, cell: &Cell, tokens: &[u32]) -> Vec<Vec<f32>> {
    let (round, mut state, _) = prefill(pair, cell);
    let forward = forward_with_taps(
        &pair.target,
        tokens,
        round.fed_token_count,
        &mut state,
        &pair.drafter_configuration.target_tap_layer_ids,
        TargetLogitRows::EveryRow,
    );
    split_rows(pair, &forward.logits)
}

fn split_rows(pair: &JourneyPair, logits: &MlxArray) -> Vec<Vec<f32>> {
    let vocabulary_size = *logits.shape().last().expect("logits have a last axis") as usize;
    host_values(pair.target.runtime(), logits)
        .chunks(vocabulary_size)
        .map(<[f32]>::to_vec)
        .collect()
}

fn highest_logit_token(row: &[f32]) -> u32 {
    let mut best_index = 0;
    for (token_index, logit) in row.iter().enumerate() {
        if *logit > row[best_index] {
            best_index = token_index;
        }
    }
    best_index as u32
}

fn top_two_margin(row: &[f32]) -> f32 {
    let (mut best, mut second) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
    for logit in row {
        if *logit > best {
            second = best;
            best = *logit;
        } else if *logit > second {
            second = *logit;
        }
    }
    best - second
}

fn largest_difference(reference: &[f32], candidate: &[f32]) -> f32 {
    let mut largest = 0.0_f32;
    for (reference_logit, candidate_logit) in reference.iter().zip(candidate) {
        largest = largest.max((reference_logit - candidate_logit).abs());
    }
    largest
}

fn agreement(reference: &[f32], candidate: &[f32]) -> &'static str {
    if highest_logit_token(reference) == highest_logit_token(candidate) {
        "same"
    } else {
        "FLIPPED"
    }
}
