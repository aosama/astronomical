//! Ground truth for draft trees on the streaming model: a tree's rows must
//! give the target's own one-token picks. A round's time is printed beside a
//! plain step's, but the model here runs without the serving engine's expert
//! residency plan, so both are far slower than serving and only their ratio
//! means anything; speed is measured through the real worker.
//!
//! The streaming model finds a layer's experts through the
//! `astro_expert_cache` and its pager, so a model that does not fit in memory
//! can still verify a draft tree. Before any serving code leans on that, this
//! compares the greedy continuation fed one token a step with the same tokens
//! verified four rows a pass as a chain, on the configured target (the test
//! run's `model_directory`, under its `paging_experiment` memory limit).
//!
//! A second measurement puts the model in the memory state the engine serves
//! in and reads where a round's time goes beside a plain step's: the graph
//! build, the evaluation, how many layers stopped the pass to bring their
//! route to the host, how long those stops took, and what was read from
//! storage. It asserts nothing.

use std::time::Instant;

use astronomical_model_serving::speculative_decoding::dflash2_drafter::DraftTree;
use astronomical_model_serving::speculative_decoding::tree_rows::TreeRows;
use astronomical_model_serving::{
    ExpertDispatchCounts, MemoryPhase, PerformanceAttribution, Qwen3_5ArtifactValidator,
    Qwen3_5Config, Qwen3_5StreamingModel, Qwen3_5Tokenizer, RequestDecoderStateStack,
};
use astronomical_runtime_integration::{MlxMemoryLimits, MlxRuntime};
use serial_test::serial;

/// Rows a round verifies: the serving tree's four (the anchor and 3 nodes).
const ROWS_PER_ROUND: usize = 4;
const ROUND_COUNT: usize = 8;
const MLX_BUFFER_CACHE_LIMIT_BYTES: usize = 2 * 1024 * 1024 * 1024;
const PROMPT: &str = "Two households, both alike in dignity, in fair Verona, where we lay our scene, \
from ancient grudge break to new mutiny, where civil blood makes civil hands unclean. \
From forth the fatal loins of these two foes a pair of star-crossed lovers take their life.";

struct StreamingTarget {
    model: Qwen3_5StreamingModel,
    config: Qwen3_5Config,
    prompt_token_ids: Vec<u32>,
}

fn load_streaming_target(prompt: &str) -> StreamingTarget {
    let test_run = astronomical_test_harness::test_run::TestRunContext::current();
    let target_directory = test_run.required_model_directory();
    let memory_limit_bytes = test_run
        .configuration
        .paging_experiment
        .maximum_mlx_memory_bytes()
        .expect("the configured memory limit is valid");
    let runtime = MlxRuntime::initialize(
        MlxMemoryLimits::new(
            usize::try_from(memory_limit_bytes).expect("the limit fits this Mac's address space"),
            MLX_BUFFER_CACHE_LIMIT_BYTES,
        )
        .expect("memory limits are valid"),
    )
    .expect("the MLX runtime initializes");
    let artifact = Qwen3_5ArtifactValidator::new()
        .validate(&target_directory, 4096)
        .expect("the target artifact validates");
    let config = artifact.config().clone();
    let tokenizer =
        Qwen3_5Tokenizer::from_validated_artifact(&artifact).expect("the tokenizer builds");
    let prompt_token_ids = tokenizer
        .encode_prompt(prompt)
        .expect("the prompt tokenizes");
    let mut model = Qwen3_5StreamingModel::load(
        runtime,
        artifact,
        &target_directory,
        crate::common::standard_qwen3_5_model_chunking_configuration(),
    )
    .expect("the target loads in the streaming engine's model");
    model
        .enable_draft_tree_verification()
        .expect("the tree kernels declare");
    eprintln!(
        "[dflash2-streaming] target {} under a {:.1} GB limit, {} prompt tokens",
        target_directory.display(),
        memory_limit_bytes as f64 / (1u64 << 30) as f64,
        prompt_token_ids.len()
    );
    StreamingTarget {
        model,
        config,
        prompt_token_ids,
    }
}

/// The prompt in the state, and the target's greedy pick after it.
fn prefilled(target: &StreamingTarget) -> (RequestDecoderStateStack, u32) {
    let mut state = crate::common::standard_request_decoder_state(&target.config);
    let last_prompt_index = target.prompt_token_ids.len() - 1;
    target
        .model
        .prefill_chunk(&target.prompt_token_ids[..last_prompt_index], 0, &mut state)
        .expect("the prompt prefills");
    let logits = target
        .model
        .forward_chunk(
            &target.prompt_token_ids[last_prompt_index..],
            last_prompt_index as u32,
            &mut state,
        )
        .expect("the last prompt token produces logits");
    let first_pick = target
        .model
        .highest_logit_token_id(&logits)
        .expect("the logits have a highest token");
    (state, first_pick)
}

/// The greedy continuation one token a step: the tokens fed and each step's
/// logits (row `r` is after feeding token `r`).
fn one_token_steps(target: &StreamingTarget, step_count: usize) -> (Vec<u32>, Vec<Vec<f32>>, f64) {
    let (mut state, first_pick) = prefilled(target);
    let mut position = target.prompt_token_ids.len() as u32;
    let mut tokens = vec![first_pick];
    let mut rows = Vec::with_capacity(step_count);
    let started_at = Instant::now();
    for step_index in 0..step_count {
        let logits = target
            .model
            .forward_chunk(&tokens[step_index..=step_index], position, &mut state)
            .expect("the one-token step runs");
        tokens.push(
            target
                .model
                .highest_logit_token_id(&logits)
                .expect("the step has a highest token"),
        );
        rows.push(logits.to_vec_f32().expect("the step's logits read back"));
        position += 1;
    }
    let milliseconds_a_step = started_at.elapsed().as_secs_f64() * 1e3 / step_count as f64;
    tokens.truncate(step_count);
    (tokens, rows, milliseconds_a_step)
}

/// The same tokens as chains of `ROWS_PER_ROUND` through the tree verify,
/// each chain committed whole: every row's logits.
fn chain_rounds(target: &StreamingTarget, tokens: &[u32]) -> (Vec<Vec<f32>>, f64) {
    let (mut state, _) = prefilled(target);
    let mut position = target.prompt_token_ids.len() as u32;
    let vocabulary_size = target.config.vocabulary_size() as usize;
    let mut rows = Vec::with_capacity(tokens.len());
    let started_at = Instant::now();
    for round_tokens in tokens.chunks(ROWS_PER_ROUND) {
        let mut chain = DraftTree::default();
        for (row_index, token_id) in round_tokens.iter().enumerate().skip(1) {
            chain.token_ids.push(*token_id);
            chain.parent_indices.push((row_index - 1).checked_sub(1));
            chain.depths.push(row_index - 1);
        }
        let tree_rows = TreeRows::from_draft_tree(round_tokens[0], &chain);
        let verification = target
            .model
            .verify_draft_tree(&tree_rows, position, &mut state, &[])
            .expect("the chain verifies on the streaming model");
        let round_logits = verification
            .per_row_logits
            .to_vec_f32()
            .expect("the round's logits read back");
        for row_logits in round_logits.chunks(vocabulary_size) {
            rows.push(row_logits.to_vec());
        }
        let kept_rows: Vec<usize> = (0..round_tokens.len()).collect();
        target
            .model
            .commit_draft_tree(&mut state, verification, &kept_rows)
            .expect("the chain commits");
        position += round_tokens.len() as u32;
    }
    let round_count = tokens.len().div_ceil(ROWS_PER_ROUND);
    let milliseconds_a_round = started_at.elapsed().as_secs_f64() * 1e3 / round_count as f64;
    (rows, milliseconds_a_round)
}

fn highest_logit_token(row: &[f32]) -> usize {
    let mut highest_token = 0;
    for (token_id, logit) in row.iter().enumerate() {
        if *logit > row[highest_token] {
            highest_token = token_id;
        }
    }
    highest_token
}

/// The gap between a row's highest logit and its runner-up.
fn top_two_margin(row: &[f32]) -> f32 {
    let highest_token = highest_logit_token(row);
    let mut runner_up = f32::NEG_INFINITY;
    for (token_id, logit) in row.iter().enumerate() {
        if token_id != highest_token {
            runner_up = runner_up.max(*logit);
        }
    }
    row[highest_token] - runner_up
}

fn largest_difference(reference: &[f32], candidate: &[f32]) -> f32 {
    let mut largest = 0.0_f32;
    for (reference_logit, candidate_logit) in reference.iter().zip(candidate) {
        largest = largest.max((reference_logit - candidate_logit).abs());
    }
    largest
}

#[test]
#[ignore = "loads a real target in the streaming model; run through cargo measure-dflash2-streaming-tree-verify"]
#[serial]
fn should_give_tree_rows_on_the_streaming_model_the_one_token_steps_picks() {
    let target = load_streaming_target(PROMPT);
    let step_count = ROWS_PER_ROUND * ROUND_COUNT;
    // The first pass of each kind reads its experts from storage and compiles
    // its kernels; the second is the steady state a long reply runs in.
    let (tokens, step_rows, _) = one_token_steps(&target, step_count);
    let (tree_rows, _) = chain_rounds(&target, &tokens);
    let (_, _, milliseconds_a_step) = one_token_steps(&target, step_count);
    let (_, milliseconds_a_round) = chain_rounds(&target, &tokens);
    assert_eq!(tree_rows.len(), step_rows.len(), "every row is verified");
    let mut agreeing_pick_count = 0;
    let mut unexplained_rows = Vec::new();
    let mut largest_logit_difference = 0.0_f32;
    for (row_index, (step_row, tree_row)) in step_rows.iter().zip(&tree_rows).enumerate() {
        let difference = largest_difference(step_row, tree_row);
        largest_logit_difference = largest_logit_difference.max(difference);
        let picks_agree = highest_logit_token(step_row) == highest_logit_token(tree_row);
        agreeing_pick_count += usize::from(picks_agree);
        // Two logits that each moved by `difference` can swap places only
        // when they started closer than twice that.
        if !picks_agree && top_two_margin(step_row) > 2.0 * difference {
            unexplained_rows.push(row_index);
        }
        eprintln!(
            "[dflash2-streaming] row {row_index:>2}: largest logit difference {difference:>7.4}, pick {}",
            if picks_agree { "same" } else { "DIFFERENT" }
        );
    }
    eprintln!(
        "[dflash2-streaming] {agreeing_pick_count}/{step_count} picks agree, largest logit difference {largest_logit_difference:.4}"
    );
    eprintln!(
        "[dflash2-streaming] without the engine's expert residency plan: plain step {milliseconds_a_step:.1} ms; {ROWS_PER_ROUND}-row round {milliseconds_a_round:.1} ms ({:.1} ms a row); break-even at {:.2} tokens kept a round",
        milliseconds_a_round / ROWS_PER_ROUND as f64,
        milliseconds_a_round / milliseconds_a_step
    );
    // Rows verified together round a little differently from rows fed alone
    // (the quantized matmul picks its kernel by row count), as on the
    // resident model, so a near-tied pick may flip. A flip the rounding
    // cannot explain, or disagreement on most rows, is a wrong verify.
    assert!(
        unexplained_rows.is_empty(),
        "these rows picked differently with a margin rounding cannot close: {unexplained_rows:?}"
    );
    assert!(
        agreeing_pick_count * 4 >= step_count * 3,
        "tree rows must reproduce most one-token picks: {agreeing_pick_count}/{step_count}"
    );
}

/// The Romeo and Juliet words the measured prompt is cut from.
const LONG_PROMPT_SOURCE: &str =
    include_str!("../../../fixtures/model_metrics_10000_romeo_and_juliet_words.txt");
/// About 500 tokens: enough for the reply's residency plan to have demand
/// to plan from, read in one short pass.
const STAGE_PROMPT_WORD_COUNT: usize = 350;
const STAGE_ROUND_COUNT: usize = 48;
/// Rounds before the measured ones: the first rounds of a reply read the
/// experts its text needs and compile the tree kernels.
const STAGE_WARM_UP_ROUND_COUNT: usize = 16;
const BYTES_PER_MB: f64 = 1_048_576.0;

/// Prints how far a measured loop is and when it should end, at most every
/// five seconds.
struct LoopProgress {
    label: &'static str,
    pass_count: usize,
    started_at: Instant,
    last_printed_at: Instant,
}

impl LoopProgress {
    fn new(label: &'static str, pass_count: usize) -> Self {
        Self {
            label,
            pass_count,
            started_at: Instant::now(),
            last_printed_at: Instant::now(),
        }
    }

    fn passed(&mut self, completed_pass_count: usize) {
        if self.last_printed_at.elapsed().as_secs() < 5 && completed_pass_count < self.pass_count {
            return;
        }
        self.last_printed_at = Instant::now();
        let elapsed_seconds = self.started_at.elapsed().as_secs_f64();
        let remaining_seconds = elapsed_seconds / completed_pass_count as f64
            * (self.pass_count - completed_pass_count) as f64;
        eprintln!(
            "[dflash2-streaming-stages] progress: {} {completed_pass_count}/{} ({}%), {elapsed_seconds:.0} s elapsed, about {remaining_seconds:.0} s left",
            self.label,
            self.pass_count,
            completed_pass_count * 100 / self.pass_count,
        );
    }
}

/// What a run of plain steps or of rounds spent, summed over the measured ones.
#[derive(Default)]
struct StageTotals {
    measured_count: usize,
    graph_build_seconds: f64,
    evaluation_seconds: f64,
    commit_seconds: f64,
    dispatch_counts: ExpertDispatchCounts,
    storage_read_bytes: u64,
}

impl StageTotals {
    fn add_dispatch_counts(&mut self, counts: ExpertDispatchCounts) {
        let totals = &mut self.dispatch_counts;
        totals.complete_page_in_hand_layer_count += counts.complete_page_in_hand_layer_count;
        totals.table_without_host_route_layer_count += counts.table_without_host_route_layer_count;
        totals.host_route_table_hit_layer_count += counts.host_route_table_hit_layer_count;
        totals.host_route_table_miss_layer_count += counts.host_route_table_miss_layer_count;
        totals.unhandled_layer_count += counts.unhandled_layer_count;
        totals.host_route_wait_nanoseconds += counts.host_route_wait_nanoseconds;
    }

    fn print(&self, label: &str) {
        let count = self.measured_count as f64;
        let counts = &self.dispatch_counts;
        eprintln!(
            "[dflash2-streaming-stages] {label}: {:.1} ms (build {:.1}, evaluation {:.1}, commit {:.1}); of the build, {:.1} ms waiting for routes",
            (self.graph_build_seconds + self.evaluation_seconds + self.commit_seconds) * 1e3
                / count,
            self.graph_build_seconds * 1e3 / count,
            self.evaluation_seconds * 1e3 / count,
            self.commit_seconds * 1e3 / count,
            counts.host_route_wait_nanoseconds as f64 / 1e6 / count,
        );
        eprintln!(
            "[dflash2-streaming-stages] {label}: layers a pass: {:.1} table without a wait, {:.1} route to host then table hit, {:.1} route to host then table miss, {:.1} page in hand, {:.1} other; {:.2} MB read from storage",
            counts.table_without_host_route_layer_count as f64 / count,
            counts.host_route_table_hit_layer_count as f64 / count,
            counts.host_route_table_miss_layer_count as f64 / count,
            counts.complete_page_in_hand_layer_count as f64 / count,
            counts.unhandled_layer_count as f64 / count,
            self.storage_read_bytes as f64 / BYTES_PER_MB / count,
        );
    }
}

fn enter_memory_phase(target: &StreamingTarget, phase: MemoryPhase, operation_token_count: usize) {
    target
        .model
        .enter_memory_phase_for_tests(
            phase,
            target.prompt_token_ids.len() as u64,
            operation_token_count as u64,
            &mut PerformanceAttribution::disabled(),
        )
        .expect("the memory phase is entered as the engine enters it");
}

/// The prompt read and the reply's residency planned, as the engine does
/// before a reply's first token.
fn prefilled_as_the_engine_would(target: &StreamingTarget) -> (RequestDecoderStateStack, u32) {
    let prefill_started_at = Instant::now();
    enter_memory_phase(target, MemoryPhase::Prefill, target.prompt_token_ids.len());
    let prefilled_state_and_pick = prefilled(target);
    target.model.prepare_expert_residency_for_reply_for_tests();
    eprintln!(
        "[dflash2-streaming-stages] progress: prompt read and reply planned in {:.1} s",
        prefill_started_at.elapsed().as_secs_f64()
    );
    prefilled_state_and_pick
}

/// Plain steps under the engine's memory state: the greedy tokens fed and
/// what the measured steps spent.
fn timed_plain_steps(target: &StreamingTarget, step_count: usize) -> (Vec<u32>, StageTotals) {
    let (mut state, first_pick) = prefilled_as_the_engine_would(target);
    let mut position = target.prompt_token_ids.len() as u32;
    let mut tokens = vec![first_pick];
    let mut totals = StageTotals::default();
    let warm_up_step_count = STAGE_WARM_UP_ROUND_COUNT * ROWS_PER_ROUND;
    let _ = target.model.take_expert_dispatch_counts();
    let mut progress = LoopProgress::new("plain steps", step_count);
    for step_index in 0..step_count {
        let read_bytes_before = target.model.request_expert_storage_read_bytes();
        let step_started_at = Instant::now();
        let logits = target
            .model
            .forward_chunk(&tokens[step_index..=step_index], position, &mut state)
            .expect("the one-token step runs");
        let graph_build_seconds = step_started_at.elapsed().as_secs_f64();
        tokens.push(
            target
                .model
                .highest_logit_token_id(&logits)
                .expect("the step has a highest token"),
        );
        let step_seconds = step_started_at.elapsed().as_secs_f64();
        let dispatch_counts = target.model.take_expert_dispatch_counts();
        if step_index >= warm_up_step_count {
            totals.measured_count += 1;
            totals.graph_build_seconds += graph_build_seconds;
            totals.evaluation_seconds += step_seconds - graph_build_seconds;
            totals.add_dispatch_counts(dispatch_counts);
            totals.storage_read_bytes +=
                target.model.request_expert_storage_read_bytes() - read_bytes_before;
        }
        position += 1;
        progress.passed(step_index + 1);
    }
    tokens.truncate(step_count);
    (tokens, totals)
}

/// One chain of `round_tokens` verified, evaluated and committed whole, as a
/// round that keeps every row; the seconds its build, evaluation and commit took.
fn timed_chain_round(
    target: &StreamingTarget,
    round_tokens: &[u32],
    position: u32,
    state: &mut RequestDecoderStateStack,
) -> [f64; 3] {
    let mut chain = DraftTree::default();
    for (row_index, token_id) in round_tokens.iter().enumerate().skip(1) {
        chain.token_ids.push(*token_id);
        chain.parent_indices.push((row_index - 1).checked_sub(1));
        chain.depths.push(row_index - 1);
    }
    let tree_rows = TreeRows::from_draft_tree(round_tokens[0], &chain);
    let round_started_at = Instant::now();
    let verification = target
        .model
        .verify_draft_tree(&tree_rows, position, state, &[])
        .expect("the chain verifies on the streaming model");
    let graph_build_seconds = round_started_at.elapsed().as_secs_f64();
    let pass_is_valid = target
        .model
        .evaluate_draft_tree_pass(&[&verification.per_row_logits])
        .expect("the pass evaluates");
    assert!(
        pass_is_valid,
        "a pass built without assumptions is always usable"
    );
    let evaluated_seconds = round_started_at.elapsed().as_secs_f64();
    let kept_rows: Vec<usize> = (0..round_tokens.len()).collect();
    target
        .model
        .commit_draft_tree(state, verification, &kept_rows)
        .expect("the chain commits");
    [
        graph_build_seconds,
        evaluated_seconds - graph_build_seconds,
        round_started_at.elapsed().as_secs_f64() - evaluated_seconds,
    ]
}

/// The same tokens as chains of `ROWS_PER_ROUND` under the engine's memory
/// state: what the measured rounds spent.
fn timed_chain_rounds(target: &StreamingTarget, tokens: &[u32]) -> StageTotals {
    let (mut state, _) = prefilled_as_the_engine_would(target);
    let mut position = target.prompt_token_ids.len() as u32;
    let mut totals = StageTotals::default();
    let _ = target.model.take_expert_dispatch_counts();
    let mut progress = LoopProgress::new("rounds", tokens.len().div_ceil(ROWS_PER_ROUND));
    for (round_index, round_tokens) in tokens.chunks(ROWS_PER_ROUND).enumerate() {
        let read_bytes_before = target.model.request_expert_storage_read_bytes();
        let [graph_build_seconds, evaluation_seconds, commit_seconds] =
            timed_chain_round(target, round_tokens, position, &mut state);
        let dispatch_counts = target.model.take_expert_dispatch_counts();
        if round_index >= STAGE_WARM_UP_ROUND_COUNT {
            totals.measured_count += 1;
            totals.graph_build_seconds += graph_build_seconds;
            totals.evaluation_seconds += evaluation_seconds;
            totals.commit_seconds += commit_seconds;
            totals.add_dispatch_counts(dispatch_counts);
            totals.storage_read_bytes +=
                target.model.request_expert_storage_read_bytes() - read_bytes_before;
        }
        position += round_tokens.len() as u32;
        progress.passed(round_index + 1);
    }
    totals
}

#[test]
#[ignore = "loads a real target in the streaming model; run through cargo measure-dflash2-streaming-round-stages"]
#[serial]
fn should_measure_where_a_streaming_tree_round_spends_its_time() {
    let long_prompt: Vec<&str> = LONG_PROMPT_SOURCE
        .split_whitespace()
        .take(STAGE_PROMPT_WORD_COUNT)
        .collect();
    let target = load_streaming_target(&long_prompt.join(" "));
    let step_count = ROWS_PER_ROUND * STAGE_ROUND_COUNT;
    let (tokens, plain_totals) = timed_plain_steps(&target, step_count);
    let round_totals = timed_chain_rounds(&target, &tokens);
    plain_totals.print("plain step");
    round_totals.print(&format!("{ROWS_PER_ROUND}-row round"));
}
