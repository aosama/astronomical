//! Gate 1 of speculative decoding: how many tokens a verify round keeps with
//! Astronomical's DFlash2 drafter on a real pair. By default that is the
//! installed dense Qwen3.8 27B package, which carries its DFlash2 drafter in
//! its own `drafter` folder; a test run that configures another target with
//! its own `drafter` folder measures that pair instead.
//!
//! Kept per round = tokens emitted / verify rounds, the bonus token included,
//! as the drafter's model card defines it. Every code journey decodes one
//! code cell: TensorFold's community benchmark, which sends the code prompt
//! as a raw completion at 0.7 with its top-k 20 and top-p 0.95, so its speed
//! compares like for like. z-lab's 6.07 was measured on the chat-template
//! form of the prompt, so the code cell reports without a floor, as does the
//! story at 1.0.
//!
//! It also prints each cell's decode speed (prefill excluded); every stage
//! still syncs with the GPU, so `dflash2_round_cost` is where stages are timed.

use std::path::PathBuf;
use std::time::Instant;

use ipc_protocol::RequestId;
use mlx_bridge::MlxArray;
use model_serving::gpu_token_sampling::sampling_distribution_rows;
use model_serving::speculative_decoding::dflash_drafter::{
    DFlashDrafterConfiguration, DFlashDrafterWeights, DrafterContextStates, bind_drafter_weights,
};
use model_serving::speculative_decoding::verification_decision::{
    DraftedSlot, TargetDistribution, VerificationDecision, decide_by_rejection_sampling,
    decide_by_token_equality,
};
use model_serving::{
    PerformanceAttribution, Qwen3_5ArtifactValidator, Qwen3_5TextModel, Qwen3_5Tokenizer,
    RequestDecoderStateStack, TextModelChunkingConfiguration,
};
use runtime_integration::{MlxMemoryLimits, MlxRuntime, MlxSafetensors};
use serial_test::serial;

use super::dflash2_journey_support::{
    UniformStream, VerifiedBlock, host_values, tap_rows, verify_block_with_taps,
};
use super::kept_prompt_state::{KeptPromptState, ReadPrompt, no_kept_prompt_state, read_prompt};

mod round_drafting;

use round_drafting::{drafter_distribution_of_path, propose_path};

pub(crate) const CODE_PROMPT: &str =
    "Write a short Python function that computes the Fibonacci sequence and explain it.";
pub(crate) const OUTPUT_TOKENS: usize = 256;
const SAMPLING_SEED: u64 = 1234;
/// The engine's draft-tree node budget, `DraftTreeOptions::default()`;
/// a `const` here because the cells below are built at compile time.
pub(crate) const ENGINE_NODE_BUDGET: usize = 15;
/// The engine's branching, `DraftTreeOptions::default()`.
pub(crate) const ENGINE_CHILDREN_PER_NODE: usize = 16;
const SAMPLED_SEED_COUNT: u64 = 3;
/// Qwen3.8's recommended sampling, as users run it and as the reference ran.
pub(crate) const TOP_P_THOUSANDTHS: u16 = 950;
pub(crate) const TOP_K: u16 = 20;
/// The larger of the two pairs: the 35B-A3B mixture-of-experts target's
/// 23 GiB with its 1 GiB drafter, or the 27B's 15 GiB with its 3.6 GiB of
/// bf16 drafter, plus working room.
const ACTIVE_MEMORY_LIMIT_BYTES: usize = 30 * 1024 * 1024 * 1024;
/// The folder inside a configured target that holds its own drafter, as the
/// engine's drafter discovery reads it.
const BUNDLED_DRAFTER_DIRECTORY_NAME: &str = "drafter";
const MLX_BUFFER_CACHE_LIMIT_BYTES: usize = 2 * 1024 * 1024 * 1024;

/// One measured cell: the prompt, the temperature (0 = greedy diagnostic), and
/// the least kept-per-round that passes.
#[derive(Clone, Copy)]
pub(crate) struct Cell {
    pub(crate) label: &'static str,
    pub(crate) prompt: &'static str,
    /// The prompt as a raw completion rather than a chat turn: TensorFold's
    /// community benchmark sends its code cell this way.
    pub(crate) is_raw_completion: bool,
    pub(crate) temperature_thousandths: u16,
    pub(crate) minimum_kept_per_round: f64,
    /// Nodes a drafted tree may hold, the anchor's row aside: the engine's
    /// own budget unless a journey compares tree sizes.
    pub(crate) node_budget: usize,
    /// Candidates a node may branch into: the engine's 16, or 1 for a chain
    /// as deep as the budget.
    pub(crate) children_per_node: usize,
    /// The keyed sampler's seed, for journeys that sample with it.
    pub(crate) keyed_seed: u64,
}

impl Cell {
    /// How many seeds a journey decodes this cell with, given the count it
    /// pools for a sampled story. Greedy text is fixed, so it runs once. The
    /// code cell runs once too: its kept count moved by 0.2 at most across
    /// three seeds on the Bonsai 2.5 pair, a spread three seeds cannot narrow
    /// enough to tell a real change from, so the extra decodes bought nothing.
    pub(crate) fn seed_count(&self, sampled_story_seed_count: u64) -> u64 {
        if self.temperature_thousandths == 0 || self.prompt == CODE_PROMPT {
            1
        } else {
            sampled_story_seed_count
        }
    }
}

/// The one code cell, then the story. z-lab publishes a floor only for
/// its chat-template code cell, so neither reports against one.
fn kept_per_round_cells() -> Vec<Cell> {
    vec![
        Cell {
            keyed_seed: SAMPLING_SEED,
            ..super::dflash2_tree_oracle::CODE_CELL
        },
        Cell {
            keyed_seed: SAMPLING_SEED,
            ..super::dflash2_journey_support::story_cell("story 1.0", 1_000)
        },
    ]
}

pub(crate) struct JourneyPair {
    pub(crate) target: Qwen3_5TextModel,
    /// The model ID chat commands name: the loaded target's folder leaf.
    pub(crate) target_model_id: String,
    pub(crate) tokenizer: Qwen3_5Tokenizer,
    pub(crate) drafter_configuration: DFlashDrafterConfiguration,
    pub(crate) drafter_weights: DFlashDrafterWeights,
    /// Keeps the drafter's lazily mapped tensors alive.
    _drafter_checkpoint: MlxSafetensors,
    /// The last prompt this pair read, so the next decode of it skips the
    /// reading (`kept_prompt_state`).
    pub(crate) kept_prompt_state: std::cell::RefCell<Option<KeptPromptState>>,
}

#[test]
#[ignore = "loads a real target and its DFlash2 drafter (the 27B pair unless configured); run through cargo measure-dflash2-kept-per-round"]
#[serial]
fn should_keep_as_many_drafted_tokens_per_round_as_z_labs_reference() {
    let journey_started_at = Instant::now();
    let pair = load_pair();
    eprintln!(
        "[dflash2-gate1] pair loaded in {} s",
        journey_started_at.elapsed().as_secs()
    );
    let mut failures = Vec::new();
    let cells = kept_per_round_cells();
    for (cell_index, cell) in cells.iter().enumerate() {
        let cell_started_at = Instant::now();
        // A sampled story's kept count swings with the text it happens to
        // sample (about 0.16 either way over one 256-token run), so it pools
        // three seeds; see `Cell::seed_count` for the cells that run once.
        let seed_count = cell.seed_count(SAMPLED_SEED_COUNT);
        let (mut rounds, mut emitted, mut decode_seconds) = (0, 0, 0.0);
        for seed_offset in 0..seed_count {
            let seed_outcome = run_cell(&pair, cell, SAMPLING_SEED + seed_offset);
            rounds += seed_outcome.rounds;
            emitted += seed_outcome.emitted_tokens;
            decode_seconds += seed_outcome.decode_seconds;
        }
        let kept_per_round = emitted as f64 / rounds as f64;
        let decode_tokens_per_second = emitted as f64 / decode_seconds;
        let elapsed_seconds = journey_started_at.elapsed().as_secs_f64();
        let remaining_seconds =
            elapsed_seconds / (cell_index + 1) as f64 * (cells.len() - cell_index - 1) as f64;
        eprintln!(
            "[dflash2-gate1 {}/{} ETA {remaining_seconds:.0} s] {} T={:.1}: {kept_per_round:.2} kept per round, \
             {decode_tokens_per_second:.1} tok/s decode ({emitted} tokens in {rounds} rounds, {:.0} s)",
            cell_index + 1,
            cells.len(),
            cell.label,
            f64::from(cell.temperature_thousandths) / 1_000.0,
            cell_started_at.elapsed().as_secs_f64()
        );
        if kept_per_round < cell.minimum_kept_per_round {
            failures.push(format!(
                "{}: {kept_per_round:.2} < {}",
                cell.label, cell.minimum_kept_per_round
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "kept per round below the reference floor: {failures:?}"
    );
}

/// The target and drafter directories: one model folder and the drafter
/// inside it, as the engine serves them. The folder is the test-run
/// configuration's `model_directory` when one is selected, so the same
/// journeys measure any paired target, otherwise the installed dense 27B
/// package (`dense_drafting` in `registry/e2e_test_model_names.json`).
pub(crate) fn pair_directories() -> (PathBuf, PathBuf, String) {
    let target_directory = test_harness::test_run::TestRunContext::current()
        .configuration
        .model_directory
        .unwrap_or_else(|| {
            crate::common::configured_installed_model_directory_by_id(
                test_harness::fixtures::dense_drafting_model_id(),
            )
        });
    let drafter_directory = target_directory.join(BUNDLED_DRAFTER_DIRECTORY_NAME);
    let target_model_id = target_directory
        .file_name()
        .expect("the target names a folder")
        .to_string_lossy()
        .into_owned();
    (target_directory, drafter_directory, target_model_id)
}

/// The pair with draft-tree verification and the row-exact tree attention
/// kernel on, as serving verifies a dense target's trees.
pub(crate) fn load_pair_verifying_trees_as_served() -> JourneyPair {
    let mut pair = load_pair();
    pair.target
        .enable_draft_tree_verification()
        .expect("the tree kernels declare");
    pair.target
        .enable_row_exact_tree_attention()
        .expect("the tree attention kernel declares");
    pair
}

pub(crate) fn load_pair() -> JourneyPair {
    let (target_directory, drafter_directory, target_model_id) = pair_directories();
    eprintln!(
        "[dflash2] target {} with drafter {}",
        target_directory.display(),
        drafter_directory.display()
    );
    let runtime = MlxRuntime::initialize(
        MlxMemoryLimits::new(ACTIVE_MEMORY_LIMIT_BYTES, MLX_BUFFER_CACHE_LIMIT_BYTES)
            .expect("memory limits are valid"),
    )
    .expect("the MLX runtime initializes");
    let artifact = Qwen3_5ArtifactValidator::new()
        .validate(&target_directory, 4096)
        .expect("the target artifact validates");
    let tokenizer =
        Qwen3_5Tokenizer::from_validated_artifact(&artifact).expect("the tokenizer builds");
    let chunking = TextModelChunkingConfiguration::new(256, 0, 0).expect("the chunking is valid");
    let mut attribution = PerformanceAttribution::disabled();
    let mut target = Qwen3_5TextModel::load_with_performance_attribution(
        runtime,
        artifact,
        &target_directory,
        true,
        chunking,
        &mut attribution,
    )
    .expect("the target loads");
    target
        .materialize_and_keep_every_dense_layer()
        .expect("the target weights materialize and every layer is kept");
    // A drafting engine converts its kept projections into register order
    // once this GPU's probe proves the kernel; every row count then uses it.
    let converted_projection_count = target
        .convert_projections_to_register_order(&mut attribution)
        .expect("the projections convert")
        .unwrap_or(0);
    let register_order_enabled = converted_projection_count > 0;
    eprintln!("[dflash2] register-ordered projections: {converted_projection_count}");
    let config_document: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(drafter_directory.join("config.json"))
            .expect("the drafter config reads"),
    )
    .expect("the drafter config parses");
    let drafter_configuration = DFlashDrafterConfiguration::from_config_document(&config_document)
        .expect("the drafter configuration is accepted");
    let drafter_checkpoint = target
        .runtime()
        .load_safetensors(
            std::fs::File::open(drafter_directory.join("model.safetensors"))
                .expect("the drafter opens"),
            None,
        )
        .expect("the drafter loads");
    let mut drafter_weights = bind_drafter_weights(
        target.runtime(),
        &drafter_checkpoint,
        &drafter_configuration,
    )
    .expect("every drafter tensor binds");
    // The drafter is served at 4 bits, as TensorFold serves it: it streams
    // about 1 GB a round instead of the checkpoint's 3.6 GB of bf16.
    drafter_weights
        .quantize_linears(target.runtime(), register_order_enabled)
        .expect("the drafter quantizes to 4 bits");
    JourneyPair {
        target,
        target_model_id,
        tokenizer,
        drafter_configuration,
        drafter_weights,
        _drafter_checkpoint: drafter_checkpoint,
        kept_prompt_state: no_kept_prompt_state(),
    }
}

/// The prompt's tokens with no chat template, as a completion request sends it.
pub(crate) fn completion_token_ids(pair: &JourneyPair, prompt: &str) -> Vec<u32> {
    pair.tokenizer
        .encode_prompt(prompt)
        .expect("the completion prompt tokenizes")
}

pub(crate) fn prompt_token_ids(pair: &JourneyPair, prompt: &str) -> Vec<u32> {
    let command = ipc_protocol::ChatGenerationCommand {
        request_id: RequestId::new(0),
        model: pair.target_model_id.clone(),
        messages: vec![ipc_protocol::ChatMessage::User {
            content: prompt.to_owned(),
            images: Vec::new(),
        }],
        tools: Vec::new(),
        tool_choice: ipc_protocol::ChatToolChoice::None,
        settings: ipc_protocol::ChatGenerationSettings {
            max_output_tokens: OUTPUT_TOKENS as u16,
            temperature_thousandths: Some(0),
            top_p_thousandths: None,
            repetition_penalty_thousandths: None,
            seed: None,
            thinking_budget: Some(0),
        },
        structured_generation: None,
    };
    pair.tokenizer
        .prepare_chat(&command, false)
        .expect("the prompt tokenizes")
        .input_token_ids()
        .to_vec()
}

/// One cell's request: prefill, then draft-and-verify rounds until the
/// output budget is spent. Returns (verify rounds, tokens emitted by them).
/// One request's drafted decode: rounds run, tokens they emitted, and the
/// wall time from the first drafted round to the last, prefill excluded.
pub(crate) struct CellOutcome {
    pub(crate) rounds: usize,
    pub(crate) emitted_tokens: usize,
    pub(crate) decode_seconds: f64,
}

pub(crate) fn run_cell(pair: &JourneyPair, cell: &Cell, sampling_seed: u64) -> CellOutcome {
    let ReadPrompt {
        mut decoder_state,
        last_logits: first_logits,
        tap_rows: pending_tap_rows,
        prompt_token_count,
    } = read_prompt(pair, cell);
    let mut uniforms = UniformStream::seeded(sampling_seed);
    let mut random_state = pair
        .target
        .runtime()
        .random_key(sampling_seed)
        .expect("a key builds");
    let mut round = RoundState {
        last_token_id: decide(pair, cell, &[], &first_logits, None, &mut uniforms).next_token_id,
        fed_token_count: prompt_token_count as u32,
        pending_tap_rows,
        context_states: DrafterContextStates::new(&pair.drafter_configuration, 0),
    };
    let decode_started_at = Instant::now();
    let mut emitted_token_ids = vec![round.last_token_id];
    let mut rounds = 0;
    while emitted_token_ids.len() <= OUTPUT_TOKENS {
        emitted_token_ids.extend(run_round(
            pair,
            cell,
            &mut round,
            &mut decoder_state,
            &mut random_state,
            &mut uniforms,
        ));
        rounds += 1;
    }
    CellOutcome {
        rounds,
        emitted_tokens: emitted_token_ids.len() - 1,
        decode_seconds: decode_started_at.elapsed().as_secs_f64(),
    }
}

pub(crate) struct RoundState {
    pub(crate) last_token_id: u32,
    /// Tokens already in the target's decoder state; `last_token_id` is not.
    pub(crate) fed_token_count: u32,
    /// Fused-tap input rows the drafter has not absorbed yet.
    pub(crate) pending_tap_rows: MlxArray,
    pub(crate) context_states: DrafterContextStates,
}

/// Drafts a block, verifies it in one target pass, keeps what the target
/// accepts and rewinds the rest. Returns the tokens the round emitted.
fn run_round(
    pair: &JourneyPair,
    cell: &Cell,
    round: &mut RoundState,
    decoder_state: &mut RequestDecoderStateStack,
    random_state: &mut MlxArray,
    uniforms: &mut UniformStream,
) -> Vec<u32> {
    let runtime = pair.target.runtime();
    let configuration = &pair.drafter_configuration;
    let path = propose_path(pair, cell, round, random_state);
    let drafted_ids: Vec<u32> = host_values(runtime, &path.token_ids)
        .iter()
        .map(|value| *value as u32)
        .collect();
    let mut verified_ids = vec![round.last_token_id];
    verified_ids.extend_from_slice(&drafted_ids);
    let verified = verify_block_with_taps(
        &pair.target,
        &verified_ids,
        round.fed_token_count,
        decoder_state,
        &configuration.target_tap_layer_ids,
    );
    let logits = verified
        .verification
        .per_row_logits
        .retain()
        .expect("the logits handle");
    let drafter_distribution = drafter_distribution_of_path(runtime, &path);
    let decision = decide(
        pair,
        cell,
        &drafted_ids,
        &logits,
        drafter_distribution.as_ref(),
        uniforms,
    );
    settle_round(pair, round, decoder_state, verified, decision);
    let mut emitted = drafted_ids[..decision.kept_draft_count].to_vec();
    emitted.push(decision.next_token_id);
    emitted
}

/// Keeps the accepted rows in the target and the drafter's pending context:
/// a round that keeps fewer rows than it verified rewinds the target to the
/// kept row's captured state, with no second pass.
pub(crate) fn settle_round(
    pair: &JourneyPair,
    round: &mut RoundState,
    decoder_state: &mut RequestDecoderStateStack,
    verified: VerifiedBlock,
    decision: VerificationDecision,
) {
    let kept_row_count = decision.kept_draft_count + 1;
    let kept_token_offset = round.fed_token_count + kept_row_count as u32;
    pair.target
        .keep_drafted_rows(verified.verification, kept_row_count, decoder_state)
        .expect("the decoder state rewinds to the kept rows");
    round.pending_tap_rows = tap_rows(
        pair.target.runtime(),
        &verified.hidden_state_taps,
        kept_row_count as i32,
    );
    round.fed_token_count = kept_token_offset;
    round.last_token_id = decision.next_token_id;
}

/// The drafter's sampled `q`, flattened `[proposals x candidates]` on the host.
struct DrafterDistribution {
    candidate_ids: Vec<u32>,
    candidate_probabilities: Vec<f32>,
}

/// The target's verdict on `drafted_ids` from its `logits` rows; with no
/// drafts it is the target's own pick for the next token.
fn decide(
    pair: &JourneyPair,
    cell: &Cell,
    drafted_ids: &[u32],
    logits: &MlxArray,
    drafter_distribution: Option<&DrafterDistribution>,
    uniforms: &mut UniformStream,
) -> VerificationDecision {
    let runtime = pair.target.runtime();
    let row_count = logits.shape()[1] as usize;
    if cell.temperature_thousandths == 0 {
        let argmax = host_values(
            runtime,
            &runtime.argmax_axis(logits, -1).expect("argmax builds"),
        );
        let target_ids: Vec<u32> = argmax.iter().map(|value| *value as u32).collect();
        return decide_by_token_equality(
            drafted_ids,
            &target_ids[target_ids.len() - drafted_ids.len() - 1..],
        );
    }
    // Only each row's top-k tokens can be sampled, so only they are read back.
    let (distribution_ids, distribution_probabilities) = sampling_distribution_rows(
        runtime,
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
    let kept_per_row = kept_ids.len() / row_count;
    let first_row = row_count - drafted_ids.len() - 1;
    let mut rows = Vec::with_capacity(drafted_ids.len() + 1);
    for row_index in first_row..row_count {
        let row_range = row_index * kept_per_row..(row_index + 1) * kept_per_row;
        rows.push(TargetDistribution {
            token_ids: &kept_ids[row_range.clone()],
            probabilities: &kept_probabilities[row_range],
        });
    }
    let candidate_count = pair.drafter_configuration.lattice_candidate_count() as usize;
    let mut drafted_slots = Vec::with_capacity(drafted_ids.len());
    if let Some(distribution) = drafter_distribution {
        for (slot_index, drafted_id) in drafted_ids.iter().enumerate() {
            let slot_range = slot_index * candidate_count..(slot_index + 1) * candidate_count;
            drafted_slots.push(DraftedSlot {
                token_id: *drafted_id,
                candidate_ids: &distribution.candidate_ids[slot_range.clone()],
                candidate_probabilities: &distribution.candidate_probabilities[slot_range],
            });
        }
    }
    let mut uniform_draws = Vec::with_capacity(drafted_ids.len() + 1);
    for _ in 0..=drafted_ids.len() {
        uniform_draws.push(uniforms.next_uniform());
    }
    decide_by_rejection_sampling(&drafted_slots, &rows, &uniform_draws)
}

#[test]
fn should_decode_the_code_cell_and_greedy_cells_once_and_sampled_stories_at_the_pooled_count() {
    let sampled_story = Cell {
        label: "story 1.0",
        prompt: "Continue the story.",
        is_raw_completion: false,
        temperature_thousandths: 1_000,
        minimum_kept_per_round: 0.0,
        node_budget: ENGINE_NODE_BUDGET,
        children_per_node: ENGINE_CHILDREN_PER_NODE,
        keyed_seed: SAMPLING_SEED,
    };
    let greedy_story = Cell {
        temperature_thousandths: 0,
        ..sampled_story
    };
    assert_eq!(sampled_story.seed_count(3), 3);
    assert_eq!(sampled_story.seed_count(0), 0, "a zero count stays zero");
    assert_eq!(greedy_story.seed_count(3), 1);
    assert_eq!(super::dflash2_tree_oracle::CODE_CELL.seed_count(3), 1);
}
