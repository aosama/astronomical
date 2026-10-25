//! DFlash speculative-decoding proof: acceptance rate and decode throughput.
//!
//! This is the measurement gate for onboarding DFlash into Astronomical. It
//! loads the real resident target (the OptiQ 4-bit Ornith-1.5-35B-A3B from
//! the e2e registry) beside the real DFlash draft (resolved as the draft's
//! library sibling directory, because a draft artifact is intentionally
//! invisible to executable model discovery), bootstraps the draft's
//! context key-value cache from the target's hidden-state taps, then
//! measures greedy decode two ways on the same continued context: one
//! token per forward (baseline) and draft-and-verify with eight
//! speculative tokens per round.
//!
//! The journey reports acceptance rate, per-mode tokens per second, and the
//! speedup ratio. It asserts only accounting invariants — token counts and
//! context bookkeeping — because the performance numbers are evidence, not
//! thresholds, and they decide whether the integration proceeds.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use astronomical_ipc_protocol::RequestId;
use astronomical_model_serving::{
    DFlashDraftConfiguration, DFlashDraftKeyValueStates, DFlashDraftLoadedWeights,
    PerformanceAttribution, Qwen3_5ArtifactValidator, Qwen3_5ModelChunkingConfiguration,
    Qwen3_5ResidentModel, Qwen3_5Tokenizer, RequestDecoderStateStack, SpeculationRoundRequest,
    ValidatedQwen3_5Artifact, VerificationLogitsMode, append_conditioned_context_key_values,
    condition_target_hidden_states, run_speculative_round,
};
use astronomical_runtime_integration::{MlxMemoryLimits, MlxRuntime};
use mlx_c_rust::MlxArray;
use serial_test::serial;
use tokio::time::timeout;

use crate::common::{
    configured_installed_model_directory_by_id, direct_mlx_test_guard,
    ornith_dflash_draft_model_id, resident_sparse_moe_model_id,
};

/// Full-sized real models on the graphics processor: the resident target
/// (~20 GB of OptiQ 4-bit weights) plus the BF16 (bfloat16) draft (~0.8 GB)
/// and both key-value caches. The active limit stays under the machine's
/// wired ceiling with headroom for activation workspaces.
const PROOF_ACTIVE_MEMORY_LIMIT_BYTES: usize = 34 * 1024 * 1024 * 1024;
const PROOF_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES: usize = 4 * 1024 * 1024 * 1024;

/// One side of the measured comparison.
const MEASURED_OUTPUT_TOKENS: usize = 500;
/// The warmup contract for throughput journeys: untimed decode tokens.
const WARMUP_OUTPUT_TOKENS: usize = 50;
/// Draft block length used by the reference serving recipes.
const SPECULATION_TOKEN_COUNT: usize = 8;
/// Prefill chunk length for the tap-capturing bootstrap, matching the
/// resident engine's default prompt-processing granularity.
const PREFILL_CHUNK_TOKENS: usize = 2048;
/// The journey's overall runner budget.
const PROOF_JOURNEY_BUDGET_SECONDS: u64 = 900;

#[tokio::test]
#[ignore = "measures DFlash speculative-decoding acceptance and decode throughput on the real resident target"]
#[serial]
async fn should_measure_dflash_speculative_acceptance_and_decode_throughput() {
    timeout(
        Duration::from_secs(PROOF_JOURNEY_BUDGET_SECONDS),
        measure_dflash_speculative_acceptance_and_decode_throughput(),
    )
    .await
    .expect("the DFlash proof journey must finish within its budget");
}

struct ProofJourney {
    target_model: Qwen3_5ResidentModel,
    target_decoder_state: RequestDecoderStateStack,
    /// Owner of both the bound tensors and the safetensors reader they stay
    /// lazy against; the reader must outlive the arrays it produced.
    draft: DFlashDraftLoadedWeights,
    draft_configuration: DFlashDraftConfiguration,
    draft_key_value_states: DFlashDraftKeyValueStates,
    prompt_token_ids: Vec<u32>,
    /// The target's greedy preference at the terminal prefill position:
    /// the first bonus token every decode phase chains from.
    prefill_terminal_bonus_token_id: u32,
    draft_directory: PathBuf,
}

struct DecodePhaseOutcome {
    context_length_after: u32,
    /// The greedy token the last forward produced but no phase has
    /// forwarded yet; the next phase's first bonus.
    next_bonus_token_id: u32,
    accepted_proposal_count: usize,
    proposed_token_count: usize,
    round_count: usize,
}

async fn measure_dflash_speculative_acceptance_and_decode_throughput() {
    let _direct_mlx_guard = direct_mlx_test_guard().await;
    let journey_started_at = Instant::now();
    let mut journey = load_proof_journey(journey_started_at);

    let context_length_tokens = journey.prompt_token_ids.len() as u32;

    // Warmup contract: fifty untimed speculative tokens so every draft and
    // target graph compiles before either side of the comparison is timed.
    println!(
        "[dflash-proof {:>4}s] warmup: {WARMUP_OUTPUT_TOKENS} untimed speculative tokens",
        journey_started_at.elapsed().as_secs()
    );
    let warmup_bonus_token_id = journey.prefill_terminal_bonus_token_id;
    let warmup_outcome = decode_speculative(
        &mut journey,
        warmup_bonus_token_id,
        context_length_tokens,
        WARMUP_OUTPUT_TOKENS,
        journey_started_at,
    );

    println!(
        "[dflash-proof {:>4}s] measuring baseline: {MEASURED_OUTPUT_TOKENS} one-token-at-a-time tokens",
        journey_started_at.elapsed().as_secs()
    );
    let baseline_started_at = Instant::now();
    let baseline_outcome = decode_baseline(
        &mut journey,
        warmup_outcome.next_bonus_token_id,
        warmup_outcome.context_length_after,
        MEASURED_OUTPUT_TOKENS,
    );
    let baseline_elapsed = baseline_started_at.elapsed();

    println!(
        "[dflash-proof {:>4}s] measuring speculative: {MEASURED_OUTPUT_TOKENS} tokens at k={SPECULATION_TOKEN_COUNT}",
        journey_started_at.elapsed().as_secs()
    );
    let speculative_started_at = Instant::now();
    let speculative_outcome = decode_speculative(
        &mut journey,
        baseline_outcome.next_bonus_token_id,
        baseline_outcome.context_length_after,
        MEASURED_OUTPUT_TOKENS,
        journey_started_at,
    );
    let speculative_elapsed = speculative_started_at.elapsed();

    let baseline_tokens_per_second = MEASURED_OUTPUT_TOKENS as f64 / baseline_elapsed.as_secs_f64();
    let speculative_tokens_per_second =
        MEASURED_OUTPUT_TOKENS as f64 / speculative_elapsed.as_secs_f64();
    let speedup_ratio = speculative_tokens_per_second / baseline_tokens_per_second;
    let acceptance_rate = speculative_outcome.accepted_proposal_count as f64
        / speculative_outcome.proposed_token_count.max(1) as f64;

    println!();
    println!("================ DFlash proof measurements ================");
    println!("target: {}", resident_sparse_moe_model_id());
    println!(
        "draft:  {} at {}",
        ornith_dflash_draft_model_id(),
        journey.draft_directory.display()
    );
    println!(
        "prompt tokens: {context_length_tokens} | measured output: {MEASURED_OUTPUT_TOKENS} per mode | k = {SPECULATION_TOKEN_COUNT}"
    );
    println!("baseline decode:     {baseline_tokens_per_second:.2} tokens/second");
    println!("speculative decode:  {speculative_tokens_per_second:.2} tokens/second");
    println!("speedup ratio:       {speedup_ratio:.3}x");
    println!(
        "proposal acceptance: {:.1}% ({} of {} proposed across {} rounds)",
        acceptance_rate * 100.0,
        speculative_outcome.accepted_proposal_count,
        speculative_outcome.proposed_token_count,
        speculative_outcome.round_count
    );
    println!(
        "average accepted proposals per round: {:.2}",
        speculative_outcome.accepted_proposal_count as f64
            / speculative_outcome.round_count.max(1) as f64
    );
    println!("===========================================================");
}

fn load_proof_journey(journey_started_at: Instant) -> ProofJourney {
    let target_directory =
        configured_installed_model_directory_by_id(resident_sparse_moe_model_id());
    let draft_directory = target_directory
        .parent()
        .expect("the target model directory has a library parent")
        .join(ornith_dflash_draft_model_id());
    assert!(
        draft_directory.join("model.safetensors").is_file(),
        "the DFlash draft must sit beside its target in the model library"
    );

    let runtime = MlxRuntime::initialize(
        MlxMemoryLimits::new(
            PROOF_ACTIVE_MEMORY_LIMIT_BYTES,
            PROOF_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
        )
        .expect("the proof journey memory limits should be valid"),
    )
    .expect("the direct MLX runtime should initialize");

    let validated_artifact: ValidatedQwen3_5Artifact = Qwen3_5ArtifactValidator::new()
        .validate(&target_directory, 4096)
        .expect("the resident target artifact should validate");
    let tokenizer = Qwen3_5Tokenizer::from_validated_artifact(&validated_artifact)
        .expect("the target tokenizer should construct");

    let model_chunking = Qwen3_5ModelChunkingConfiguration::new(256, 0, 1, 0)
        .expect("the proof chunking configuration should validate");
    let mut disabled_attribution = PerformanceAttribution::disabled();
    let target_model = Qwen3_5ResidentModel::load_with_performance_attribution(
        runtime,
        validated_artifact,
        &target_directory,
        true,
        model_chunking,
        &mut disabled_attribution,
    )
    .expect("the resident target model should load");
    target_model
        .materialize_target_weights()
        .expect("the target weights should materialize");
    println!(
        "[dflash-proof {:>4}s] target loaded",
        journey_started_at.elapsed().as_secs()
    );

    let draft_config_bytes =
        std::fs::read(draft_directory.join("config.json")).expect("the draft config should read");
    let draft_config_document: serde_json::Value =
        serde_json::from_slice(&draft_config_bytes).expect("the draft config should parse");
    let draft_configuration = DFlashDraftConfiguration::from_config_value(&draft_config_document)
        .expect("the DFlash draft configuration should parse");
    let draft = DFlashDraftLoadedWeights::load(
        target_model.runtime(),
        &draft_directory,
        &draft_configuration,
    )
    .expect("the DFlash draft weights should load and match every declared shape");
    let draft_key_value_states =
        DFlashDraftKeyValueStates::empty(draft_configuration.layer_count() as usize);
    println!(
        "[dflash-proof {:>4}s] draft loaded ({} bound tensors, {} bytes)",
        journey_started_at.elapsed().as_secs(),
        bound_draft_tensor_count(&draft),
        draft.weights.bound_tensor_bytes()
    );

    let target_decoder_state =
        RequestDecoderStateStack::empty_from_decoder_cache_layout_with_full_attention_kv_state_growth_tokens(
            &target_model.decoder_cache_layout(),
            256,
        )
        .expect("the target decoder-state stack should allocate");

    // Ten thousand Romeo-and-Juliet tokens inside the measurement band. The
    // fixture is the shared acceptance excerpt source (the same text every
    // throughput journey reads), inlined here because the direct-MLX test
    // binary does not link the serving support module.
    let prompt_text = romeo_and_juliet_excerpt(10_000);
    let mut prompt_token_ids = tokenizer
        .encode_prompt(&prompt_text)
        .expect("the Romeo and Juliet fixture should encode");
    prompt_token_ids.truncate(10_000);
    assert!(
        (9_000..=11_000).contains(&prompt_token_ids.len()),
        "the measured prompt must stay inside the ten-thousand-token band"
    );

    let mut journey = ProofJourney {
        target_model,
        target_decoder_state,
        draft,
        draft_configuration,
        draft_key_value_states,
        prompt_token_ids,
        prefill_terminal_bonus_token_id: 0,
        draft_directory,
    };
    let prefill_terminal_bonus_token_id = prefill_and_bootstrap_draft_context(&mut journey);
    journey.prefill_terminal_bonus_token_id = prefill_terminal_bonus_token_id;
    journey
}

fn bound_draft_tensor_count(draft: &DFlashDraftLoadedWeights) -> usize {
    11 * draft.weights.layers.len() + 3
}

/// The shared Romeo-and-Juliet fixture, truncated by word count exactly as
/// the serving support helper does.
fn romeo_and_juliet_excerpt(maximum_word_count: usize) -> String {
    include_str!("../../../fixtures/model_metrics_10000_romeo_and_juliet_words.txt")
        .split_whitespace()
        .take(maximum_word_count)
        .collect::<Vec<_>>()
        .join(" ")
}

/// Runs the tap-capturing prefill in chunks and grows the draft's context
/// key-value cache from the target's hidden states. Returns the target's
/// greedy preference at the terminal position: the first bonus token.
fn prefill_and_bootstrap_draft_context(journey: &mut ProofJourney) -> u32 {
    let prefill_started_at = Instant::now();
    let prompt_token_count = journey.prompt_token_ids.len();
    let mut next_chunk_start = 0_usize;
    let mut terminal_bonus_token_id = 0_u32;
    while next_chunk_start < prompt_token_count {
        let chunk_end = (next_chunk_start + PREFILL_CHUNK_TOKENS).min(prompt_token_count);
        let is_terminal_chunk = chunk_end == prompt_token_count;
        let chunk_token_ids = &journey.prompt_token_ids[next_chunk_start..chunk_end];
        let logits_mode = if is_terminal_chunk {
            VerificationLogitsMode::LastRowOnly
        } else {
            VerificationLogitsMode::NoLogits
        };
        let mut attribution = PerformanceAttribution::disabled();
        let chunk_output = journey
            .target_model
            .build_speculative_verification_forward(
                chunk_token_ids,
                next_chunk_start as u32,
                &mut journey.target_decoder_state,
                journey.draft_configuration.target_layer_ids(),
                logits_mode,
                &mut attribution,
            )
            .expect("the tap-capturing prefill chunk should forward");
        let conditioned_states = condition_target_hidden_states(
            journey.target_model.runtime(),
            &journey.draft.weights,
            &chunk_output.hidden_state_taps,
            journey.draft_configuration.rms_norm_epsilon(),
        )
        .expect("the chunk taps should condition");
        append_conditioned_context_key_values(
            journey.target_model.runtime(),
            &journey.draft.weights,
            &journey.draft_configuration,
            &mut journey.draft_key_value_states,
            &conditioned_states,
            next_chunk_start as i32,
        )
        .expect("the draft context key values should append");
        if is_terminal_chunk {
            terminal_bonus_token_id = journey.argmax_first_row(
                chunk_output
                    .per_position_logits
                    .as_ref()
                    .expect("the terminal chunk must produce logits"),
            );
        }
        next_chunk_start = chunk_end;
        println!(
            "[dflash-proof] prefill {}/{} tokens ({:.0}%) in {:.1}s",
            chunk_end,
            prompt_token_count,
            100.0 * chunk_end as f64 / prompt_token_count as f64,
            prefill_started_at.elapsed().as_secs_f64()
        );
    }
    assert_eq!(
        journey.draft_key_value_states.offset_tokens(),
        prompt_token_count as i32,
        "the draft context cache must cover the whole prompt"
    );
    println!(
        "[dflash-proof] prefill complete in {:.1}s",
        prefill_started_at.elapsed().as_secs_f64()
    );
    terminal_bonus_token_id
}

/// Baseline: one token per forward — the cost shape of today's decode path,
/// driven through the same per-layer forward the speculative side uses so
/// the comparison is like-for-like. Every forwarded token is emitted; the
/// final forward's greedy preference becomes the next phase's bonus.
fn decode_baseline(
    journey: &mut ProofJourney,
    starting_bonus_token_id: u32,
    starting_context_tokens: u32,
    output_token_count: usize,
) -> DecodePhaseOutcome {
    let mut emitted_tokens = 0_usize;
    let mut current_token_id = starting_bonus_token_id;
    let mut current_token_position = starting_context_tokens;
    let mut attribution = PerformanceAttribution::disabled();
    let mut last_preferred_token_id = starting_bonus_token_id;
    while emitted_tokens < output_token_count {
        let forward_output = journey
            .target_model
            .build_speculative_verification_forward(
                &[current_token_id],
                current_token_position,
                &mut journey.target_decoder_state,
                &[],
                VerificationLogitsMode::AllPositions,
                &mut attribution,
            )
            .expect("the baseline one-token forward should succeed");
        last_preferred_token_id = journey.argmax_first_row(
            forward_output
                .per_position_logits
                .as_ref()
                .expect("the baseline forward must produce logits"),
        );
        current_token_id = last_preferred_token_id;
        current_token_position += 1;
        emitted_tokens += 1;
        if emitted_tokens.is_multiple_of(100) {
            println!(
                "[baseline {:>3}%] {emitted_tokens}/{output_token_count} tokens",
                100 * emitted_tokens / output_token_count
            );
        }
    }
    DecodePhaseOutcome {
        context_length_after: current_token_position,
        next_bonus_token_id: last_preferred_token_id,
        accepted_proposal_count: 0,
        proposed_token_count: 0,
        round_count: output_token_count,
    }
}

/// Speculative: draft-and-verify rounds at the configured block length.
fn decode_speculative(
    journey: &mut ProofJourney,
    starting_bonus_token_id: u32,
    starting_context_tokens: u32,
    output_token_count: usize,
    journey_started_at: Instant,
) -> DecodePhaseOutcome {
    let mut emitted_tokens = 0_usize;
    let mut accepted_proposal_count = 0_usize;
    let mut proposed_token_count = 0_usize;
    let mut round_count = 0_usize;
    let mut bonus_token_id = starting_bonus_token_id;
    let mut context_length_tokens = starting_context_tokens;
    let mut attribution = PerformanceAttribution::disabled();
    while emitted_tokens < output_token_count {
        let remaining_tokens = output_token_count - emitted_tokens;
        let speculation_token_count = SPECULATION_TOKEN_COUNT.min(remaining_tokens);
        let round_request = SpeculationRoundRequest {
            request_id: RequestId::new(0),
            bonus_token_id,
            speculation_token_count,
            context_length_tokens,
        };
        let round_outcome = run_speculative_round(
            &journey.target_model,
            &mut journey.target_decoder_state,
            &journey.draft.weights,
            &journey.draft_configuration,
            &mut journey.draft_key_value_states,
            &round_request,
            &mut attribution,
        )
        .expect("the speculative round should succeed");
        accepted_proposal_count += round_outcome.accepted_proposal_count;
        proposed_token_count += speculation_token_count;
        round_count += 1;
        emitted_tokens += round_outcome.emitted_token_ids.len().min(remaining_tokens);
        context_length_tokens = round_outcome.context_length_after_round;
        bonus_token_id = round_outcome.next_bonus_token_id;
        if emitted_tokens.is_multiple_of(100) {
            let elapsed_seconds = journey_started_at.elapsed().as_secs_f64();
            let eta_seconds = elapsed_seconds * (output_token_count - emitted_tokens) as f64
                / emitted_tokens as f64;
            println!(
                "[speculative {:>3}%] {emitted_tokens}/{output_token_count} tokens | eta {:.0}s",
                100 * emitted_tokens / output_token_count,
                eta_seconds
            );
        }
    }
    DecodePhaseOutcome {
        context_length_after: context_length_tokens,
        next_bonus_token_id: bonus_token_id,
        accepted_proposal_count,
        proposed_token_count,
        round_count,
    }
}

impl ProofJourney {
    fn argmax_first_row(&self, logits: &MlxArray) -> u32 {
        let runtime = self.target_model.runtime();
        let row_argmax = runtime
            .argmax_axis(logits, -1)
            .expect("the logits argmax should build");
        runtime
            .evaluate_arrays(&[&row_argmax])
            .expect("the argmax should evaluate");
        let first_row = runtime
            .slice(&row_argmax, &[0, 0], &[1, 1], &[1, 1])
            .expect("the first argmax row should slice");
        first_row
            .item_u32()
            .expect("the argmax token should read to the host")
    }
}
