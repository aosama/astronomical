//! DFlash speculative-decoding proof: acceptance rate and decode throughput.
//!
//! This is the diagnostic instrument for the DFlash compartment, which is
//! disabled by design and unreachable from engine wiring (see the
//! speculative-decode lifecycle module): the journey loads the real resident
//! target (the OptiQ 4-bit Ornith-1.5-35B-A3B from the e2e registry) beside
//! the real DFlash draft (resolved as the draft's library sibling directory,
//! because a draft artifact is intentionally invisible to executable model
//! discovery), bootstraps the draft's context key-value cache from the
//! target's hidden-state taps, then measures decode two ways on the same
//! continued context: one token per forward (baseline) and draft-and-verify,
//! both under the scenario's verification temperature. The committed
//! scenario is the realistic serving point — temperature 1.0 (the artifact
//! default users get), block length 4 (the historical engine
//! configuration).
//!
//! The journey reports acceptance rate, per-mode tokens per second, and the
//! speedup ratio. It asserts only accounting invariants — token counts and
//! context bookkeeping — because the performance numbers are evidence, not
//! thresholds, and they decide whether the integration proceeds.
//!
//! Run via `cargo test --release -p astronomical-acceptance --features
//! direct-mlx --test direct_mlx
//! should_measure_dflash_speculative_acceptance_and_decode_throughput --
//! --ignored --nocapture --test-threads=1`.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use astronomical_ipc_protocol::RequestId;
use astronomical_model_serving::{
    DFlashDraftConfiguration, DFlashDraftKeyValueStates, DFlashDraftLoadedWeights,
    PerformanceAttribution, Qwen3_5ArtifactValidator, Qwen3_5ModelChunkingConfiguration,
    Qwen3_5ResidentModel, Qwen3_5Tokenizer, RequestDecoderStateStack, SpeculationRoundRequest,
    ValidatedQwen3_5Artifact, VerificationLogitsMode, VerificationTokenChoice,
    append_conditioned_context_key_values, condition_target_hidden_states, run_speculative_round,
};
use astronomical_runtime_integration::{MlxMemoryLimits, MlxRuntime};
use mlx_c_rust::MlxArray;
use serial_test::serial;
use tokio::time::timeout;

use super::dflash_proof_scenario::{
    ProofFixtureKind, ProofScenario, load_proof_scenario, progressive_source_code_excerpt,
    romeo_and_juliet_excerpt,
};
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

/// The warmup contract for throughput journeys: untimed decode tokens.
const WARMUP_OUTPUT_TOKENS: usize = 50;
/// The serving default top-k the classic path applies alongside
/// temperature.
const VERIFICATION_TOP_K: u16 = 40;
/// The serving default nucleus the classic path applies alongside
/// temperature.
const VERIFICATION_TOP_P_THOUSANDTHS: u16 = 1_000;

/// The journey's random-state seed for sampled verification; fixed so
/// reruns of the same temperature are comparable.
const VERIFICATION_SAMPLING_SEED: u64 = 20_260_925;
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
    /// The measurement scenario this journey runs; the decode phases read
    /// their block length and verification mode from it.
    scenario: ProofScenario,
    prompt_token_ids: Vec<u32>,
    /// Detokenizes the measured phases' output for the regime-classification
    /// tail print.
    tokenizer: Qwen3_5Tokenizer,
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
    /// Every token the phase emitted, in order. The speculative phase's
    /// tail is detokenized after measurement so a low-entropy repetition
    /// loop cannot masquerade as high acceptance.
    emitted_token_ids: Vec<u32>,
}

async fn measure_dflash_speculative_acceptance_and_decode_throughput() {
    let _direct_mlx_guard = direct_mlx_test_guard().await;
    let journey_started_at = Instant::now();
    let mut journey = load_proof_journey(journey_started_at);

    let context_length_tokens = journey.prompt_token_ids.len() as u32;
    let measured_output_tokens = journey.scenario.measured_output_tokens;

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

    // Speculative first: only it requires the draft and target caches to
    // stay in lockstep, and every phase before it preserved that sync.
    // Baseline runs last because one-token forwards advance only the
    // target's cache — the draft would be left behind and could not be
    // cheaply re-synced without contaminating the baseline timing.
    println!(
        "[dflash-proof {:>4}s] measuring speculative: {measured_output_tokens} tokens at k={}",
        journey_started_at.elapsed().as_secs(),
        journey.scenario.speculation_tokens
    );
    let speculative_started_at = Instant::now();
    let speculative_outcome = decode_speculative(
        &mut journey,
        warmup_bonus_token_id,
        warmup_outcome.context_length_after,
        measured_output_tokens,
        journey_started_at,
    );
    let speculative_elapsed = speculative_started_at.elapsed();

    println!(
        "[dflash-proof {:>4}s] measuring baseline: {measured_output_tokens} one-token-at-a-time tokens",
        journey_started_at.elapsed().as_secs()
    );
    let baseline_started_at = Instant::now();
    let baseline_outcome = decode_baseline(
        &mut journey,
        speculative_outcome.next_bonus_token_id,
        speculative_outcome.context_length_after,
        measured_output_tokens,
    );
    let baseline_elapsed = baseline_started_at.elapsed();

    let baseline_tokens_per_second = measured_output_tokens as f64 / baseline_elapsed.as_secs_f64();
    let speculative_tokens_per_second =
        measured_output_tokens as f64 / speculative_elapsed.as_secs_f64();
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
        "prompt tokens: {context_length_tokens} | measured output: {measured_output_tokens} per mode | k = {}",
        journey.scenario.speculation_tokens
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
    println!(
        "final context length after both phases: {} tokens",
        baseline_outcome.context_length_after
    );
    println!("===========================================================");

    // Regime classification: detokenize the speculative phase's tail. The
    // acceptance number alone cannot distinguish healthy low-entropy
    // generation (structured code continuation) from a degenerate
    // repetition loop, and the two flatter throughput differently.
    let tail_token_count = 200.min(speculative_outcome.emitted_token_ids.len());
    let tail_start_index = speculative_outcome.emitted_token_ids.len() - tail_token_count;
    let mut tail_decoder = journey.tokenizer.incremental_decoder();
    let mut tail_text = String::new();
    for tail_token_id in &speculative_outcome.emitted_token_ids[tail_start_index..] {
        if let Some(piece) = tail_decoder
            .push_token(*tail_token_id)
            .expect("the speculative tail token should decode")
        {
            tail_text.push_str(&piece);
        }
    }
    if let Some(final_piece) = tail_decoder
        .finish()
        .expect("the speculative tail decoder should finish")
    {
        tail_text.push_str(&final_piece);
    }
    println!();
    println!("---- speculative tail: last {tail_token_count} tokens ----");
    println!("{tail_text}");
    println!("---------------------------------------------------------");
}

fn load_proof_journey(journey_started_at: Instant) -> ProofJourney {
    let scenario = load_proof_scenario();
    let target_directory =
        configured_installed_model_directory_by_id(resident_sparse_moe_model_id());
    let draft_directory = resolve_draft_library_directory(&target_directory);
    assert!(
        draft_directory.join("model.safetensors").is_file(),
        "the resolved DFlash draft directory must carry model.safetensors"
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
    let mut target_model = Qwen3_5ResidentModel::load_with_performance_attribution(
        runtime,
        validated_artifact,
        &target_directory,
        true,
        model_chunking,
        &mut disabled_attribution,
    )
    .expect("the resident target model should load");
    target_model
        .materialize_complete_weights_matching_production()
        .expect("the target weights and resident experts should materialize");
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

    // Romeo-and-Juliet tokens inside the measurement band, sized by the
    // context sweep knob. The fixture is the shared acceptance excerpt
    // source (the same text every throughput journey reads), inlined here
    // because the direct-MLX test binary does not link the serving support
    // module; the fifty-thousand-word file backs context sweeps beyond
    // ten thousand tokens.
    let requested_context_token_count = scenario.context_tokens;
    let fixture_text = match scenario.fixture {
        ProofFixtureKind::SourceCode => progressive_source_code_excerpt(50_000),
        ProofFixtureKind::RomeoAndJuliet => romeo_and_juliet_excerpt(50_000),
    };
    let fixture_label = match scenario.fixture {
        ProofFixtureKind::SourceCode => "progressive source code",
        ProofFixtureKind::RomeoAndJuliet => "Romeo and Juliet",
    };
    let mut prompt_token_ids = if scenario.chat_formatted {
        // The exact worker tokenization: the chat template applied to the
        // fixture as one user message, isolating the chat-format variable
        // against the raw-continuation default.
        let chat_generation_command = astronomical_ipc_protocol::ChatGenerationCommand {
            request_id: RequestId::new(0),
            model: resident_sparse_moe_model_id().to_owned(),
            messages: vec![astronomical_ipc_protocol::ChatMessage::User {
                content: format!(
                    "{fixture_text}\n\nContinue this text exactly where it \
                     stops, in the same voice, without any commentary."
                ),
                images: Vec::new(),
            }],
            tools: Vec::new(),
            tool_choice: astronomical_ipc_protocol::ChatToolChoice::None,
            settings: astronomical_ipc_protocol::ChatGenerationSettings {
                max_output_tokens: 4_096,
                temperature_thousandths: Some(0),
                top_p_thousandths: None,
                repetition_penalty_thousandths: None,
                seed: None,
                thinking_budget: Some(0),
            },
            structured_generation: None,
        };
        let inference_request = tokenizer
            .prepare_chat(&chat_generation_command, false)
            .expect("the chat-formatted fixture should tokenize");
        inference_request.input_token_ids().to_vec()
    } else {
        tokenizer
            .encode_prompt(&fixture_text)
            .expect("the measured fixture should encode")
    };
    prompt_token_ids.truncate(requested_context_token_count);
    assert!(
        !prompt_token_ids.is_empty() && prompt_token_ids.len() <= requested_context_token_count,
        "the measured prompt must hold between one and {requested_context_token_count} tokens"
    );
    println!(
        "[dflash-proof] scenario {:?} | fixture={} measured_context_tokens={}",
        scenario,
        fixture_label,
        prompt_token_ids.len(),
    );

    let mut journey = ProofJourney {
        target_model,
        target_decoder_state,
        draft,
        draft_configuration,
        draft_key_value_states,
        scenario,
        prompt_token_ids,
        tokenizer,
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

/// Resolves the draft's directory by leaf id across the library's
/// organization directories: the target and its draft may ship under
/// different organizations (the OptiQ target sits under one, the draft
/// under another), so only the library root is a safe search base.
fn resolve_draft_library_directory(target_directory: &std::path::Path) -> PathBuf {
    let library_root = target_directory
        .parent()
        .and_then(|organization_directory| organization_directory.parent())
        .expect("the target model directory must sit inside a library organization directory");
    let draft_leaf_id = ornith_dflash_draft_model_id();
    let organization_directories = std::fs::read_dir(library_root)
        .expect("the model library root should list")
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.path().is_dir());
    for organization_entry in organization_directories {
        let candidate_directory = organization_entry.path().join(draft_leaf_id);
        if candidate_directory.join("model.safetensors").is_file() {
            return candidate_directory;
        }
    }
    panic!(
        "the DFlash draft {draft_leaf_id} must be installed under one organization directory of {}",
        library_root.display()
    );
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
        emitted_token_ids: Vec::new(),
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
    let mut phase_emitted_token_ids: Vec<u32> = Vec::new();
    let mut listed_new_token_count = 0_usize;
    let mut bonus_token_id = starting_bonus_token_id;
    let mut context_length_tokens = starting_context_tokens;
    let mut attribution = PerformanceAttribution::disabled();
    // Greedy verification drafts against argmax; sampled verification draws
    // one tempered target sample per position through the same transform
    // the classic path applies, from a fixed seed so temperature reruns
    // are comparable.
    let target_token_choice = match journey.scenario.temperature_thousandths {
        0 => VerificationTokenChoice::HighestLogit,
        temperature_thousandths => VerificationTokenChoice::TopKTopP {
            temperature_thousandths,
            top_k: VERIFICATION_TOP_K,
            top_p_thousandths: VERIFICATION_TOP_P_THOUSANDTHS,
        },
    };
    let mut random_state = match target_token_choice.consumes_random_state() {
        true => Some(
            journey
                .target_model
                .runtime()
                .random_key(VERIFICATION_SAMPLING_SEED)
                .expect("the sampled journey should build its random state"),
        ),
        false => None,
    };
    println!(
        "[speculative] verification mode: {target_token_choice:?} at block length {}",
        journey.scenario.speculation_tokens
    );
    while emitted_tokens < output_token_count {
        let remaining_tokens = output_token_count - emitted_tokens;
        let speculation_token_count = journey.scenario.speculation_tokens.min(remaining_tokens);
        let round_request = SpeculationRoundRequest {
            request_id: RequestId::new(0),
            bonus_token_id,
            speculation_token_count,
            context_length_tokens,
            target_token_choice,
        };
        let round_outcome = run_speculative_round(
            &journey.target_model,
            &mut journey.target_decoder_state,
            &journey.draft.weights,
            &journey.draft_configuration,
            &mut journey.draft_key_value_states,
            &round_request,
            &mut random_state,
            &mut attribution,
        )
        .expect("the speculative round should succeed");
        // Every round's emitted list repeats the previous round's last
        // token as its bonus seed. The engine emits that boundary token
        // exactly once (the queue tail becomes the next round's bonus);
        // the harness must deduplicate identically, or it double-counts
        // one token per round and inflates throughput by up to the round
        // count — which is exactly what every earlier report of this
        // journey did.
        if phase_emitted_token_ids.is_empty() {
            phase_emitted_token_ids.extend_from_slice(&round_outcome.emitted_token_ids);
        } else {
            phase_emitted_token_ids.extend_from_slice(&round_outcome.emitted_token_ids[1..]);
        }
        accepted_proposal_count += round_outcome.accepted_proposal_count;
        proposed_token_count += speculation_token_count;
        round_count += 1;
        // New unique tokens this round: the emitted list minus the bonus
        // seed, which the previous round already counted.
        let round_new_token_count = round_outcome.emitted_token_ids.len() - 1;
        listed_new_token_count += round_new_token_count;
        emitted_tokens += round_new_token_count.min(remaining_tokens);
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
    // Accounting invariant: every round lists exactly (its emitted list
    // length minus the bonus seed) new tokens, and only the final round may
    // be truncated by the remaining-token clamp. The historical double-count
    // bug credited the full list length per round — inflating the count by
    // the round count — and would fail this bound immediately.
    assert!(
        listed_new_token_count >= emitted_tokens
            && listed_new_token_count - emitted_tokens <= journey.scenario.speculation_tokens,
        "speculative accounting must satisfy listed-new = sum(emitted list length - 1), differing from the emitted count only by the final round's clamp: listed {listed_new_token_count}, emitted {emitted_tokens}, rounds {round_count}"
    );
    DecodePhaseOutcome {
        context_length_after: context_length_tokens,
        next_bonus_token_id: bonus_token_id,
        accepted_proposal_count,
        proposed_token_count,
        round_count,
        emitted_token_ids: phase_emitted_token_ids,
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
