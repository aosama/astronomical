//! Gate 1 of speculative decoding on the dense Qwen3.8 27B: how many tokens a
//! verify round keeps with Astronomical's DFlash2 drafter on the real pair
//! (`TensorFold/Qwen3.8-27B-MLX-4bit` and `z-lab/Qwen3.8-27B-DFlash2`).
//!
//! Kept per round = tokens emitted / verify rounds, the bonus token included,
//! as the drafter's model card defines it. z-lab's own reference loop kept
//! 6.07 on the code prompt at temperature 0.7 and 3.00 on the chat prompt at
//! 1.0 on this pair (single path, 256 tokens, thinking off), so the floors
//! below sit about 10% under those. Greedy runs only as a labelled diagnostic.
//!
//! This measures acceptance, not speed: a refused draft restores the target's
//! decoder state and runs the kept rows again, the slow rewind Gate 2 replaces.

use std::time::Instant;

use astronomical_ipc_protocol::RequestId;
use astronomical_model_serving::speculative_decoding::dflash2_drafter::{
    DFlash2DraftTarget, DFlash2DrafterConfiguration, DFlash2DrafterWeights, DrafterContextStates,
    PathChoice, bind_drafter_weights, drafted_hidden_rows, select_path,
};
use astronomical_model_serving::speculative_decoding::verification_decision::{
    DraftedSlot, VerificationDecision, decide_by_rejection_sampling, decide_by_token_equality,
};
use astronomical_model_serving::{
    PerformanceAttribution, Qwen3_5ArtifactValidator, Qwen3_5ModelChunkingConfiguration,
    Qwen3_5ResidentModel, Qwen3_5Tokenizer, RequestDecoderStateStack, VerificationLogitsMode,
};
use astronomical_runtime_integration::{MlxMemoryLimits, MlxRuntime, MlxSafetensors};
use mlx_bridge::{MlxArray, MlxDtype};
use serial_test::serial;

use super::dflash2_journey_support::{
    UniformStream, host_values, huggingface_hub_snapshot, tap_rows,
};

const TARGET_REPOSITORY: &str = "TensorFold/Qwen3.8-27B-MLX-4bit";
const DRAFTER_REPOSITORY: &str = "z-lab/Qwen3.8-27B-DFlash2";
const CODE_PROMPT: &str =
    "Write a short Python function that computes the Fibonacci sequence and explain it.";
const CHAT_PROMPT: &str = "Explain how matrix multiplication uses a GPU in plain English, then give a small numerical example.";
const OUTPUT_TOKENS: usize = 256;
/// Qwen3.8's recommended sampling, as users run it and as the reference ran.
const TOP_P_THOUSANDTHS: u16 = 950;
const TOP_K: u16 = 20;
/// 15 GiB of target weights, 3.6 GiB of bf16 drafter, and working room.
const ACTIVE_MEMORY_LIMIT_BYTES: usize = 26 * 1024 * 1024 * 1024;
const MLX_BUFFER_CACHE_LIMIT_BYTES: usize = 2 * 1024 * 1024 * 1024;

/// One measured cell: the prompt, the temperature (0 = greedy diagnostic), and
/// the least kept-per-round that passes.
struct Cell {
    label: &'static str,
    prompt: &'static str,
    temperature_thousandths: u16,
    minimum_kept_per_round: f64,
}

const CELLS: [Cell; 4] = [
    Cell {
        label: "code",
        prompt: CODE_PROMPT,
        temperature_thousandths: 700,
        minimum_kept_per_round: 5.5,
    },
    Cell {
        label: "chat",
        prompt: CHAT_PROMPT,
        temperature_thousandths: 1_000,
        minimum_kept_per_round: 2.7,
    },
    Cell {
        label: "code greedy (diagnostic)",
        prompt: CODE_PROMPT,
        temperature_thousandths: 0,
        minimum_kept_per_round: 5.7,
    },
    Cell {
        label: "chat greedy (diagnostic)",
        prompt: CHAT_PROMPT,
        temperature_thousandths: 0,
        minimum_kept_per_round: 2.8,
    },
];

struct JourneyPair {
    target: Qwen3_5ResidentModel,
    tokenizer: Qwen3_5Tokenizer,
    drafter_configuration: DFlash2DrafterConfiguration,
    drafter_weights: DFlash2DrafterWeights,
    /// Keeps the drafter's lazily mapped tensors alive.
    _drafter_checkpoint: MlxSafetensors,
}

#[test]
#[ignore = "loads the real 27B and its DFlash2 drafter; run through cargo measure-dflash2-kept-per-round"]
#[serial]
fn should_keep_as_many_drafted_tokens_per_round_as_z_labs_reference() {
    let journey_started_at = Instant::now();
    let pair = load_pair();
    eprintln!(
        "[dflash2-gate1] pair loaded in {} s",
        journey_started_at.elapsed().as_secs()
    );
    let mut failures = Vec::new();
    for (cell_index, cell) in CELLS.iter().enumerate() {
        let cell_started_at = Instant::now();
        let (rounds, emitted) = run_cell(&pair, cell);
        let kept_per_round = emitted as f64 / rounds as f64;
        let elapsed_seconds = journey_started_at.elapsed().as_secs_f64();
        let remaining_seconds =
            elapsed_seconds / (cell_index + 1) as f64 * (CELLS.len() - cell_index - 1) as f64;
        eprintln!(
            "[dflash2-gate1 {}/{} ETA {remaining_seconds:.0} s] {} T={:.1}: {kept_per_round:.2} kept per round \
             ({emitted} tokens in {rounds} rounds, {:.0} s)",
            cell_index + 1,
            CELLS.len(),
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

fn load_pair() -> JourneyPair {
    let target_directory = huggingface_hub_snapshot(TARGET_REPOSITORY);
    let drafter_directory = huggingface_hub_snapshot(DRAFTER_REPOSITORY);
    let runtime = MlxRuntime::initialize(
        MlxMemoryLimits::new(ACTIVE_MEMORY_LIMIT_BYTES, MLX_BUFFER_CACHE_LIMIT_BYTES)
            .expect("memory limits are valid"),
    )
    .expect("the MLX runtime initializes");
    let artifact = Qwen3_5ArtifactValidator::new()
        .validate(&target_directory, 4096)
        .expect("the 27B artifact validates");
    let tokenizer =
        Qwen3_5Tokenizer::from_validated_artifact(&artifact).expect("the tokenizer builds");
    let chunking =
        Qwen3_5ModelChunkingConfiguration::new(256, 0, 0).expect("the chunking is valid");
    let mut attribution = PerformanceAttribution::disabled();
    let mut target = Qwen3_5ResidentModel::load_with_performance_attribution(
        runtime,
        artifact,
        &target_directory,
        true,
        chunking,
        &mut attribution,
    )
    .expect("the 27B loads in the resident engine");
    target
        .materialize_complete_weights_matching_production()
        .expect("the 27B weights materialize");
    let config_document: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(drafter_directory.join("config.json"))
            .expect("the drafter config reads"),
    )
    .expect("the drafter config parses");
    let drafter_configuration = DFlash2DrafterConfiguration::from_config_document(&config_document)
        .expect("the drafter configuration is accepted");
    let drafter_checkpoint = target
        .runtime()
        .load_safetensors(
            std::fs::File::open(drafter_directory.join("model.safetensors"))
                .expect("the drafter opens"),
            None,
        )
        .expect("the drafter loads");
    let drafter_weights = bind_drafter_weights(
        target.runtime(),
        &drafter_checkpoint,
        &drafter_configuration,
    )
    .expect("every drafter tensor binds");
    JourneyPair {
        target,
        tokenizer,
        drafter_configuration,
        drafter_weights,
        _drafter_checkpoint: drafter_checkpoint,
    }
}

fn prompt_token_ids(pair: &JourneyPair, prompt: &str) -> Vec<u32> {
    let command = astronomical_ipc_protocol::ChatGenerationCommand {
        request_id: RequestId::new(0),
        model: TARGET_REPOSITORY.to_owned(),
        messages: vec![astronomical_ipc_protocol::ChatMessage::User {
            content: prompt.to_owned(),
            images: Vec::new(),
        }],
        tools: Vec::new(),
        tool_choice: astronomical_ipc_protocol::ChatToolChoice::None,
        settings: astronomical_ipc_protocol::ChatGenerationSettings {
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
fn run_cell(pair: &JourneyPair, cell: &Cell) -> (usize, usize) {
    let tap_layer_ids = &pair.drafter_configuration.target_tap_layer_ids;
    let prompt = prompt_token_ids(pair, cell.prompt);
    let mut decoder_state = RequestDecoderStateStack::empty_from_astro_decoder_state_cache_layout_with_full_attention_kv_state_growth_tokens(
        &pair.target.astro_decoder_state_cache_layout(),
        256,
    )
    .expect("the decoder state allocates");
    let mut uniforms = UniformStream::seeded(1234);
    let mut random_state = pair
        .target
        .runtime()
        .random_key(1234)
        .expect("a key builds");
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
    let first_logits = prefill
        .per_position_logits
        .expect("the last row has logits");
    let mut round = RoundState {
        last_token_id: decide(pair, cell, &[], &first_logits, None, &mut uniforms).next_token_id,
        fed_token_count: prompt.len() as u32,
        pending_tap_rows: tap_rows(
            pair.target.runtime(),
            &prefill.hidden_state_taps,
            prompt.len() as i32,
        ),
        context_states: DrafterContextStates::new(&pair.drafter_configuration, 0),
    };
    let (mut rounds, mut emitted) = (0, 0);
    while emitted < OUTPUT_TOKENS {
        emitted += run_round(
            pair,
            cell,
            &mut round,
            &mut decoder_state,
            &mut random_state,
            &mut uniforms,
        );
        rounds += 1;
    }
    (rounds, emitted)
}

struct RoundState {
    last_token_id: u32,
    /// Tokens already in the target's decoder state; `last_token_id` is not.
    fed_token_count: u32,
    /// Fused-tap input rows the drafter has not absorbed yet.
    pending_tap_rows: MlxArray,
    context_states: DrafterContextStates,
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
) -> usize {
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
    let anchor = runtime
        .array_from_u32(&[round.last_token_id], &[1])
        .expect("the anchor builds");
    let mut path_choice = if cell.temperature_thousandths == 0 {
        PathChoice::Greedy
    } else {
        PathChoice::Sampled {
            temperature: f32::from(cell.temperature_thousandths) / 1_000.0,
            random_state,
        }
    };
    let path = select_path(
        runtime,
        &pair.drafter_weights.candidate_selector,
        configuration,
        &hidden,
        &draft_logits,
        &anchor,
        &mut path_choice,
    )
    .expect("the selector walks a path");
    let drafted_ids: Vec<u32> = host_values(runtime, &path.token_ids)
        .iter()
        .map(|value| *value as u32)
        .collect();
    let checkpoint = decoder_state
        .checkpoint()
        .expect("the decoder state checkpoints");
    let mut verified_ids = vec![round.last_token_id];
    verified_ids.extend_from_slice(&drafted_ids);
    let verification = pair
        .target
        .build_speculative_verification_forward(
            &verified_ids,
            round.fed_token_count,
            decoder_state,
            &configuration.target_tap_layer_ids,
            VerificationLogitsMode::AllPositions,
            &mut PerformanceAttribution::disabled(),
        )
        .expect("the target verifies the block");
    let logits = verification
        .per_position_logits
        .expect("every row has logits");
    let drafter_distribution = match &path.candidate_probabilities {
        Some(probabilities) => Some(DrafterDistribution {
            candidate_ids: host_values(runtime, &path.candidate_ids)
                .iter()
                .map(|value| *value as u32)
                .collect(),
            candidate_probabilities: host_values(runtime, probabilities),
        }),
        None => None,
    };
    let decision = decide(
        pair,
        cell,
        &drafted_ids,
        &logits,
        drafter_distribution.as_ref(),
        uniforms,
    );
    settle_round(
        pair,
        round,
        decoder_state,
        checkpoint,
        &verified_ids,
        &verification.hidden_state_taps,
        decision,
    );
    decision.kept_draft_count + 1
}

/// Keeps the accepted rows in the target and the drafter's pending context.
fn settle_round(
    pair: &JourneyPair,
    round: &mut RoundState,
    decoder_state: &mut RequestDecoderStateStack,
    checkpoint: astronomical_model_serving::RequestDecoderStateStackCheckpoint,
    verified_ids: &[u32],
    verification_taps: &[MlxArray],
    decision: VerificationDecision,
) {
    let kept_row_count = decision.kept_draft_count + 1;
    if kept_row_count < verified_ids.len() {
        decoder_state
            .restore_checkpoint(checkpoint)
            .expect("the decoder state restores");
        pair.target
            .build_speculative_verification_forward(
                &verified_ids[..kept_row_count],
                round.fed_token_count,
                decoder_state,
                &pair.drafter_configuration.target_tap_layer_ids,
                VerificationLogitsMode::NoLogits,
                &mut PerformanceAttribution::disabled(),
            )
            .expect("the kept rows run again");
    }
    round.pending_tap_rows = tap_rows(
        pair.target.runtime(),
        verification_taps,
        kept_row_count as i32,
    );
    round.fed_token_count += kept_row_count as u32;
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
    let probabilities = pair
        .target
        .sampling_probability_rows(
            logits,
            cell.temperature_thousandths,
            TOP_P_THOUSANDTHS,
            TOP_K,
        )
        .expect("the target's sampling law applies");
    let probability_values = host_values(
        runtime,
        &runtime
            .astype(&probabilities, MlxDtype::Float32)
            .expect("probabilities convert"),
    );
    let vocabulary_size = probability_values.len() / row_count;
    let rows: Vec<&[f32]> = probability_values
        .chunks(vocabulary_size)
        .skip(row_count - drafted_ids.len() - 1)
        .collect();
    let candidate_count = pair.drafter_configuration.selector_candidate_count as usize;
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
