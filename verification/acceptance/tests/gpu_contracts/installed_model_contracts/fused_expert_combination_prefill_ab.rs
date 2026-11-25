//! Resident MoE prefill A/B: the two-step expert combination against the fused
//! one, on the same loaded weights in one process.
//!
//! Arm A, two-step: the sorted expert weighted-sum kernel followed by MLX's
//! compiled sigmoid-multiply-add graph for the shared expert. Arm B, fused:
//! both in one Metal pass, which the resident loader turns on when its
//! capability probe passes. Measured from the raw model forward - no daemon,
//! no IPC, no memory admission, no chunk planner - because this is the lowest
//! layer the kernel changes.
//!
//! Rungs: one prompt chunk of 2,048, 3,072 and 4,096 tokens, then the whole
//! Shakespeare prompt (about 10,000 tokens) in 2,048-token chunks, the input
//! size the repository's performance rule asks for. Prefill only: the kernel
//! runs on prompt chunks and one-token decode never reaches it, so no output
//! tokens are generated.
//!
//! Per rung: one warmup per arm that also checks parity (the final chunk runs
//! through `forward_chunk` and both arms' logits must be equal), then pairs
//! whose order alternates (A,B then B,A) so drift and order effects land on
//! both arms. Min, median and max per arm are reported.
//!
//! Run with `cargo measure-fused-expert-combination-prefill` (see
//! `.cargo/config.toml`), on an optimized build with the Development daemon
//! stopped. The xtask case budget bounds the run: the forwards block the async
//! test body, so an in-test timeout could never fire.

use std::time::Instant;

use config::AstronomicalConfig;
use mlx_bridge::MlxArray;
use model_serving::{
    Qwen3_5ArtifactValidator, Qwen3_5ResidentModel, Qwen3_5Tokenizer,
    TextModelChunkingConfiguration,
};
use runtime_integration::{MlxMemoryLimits, MlxRuntime};
use serial_test::serial;

const PROBE_NAME: &str = "fused-expert-combination-prefill-ab";

/// The resident MoE model the fused kernel was built for.
const MODEL_ID: &str = "Ornith-1.5-35B-A3B-OptiQ-4bit-vision-DFLASH2";

/// Single-chunk rungs, in tokens.
const SINGLE_CHUNK_SIZES: [usize; 3] = [2_048, 3_072, 4_096];

/// Chunk size of the whole-prompt rung: the resident engine's prefill chunk.
const WHOLE_PROMPT_CHUNK_TOKENS: usize = 2_048;

/// Timed pairs per rung after the warmups; the order alternates per pair.
const MEASURED_PAIRS_PER_RUNG: usize = 6;

/// The measured context this artifact validates under this A/B.
const MAXIMUM_OUTPUT_TOKENS: u32 = 10_240;

/// The production standard worker chunking tuple the daemon resolves for the
/// resident engine; the raw forwards must run with the same shapes.
const FULL_ATTENTION_KV_GROWTH_TOKENS: u32 = 256;
const PREFILL_GRAPH_SUBMISSION_LAYER_INTERVAL: u32 = 0;
const SSD_PAGING_GENERATION_LAYER_INTERVAL: u32 = 3;

/// A 2 GB reserve under a 36 GB ceiling so resident materialization and the
/// request-local decoder state do not evict weights mid-run.
const AB_EFFECTIVE_MLX_CEILING_BYTES: usize = 36 * 1_073_741_824 - 2 * 1_073_741_824;

/// Which MoE combination path a prefill runs through.
#[derive(Clone, Copy)]
enum CombinationArm {
    TwoStep,
    Fused,
}

impl CombinationArm {
    const fn label(self) -> &'static str {
        match self {
            Self::TwoStep => "A_two_step",
            Self::Fused => "B_fused",
        }
    }
}

/// One rung: the prompt prefix it prefills and the chunk length it uses.
struct PrefillRung<'tokens> {
    rung_label: String,
    prompt_token_ids: &'tokens [u32],
    chunk_tokens: usize,
}

/// Counts forwards across the whole run so every progress line carries an ETA.
struct AbProgress {
    started_at: Instant,
    completed_forward_count: usize,
    planned_forward_count: usize,
}

impl AbProgress {
    fn record_forward(&mut self) -> String {
        self.completed_forward_count += 1;
        let elapsed_seconds = self.started_at.elapsed().as_secs_f64();
        let remaining_forward_count = self.planned_forward_count - self.completed_forward_count;
        let eta_seconds =
            elapsed_seconds / self.completed_forward_count as f64 * remaining_forward_count as f64;
        format!(
            "completed={} of {} percent={:.0} eta_seconds={eta_seconds:.0}",
            self.completed_forward_count,
            self.planned_forward_count,
            self.completed_forward_count as f64 * 100.0 / self.planned_forward_count as f64
        )
    }
}

#[tokio::test]
#[ignore = "loads the resident MoE model and A/B times the fused expert combination"]
#[serial]
async fn should_report_fused_expert_combination_prefill_ab_across_chunk_sizes() {
    let _gpu_exclusive = crate::common::direct_mlx_test_guard().await;
    let (mut model, prompt_token_ids) = load_resident_model_and_prompt();
    let mut prefill_rungs = Vec::new();
    for chunk_size in SINGLE_CHUNK_SIZES {
        prefill_rungs.push(PrefillRung {
            rung_label: format!("single_chunk_{chunk_size}"),
            prompt_token_ids: &prompt_token_ids[..chunk_size],
            chunk_tokens: chunk_size,
        });
    }
    prefill_rungs.push(PrefillRung {
        rung_label: format!("whole_prompt_{}_in_chunks", prompt_token_ids.len()),
        prompt_token_ids: &prompt_token_ids,
        chunk_tokens: WHOLE_PROMPT_CHUNK_TOKENS,
    });
    let mut ab_progress = AbProgress {
        started_at: Instant::now(),
        completed_forward_count: 0,
        planned_forward_count: prefill_rungs.len() * (2 + 2 * MEASURED_PAIRS_PER_RUNG),
    };
    for prefill_rung in &prefill_rungs {
        report_prefill_ab_for_rung(&mut model, prefill_rung, &mut ab_progress);
    }
    eprintln!("[{PROBE_NAME}] status=success");
}

fn load_resident_model_and_prompt() -> (Qwen3_5ResidentModel, Vec<u32>) {
    let development_config = AstronomicalConfig::load_from_development_location()
        .expect("the Development configuration should load for the prefill A/B");
    let model_directory =
        crate::common::configured_discovered_model_by_id(&development_config, MODEL_ID)
            .model_directory;
    eprintln!("[{PROBE_NAME}] status=start phase=artifact_validation model={MODEL_ID}");
    let validated_artifact = Qwen3_5ArtifactValidator::new()
        .validate(&model_directory, MAXIMUM_OUTPUT_TOKENS)
        .expect("the resident MoE artifact should pass validation");
    let tokenizer = Qwen3_5Tokenizer::from_validated_artifact(&validated_artifact)
        .expect("the tokenizer should load from the validated artifact");
    let prompt_token_ids = shakespeare_prompt_token_ids(&tokenizer);
    assert!(
        prompt_token_ids.len() >= SINGLE_CHUNK_SIZES[SINGLE_CHUNK_SIZES.len() - 1],
        "the Shakespeare prompt must reach the largest single chunk"
    );
    let model_chunking = TextModelChunkingConfiguration::new(
        FULL_ATTENTION_KV_GROWTH_TOKENS,
        PREFILL_GRAPH_SUBMISSION_LAYER_INTERVAL,
        SSD_PAGING_GENERATION_LAYER_INTERVAL,
    )
    .expect("the standard chunking tuple should be valid");
    let mlx_memory_limits = MlxMemoryLimits::new(
        AB_EFFECTIVE_MLX_CEILING_BYTES,
        AB_EFFECTIVE_MLX_CEILING_BYTES,
    )
    .expect("the MLX memory limits should be valid");
    let runtime = MlxRuntime::initialize(mlx_memory_limits)
        .expect("the MLX runtime should initialize under the A/B limits");
    eprintln!("[{PROBE_NAME}] status=progress phase=model_load");
    let mut model = Qwen3_5ResidentModel::load(
        runtime,
        validated_artifact,
        &model_directory,
        model_chunking,
    )
    .expect("the resident model should bind from the validated artifact");
    model
        .materialize_complete_weights_matching_production()
        .expect("the resident model should materialize its complete weights");
    eprintln!("[{PROBE_NAME}] status=success phase=model_load");
    (model, prompt_token_ids)
}

fn select_combination_arm(model: &mut Qwen3_5ResidentModel, combination_arm: CombinationArm) {
    match combination_arm {
        CombinationArm::TwoStep => model.disable_fused_sorted_expert_shared_combination(),
        CombinationArm::Fused => model
            .enable_fused_sorted_expert_shared_combination()
            .expect("the fused expert combination kernel should build"),
    }
}

/// Warms both arms (checking their logits are equal), then times alternating
/// pairs and prints each arm's min, median and max plus B's change against A.
fn report_prefill_ab_for_rung(
    model: &mut Qwen3_5ResidentModel,
    prefill_rung: &PrefillRung<'_>,
    ab_progress: &mut AbProgress,
) {
    assert_rung_logits_identical_across_arms(model, prefill_rung, ab_progress);
    let mut two_step_seconds = Vec::new();
    let mut fused_seconds = Vec::new();
    for pair_index in 0..MEASURED_PAIRS_PER_RUNG {
        let pair_order = if pair_index % 2 == 0 {
            [CombinationArm::TwoStep, CombinationArm::Fused]
        } else {
            [CombinationArm::Fused, CombinationArm::TwoStep]
        };
        for combination_arm in pair_order {
            select_combination_arm(model, combination_arm);
            let prefill_seconds = timed_prefill_seconds(model, prefill_rung);
            eprintln!(
                "[{PROBE_NAME}] status=progress rung={} pair={} arm={} prefill_seconds={prefill_seconds:.3} {}",
                prefill_rung.rung_label,
                pair_index + 1,
                combination_arm.label(),
                ab_progress.record_forward()
            );
            match combination_arm {
                CombinationArm::TwoStep => two_step_seconds.push(prefill_seconds),
                CombinationArm::Fused => fused_seconds.push(prefill_seconds),
            }
        }
    }
    print_rung_summary(prefill_rung, &mut two_step_seconds, &mut fused_seconds);
}

fn print_rung_summary(
    prefill_rung: &PrefillRung<'_>,
    two_step_seconds: &mut [f64],
    fused_seconds: &mut [f64],
) {
    let rung_token_count = prefill_rung.prompt_token_ids.len() as f64;
    two_step_seconds.sort_by(f64::total_cmp);
    fused_seconds.sort_by(f64::total_cmp);
    let two_step_median_tokens_per_second =
        rung_token_count / two_step_seconds[two_step_seconds.len() / 2];
    let fused_median_tokens_per_second = rung_token_count / fused_seconds[fused_seconds.len() / 2];
    let fused_change_percent =
        (fused_median_tokens_per_second / two_step_median_tokens_per_second - 1.0) * 100.0;
    eprintln!(
        "[{PROBE_NAME}] status=summary rung={} tokens={rung_token_count} A_two_step_seconds_min_median_max={:.3}/{:.3}/{:.3} B_fused_seconds_min_median_max={:.3}/{:.3}/{:.3} A_median_tokens_per_second={two_step_median_tokens_per_second:.1} B_median_tokens_per_second={fused_median_tokens_per_second:.1} B_change_percent={fused_change_percent:+.2}",
        prefill_rung.rung_label,
        two_step_seconds[0],
        two_step_seconds[two_step_seconds.len() / 2],
        two_step_seconds[two_step_seconds.len() - 1],
        fused_seconds[0],
        fused_seconds[fused_seconds.len() / 2],
        fused_seconds[fused_seconds.len() - 1],
    );
}

/// The warmup for each arm: the rung's chunks with the final chunk through
/// `forward_chunk`, so the shape's first-use JIT compile lands outside the
/// timed runs and both arms' final logits can be compared.
fn assert_rung_logits_identical_across_arms(
    model: &mut Qwen3_5ResidentModel,
    prefill_rung: &PrefillRung<'_>,
    ab_progress: &mut AbProgress,
) {
    let mut final_chunk_logits_by_arm = Vec::new();
    for combination_arm in [CombinationArm::TwoStep, CombinationArm::Fused] {
        select_combination_arm(model, combination_arm);
        final_chunk_logits_by_arm.push(final_chunk_logits(model, prefill_rung));
        eprintln!(
            "[{PROBE_NAME}] status=progress rung={} warmup_arm={} {}",
            prefill_rung.rung_label,
            combination_arm.label(),
            ab_progress.record_forward()
        );
    }
    let are_logits_identical = model
        .runtime
        .array_equal(
            &final_chunk_logits_by_arm[0],
            &final_chunk_logits_by_arm[1],
            false,
        )
        .expect("the logits comparison should build")
        .item_bool()
        .expect("the logits comparison should evaluate");
    eprintln!(
        "[{PROBE_NAME}] status=parity rung={} logits_shape={:?} identical={are_logits_identical}",
        prefill_rung.rung_label,
        final_chunk_logits_by_arm[0].shape()
    );
    assert!(
        are_logits_identical,
        "the fused expert combination must leave the model's logits unchanged"
    );
}

fn final_chunk_logits(model: &Qwen3_5ResidentModel, prefill_rung: &PrefillRung<'_>) -> MlxArray {
    let mut decoder_state = model
        .empty_request_decoder_state_matching_loaded_weights(kv_growth_tokens())
        .expect("the warmup decoder state should allocate");
    let chunk_starts = rung_chunk_starts(prefill_rung);
    let final_chunk_start = chunk_starts[chunk_starts.len() - 1];
    for &chunk_start in &chunk_starts[..chunk_starts.len() - 1] {
        model
            .prefill_chunk(
                rung_chunk(prefill_rung, chunk_start),
                chunk_start as u32,
                &mut decoder_state,
            )
            .expect("the warmup prefill chunk should complete");
    }
    model
        .forward_chunk(
            rung_chunk(prefill_rung, final_chunk_start),
            final_chunk_start as u32,
            &mut decoder_state,
        )
        .expect("the warmup final forward should complete")
}

/// One full prefill of the rung from an empty decoder state. `prefill_chunk`
/// evaluates the decoder state before returning, so the wall clock covers the
/// GPU work, not only graph construction.
fn timed_prefill_seconds(model: &Qwen3_5ResidentModel, prefill_rung: &PrefillRung<'_>) -> f64 {
    let mut decoder_state = model
        .empty_request_decoder_state_matching_loaded_weights(kv_growth_tokens())
        .expect("the prefill decoder state should allocate");
    let prefill_started_at = Instant::now();
    for chunk_start in rung_chunk_starts(prefill_rung) {
        model
            .prefill_chunk(
                rung_chunk(prefill_rung, chunk_start),
                chunk_start as u32,
                &mut decoder_state,
            )
            .expect("the prefill chunk should complete");
    }
    prefill_started_at.elapsed().as_secs_f64()
}

fn rung_chunk_starts(prefill_rung: &PrefillRung<'_>) -> Vec<usize> {
    (0..prefill_rung.prompt_token_ids.len())
        .step_by(prefill_rung.chunk_tokens)
        .collect()
}

fn rung_chunk<'tokens>(prefill_rung: &PrefillRung<'tokens>, chunk_start: usize) -> &'tokens [u32] {
    let chunk_end =
        (chunk_start + prefill_rung.chunk_tokens).min(prefill_rung.prompt_token_ids.len());
    &prefill_rung.prompt_token_ids[chunk_start..chunk_end]
}

fn kv_growth_tokens() -> i32 {
    i32::try_from(FULL_ATTENTION_KV_GROWTH_TOKENS).expect("the growth tokens fit i32")
}

/// Encodes the Shakespeare prompt, rendered in the same chat shape the daemon
/// serves so token statistics match a real request.
fn shakespeare_prompt_token_ids(tokenizer: &Qwen3_5Tokenizer) -> Vec<u32> {
    let romeo_and_juliet_source =
        include_str!("../../fixtures/model_metrics_10000_tokens_romeo_and_juliet.txt");
    let rendered_prompt = format!(
        "<|im_start|>user\nUse this Romeo and Juliet source:\n{romeo_and_juliet_source}<|im_end|>\n<|im_start|>assistant\n<think>\n"
    );
    tokenizer
        .encode_prompt(&rendered_prompt)
        .expect("the Shakespeare prompt should encode")
}
