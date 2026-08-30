//! Direct automatic sparse-expert prefill checks for the real Ornith artifact.
//!
//! This isolates the prefill math path by calling `forward_chunk` with the full
//! rendered prompt. No decode loop, REST surface, worker process, or OpenCode
//! adapter participates in this test.

use std::future::Future;
use std::time::{Duration, Instant};

use astronomical_ipc_protocol::{
    ChatGenerationCommand, ChatGenerationSettings, ChatMessage, ChatToolChoice, RequestId,
};
use astronomical_model_serving::{
    Qwen3_5ArtifactValidator, Qwen3_5MoEPagedPrefillExecutionMode, Qwen3_5Model, Qwen3_5Tokenizer,
};
use astronomical_runtime_integration::MlxRuntime;
use tokio::time::{MissedTickBehavior, interval, sleep};

const PREFILL_COMPARISON_TIMEOUT: Duration = Duration::from_secs(120);
const TOP_LOGIT_COUNT: usize = 8;
pub(crate) const MAXIMUM_EXPECTED_AUTOMATIC_PREFILL_ABSOLUTE_LOGIT_DELTA: f32 = 0.0;
#[tokio::test]
#[ignore = "loads the full Ornith model twice to compare automatic compact multi-token prefill logits"]
async fn should_match_automatic_compact_multi_token_prefill_after_contiguous_index_copy() {
    require_prefill_comparison_completion(run_compact_multi_token_forward_comparison()).await;
}
async fn run_compact_multi_token_forward_comparison() {
    let prompt_token_ids = prepare_reproduced_prompt_token_ids();
    run_automatic_prefill_comparison(
        &prompt_token_ids,
        "compact_multi_token_prompt_prepared",
        "automatic_compact_diagnostic",
        "automatic_standard_reference",
        Some(Qwen3_5MoEPagedPrefillExecutionMode::CompactPromptDiagnostic),
        MAXIMUM_EXPECTED_AUTOMATIC_PREFILL_ABSOLUTE_LOGIT_DELTA,
    )
    .await;
}

async fn run_automatic_prefill_comparison(
    prompt_token_ids: &[u32],
    progress_phase_label: &'static str,
    candidate_prefill_path_label: &'static str,
    reference_prefill_path_label: &'static str,
    paged_prefill_execution_mode: Option<Qwen3_5MoEPagedPrefillExecutionMode>,
    maximum_expected_absolute_logit_delta: f32,
) {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let test_started_at = Instant::now();
    eprintln!(
        "[paged-prefill-compare] status=progress phase={progress_phase_label} token_count={}",
        prompt_token_ids.len()
    );
    let candidate_prefill_snapshot = run_prefill_snapshot(
        prompt_token_ids,
        candidate_prefill_path_label,
        test_started_at,
        paged_prefill_execution_mode,
    )
    .await;
    let reference_prefill_snapshot = run_prefill_snapshot(
        prompt_token_ids,
        reference_prefill_path_label,
        test_started_at,
        None,
    )
    .await;
    assert_matching_forward_snapshots(
        &reference_prefill_snapshot,
        &candidate_prefill_snapshot,
        maximum_expected_absolute_logit_delta,
    );
}

fn assert_matching_forward_snapshots(
    reference_prefill_snapshot: &PrefillSnapshot,
    candidate_prefill_snapshot: &PrefillSnapshot,
    maximum_expected_absolute_logit_delta: f32,
) {
    let maximum_absolute_logit_delta = maximum_absolute_difference(
        &reference_prefill_snapshot.final_position_logits,
        &candidate_prefill_snapshot.final_position_logits,
    );
    eprintln!(
        "[paged-prefill-compare] status=progress phase=comparison reference_highest_logit_token_id={} candidate_highest_logit_token_id={} max_abs_logit_delta={:.6} reference_top_logits={} candidate_top_logits={}",
        reference_prefill_snapshot.highest_logit_token_id,
        candidate_prefill_snapshot.highest_logit_token_id,
        maximum_absolute_logit_delta,
        format_top_logits(&reference_prefill_snapshot.top_logits),
        format_top_logits(&candidate_prefill_snapshot.top_logits)
    );

    assert_eq!(
        candidate_prefill_snapshot.highest_logit_token_id,
        reference_prefill_snapshot.highest_logit_token_id,
        "automatic prefill changed the first highest-logit token; reference_top_logits={} candidate_top_logits={}",
        format_top_logits(&reference_prefill_snapshot.top_logits),
        format_top_logits(&candidate_prefill_snapshot.top_logits)
    );
    assert!(
        maximum_absolute_logit_delta <= maximum_expected_absolute_logit_delta,
        "automatic prefill logits diverged: max_abs_logit_delta={maximum_absolute_logit_delta:.6}, reference_top_logits={}, candidate_top_logits={}",
        format_top_logits(&reference_prefill_snapshot.top_logits),
        format_top_logits(&candidate_prefill_snapshot.top_logits)
    );
}

pub(crate) async fn require_prefill_comparison_completion(
    comparison_future: impl Future<Output = ()>,
) {
    let started_at = Instant::now();
    let timeout_deadline = sleep(PREFILL_COMPARISON_TIMEOUT);
    let mut progress_interval = interval(Duration::from_secs(10));
    progress_interval.set_missed_tick_behavior(MissedTickBehavior::Skip);
    tokio::pin!(comparison_future);
    tokio::pin!(timeout_deadline);
    progress_interval.tick().await;
    eprintln!(
        "[paged-prefill-compare] status=timeout_guard_started timeout_seconds={}",
        PREFILL_COMPARISON_TIMEOUT.as_secs()
    );

    loop {
        tokio::select! {
            () = &mut comparison_future => {
                eprintln!(
                    "[paged-prefill-compare] status=completed elapsed_seconds={:.1}",
                    started_at.elapsed().as_secs_f64()
                );
                return;
            }
            () = &mut timeout_deadline => {
                panic!("the direct paged prefill comparison exceeded {} seconds", PREFILL_COMPARISON_TIMEOUT.as_secs());
            }
            _ = progress_interval.tick() => {
                let elapsed = started_at.elapsed();
                let remaining = PREFILL_COMPARISON_TIMEOUT.saturating_sub(elapsed);
                eprintln!(
                    "[paged-prefill-compare] status=running elapsed_seconds={:.0} ETA<={:.0}",
                    elapsed.as_secs_f64(),
                    remaining.as_secs_f64()
                );
            }
        }
    }
}

pub(crate) fn prepare_reproduced_prompt_token_ids() -> Vec<u32> {
    let model_directory = crate::common::configured_large_sparse_moe_model_directory();
    let validated_artifact = Qwen3_5ArtifactValidator::new()
        .validate(&model_directory, 20_480)
        .expect("the Ornith artifact should validate before tokenizer loading");
    let tokenizer = Qwen3_5Tokenizer::from_validated_artifact(&validated_artifact)
        .expect("the Ornith tokenizer should load from validated model metadata");
    let chat_generation_command = ChatGenerationCommand {
        request_id: RequestId::new(9_001),
        model: crate::common::large_sparse_moe_model_id().to_owned(),
        messages: vec![ChatMessage::User {
            content: "so what folder are we in?".to_owned(),
            images: Vec::new(),
        }],
        tools: Vec::new(),
        tool_choice: ChatToolChoice::None,
        settings: ChatGenerationSettings {
            max_output_tokens: 512,
            temperature_thousandths: None,
            top_p_thousandths: None,
            seed: None,
            thinking_budget: Some(256),
        },
        qwen_thinking_channel_seed: None,
    };
    tokenizer
        .prepare_chat(&chat_generation_command, true)
        .expect("the reproduced chat command should prepare for Ornith")
        .input_token_ids()
        .to_vec()
}

pub(crate) async fn run_prefill_snapshot(
    prompt_token_ids: &[u32],
    prefill_path_label: &'static str,
    test_started_at: Instant,
    paged_prefill_execution_mode: Option<Qwen3_5MoEPagedPrefillExecutionMode>,
) -> PrefillSnapshot {
    assert!(
        test_started_at.elapsed() < PREFILL_COMPARISON_TIMEOUT,
        "prefill comparison should make progress before timeout"
    );
    eprintln!(
        "[paged-prefill-compare] status=progress phase=artifact_validation path={prefill_path_label} elapsed_seconds={:.2}",
        test_started_at.elapsed().as_secs_f64()
    );
    let model_directory = crate::common::configured_large_sparse_moe_model_directory();
    let validated_artifact = Qwen3_5ArtifactValidator::new()
        .validate(&model_directory, 20_480)
        .expect("the Ornith artifact should validate before native loading");
    let config = validated_artifact.config().clone();
    let mlx_memory_limits = crate::common::sample_serving_acceptance_mlx_memory_limits().await;
    let runtime = MlxRuntime::initialize(mlx_memory_limits)
        .expect("the direct MLX runtime should initialize");
    eprintln!(
        "[paged-prefill-compare] status=progress phase=model_load path={prefill_path_label} residency=automatic elapsed_seconds={:.2}",
        test_started_at.elapsed().as_secs_f64()
    );
    let model_load_started_at = Instant::now();
    let qwen3_5_model = Qwen3_5Model::load(
        runtime,
        validated_artifact,
        &model_directory,
        false,
        crate::common::standard_qwen3_5_model_chunking_configuration(),
    )
    .expect("the complete Ornith model should bind from validated descriptors");
    eprintln!(
        "[paged-prefill-compare] status=progress phase=model_loaded path={prefill_path_label} load_elapsed_seconds={:.2} total_elapsed_seconds={:.2}",
        model_load_started_at.elapsed().as_secs_f64(),
        test_started_at.elapsed().as_secs_f64()
    );
    let mut request_decoder_state = crate::common::standard_request_decoder_state(&config);
    let prefill_started_at = Instant::now();
    let final_position_logits = match paged_prefill_execution_mode {
        Some(paged_prefill_execution_mode) => qwen3_5_model
            .forward_chunk_with_paged_prefill_execution_mode_for_tests(
                prompt_token_ids,
                0,
                &mut request_decoder_state,
                paged_prefill_execution_mode,
            ),
        None => qwen3_5_model.forward_chunk(prompt_token_ids, 0, &mut request_decoder_state),
    }
    .unwrap_or_else(|error| panic!("{prefill_path_label} prefill should produce logits: {error}"));
    let highest_logit_token_id = qwen3_5_model
        .highest_logit_token_id(&final_position_logits)
        .unwrap_or_else(|error| {
            panic!("{prefill_path_label} logits should produce a highest-logit token: {error}")
        });
    let final_position_logits = final_position_logits.to_vec_f32().unwrap_or_else(|error| {
        panic!("{prefill_path_label} logits should materialize to CPU: {error}")
    });
    let prefill_elapsed = prefill_started_at.elapsed();
    let top_logits = top_logits(&final_position_logits, TOP_LOGIT_COUNT);
    eprintln!(
        "[paged-prefill-compare] status=progress phase=prefill_done path={prefill_path_label} prefill_elapsed_seconds={:.2} highest_logit_token_id={} top_logits={} total_elapsed_seconds={:.2}",
        prefill_elapsed.as_secs_f64(),
        highest_logit_token_id,
        format_top_logits(&top_logits),
        test_started_at.elapsed().as_secs_f64()
    );
    qwen3_5_model
        .runtime()
        .clear_allocator_cache()
        .expect("the test should release reclaimable MLX allocator cache between model loads");
    PrefillSnapshot {
        highest_logit_token_id,
        final_position_logits,
        top_logits,
    }
}

pub(crate) struct PrefillSnapshot {
    pub(crate) highest_logit_token_id: u32,
    pub(crate) final_position_logits: Vec<f32>,
    top_logits: Vec<TokenLogit>,
}

#[derive(Clone)]
struct TokenLogit {
    token_id: usize,
    logit: f32,
}

fn top_logits(final_position_logits: &[f32], top_logit_count: usize) -> Vec<TokenLogit> {
    let mut token_logits = final_position_logits
        .iter()
        .copied()
        .enumerate()
        .map(|(token_id, logit)| TokenLogit { token_id, logit })
        .collect::<Vec<_>>();
    token_logits.sort_by(|left_logit, right_logit| {
        right_logit
            .logit
            .total_cmp(&left_logit.logit)
            .then_with(|| left_logit.token_id.cmp(&right_logit.token_id))
    });
    token_logits.truncate(top_logit_count);
    token_logits
}

pub(crate) fn maximum_absolute_difference(
    reference_logits: &[f32],
    candidate_logits: &[f32],
) -> f32 {
    assert_eq!(
        reference_logits.len(),
        candidate_logits.len(),
        "resident and paged logits should have the same vocabulary length"
    );
    reference_logits
        .iter()
        .zip(candidate_logits)
        .enumerate()
        .map(
            |(vocabulary_token_id, (reference_logit, candidate_logit))| {
                assert!(
                    reference_logit.is_finite() && candidate_logit.is_finite(),
                    "logit comparison requires finite values: vocabulary_token_id={vocabulary_token_id}, reference_logit={reference_logit}, candidate_logit={candidate_logit}"
                );
                (reference_logit - candidate_logit).abs()
            },
        )
        .fold(0.0_f32, f32::max)
}

fn format_top_logits(token_logits: &[TokenLogit]) -> String {
    token_logits
        .iter()
        .map(|token_logit| format!("{}:{:.4}", token_logit.token_id, token_logit.logit))
        .collect::<Vec<_>>()
        .join(",")
}
