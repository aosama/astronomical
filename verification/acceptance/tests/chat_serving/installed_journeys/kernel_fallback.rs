//! Serving journeys that force every custom Metal kernel verdict to
//! unsupported and prove the request still completes through the public MLX
//! path with structurally valid output.
//!
//! The macOS 14+ best-effort support promise rests on one claim: on a GPU or
//! OS combination the project has never tested, an unprovable kernel degrades
//! throughput, never correctness. These journeys protect that claim through
//! the full engine serving path — artifact validation, model loading, prefill,
//! and decode — not just the kernel dispatch layer covered by the direct-MLX
//! demotion parity journeys.
//!
//! The forced verdicts are installed before the engine loads, so the loader
//! consults the same once-per-worker `OnceLock` production uses and retains
//! the fallback route for the whole process. One journey per process matches
//! the bounded serial execution rule for real-model journeys.

use std::time::Duration;

use ipc_protocol::RequestId;
use model_serving::{
    InferenceEngine, PerformanceAttribution, Qwen3_5ArtifactValidator, TextInferenceRequest,
    install_forced_worker_verdicts_for_tests, worker_process_kernel_capabilities,
};
use runtime_integration::MlxRuntime;
use tokio::time::timeout;

use super::small_chunk_text_engine::{
    construct_small_chunk_text_engine, count_tokens_within_vocabulary,
};
use crate::serving_acceptance::support::{IMAGE_PAD_TOKEN_ID, LOCAL_AI_PROMPT_TOKEN_IDS};

const FORCED_FALLBACK_MAX_TOKENS: usize = 16;

const FORCED_FALLBACK_TIMEOUT_SECONDS: u64 = 120;

/// Forces every Qwen3.5-consumed kernel family unsupported with a distinct
/// reason variant, then serves a sampled Romeo and Juliet continuation. The
/// loader demotes each family to the MLX ops fallback at load time, so the
/// whole generation runs without any custom kernel.
#[tokio::test]
#[ignore = "loads the complete Ornith artifact with every kernel family demoted"]
async fn should_serve_romeo_and_juliet_through_the_mlx_fallback_when_every_kernel_is_unsupported() {
    timeout(
        Duration::from_secs(FORCED_FALLBACK_TIMEOUT_SECONDS),
        run_forced_fallback_romeo_continuation(),
    )
    .await
    .expect("the forced-fallback continuation must finish within 120 seconds");
}

async fn run_forced_fallback_romeo_continuation() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    install_forced_worker_verdicts_for_tests(
        crate::common::forced_unsupported_worker_kernel_capabilities(),
    );
    let mlx_memory_limits = crate::common::sample_serving_acceptance_mlx_memory_limits().await;
    // The engine consults the same once-per-worker verdict owner, so proving
    // the forced verdicts flow through that owner proves the loader demotes
    // every family — the fallback cannot be silently skipped.
    let runtime = MlxRuntime::initialize(mlx_memory_limits)
        .expect("the forced-fallback journey runtime should initialize");
    let retained_verdicts =
        worker_process_kernel_capabilities(&runtime, &mut PerformanceAttribution::disabled());
    for family in [
        model_serving::kernel_capability::SORTED_EXPERT_WEIGHTED_SUM,
        model_serving::kernel_capability::GATED_DELTA_SEQUENCE,
        model_serving::kernel_capability::GATED_DELTA_BOUNDARY_CHECKPOINT,
    ] {
        assert!(
            !retained_verdicts.is_custom_kernel_supported(family),
            "family {family:?} must stay demoted for the whole worker process"
        );
    }
    let model_directory = crate::common::configured_large_sparse_moe_streaming_model_directory();
    let validated_artifact = Qwen3_5ArtifactValidator::new()
        .validate(&model_directory, 20_480)
        .expect("the Ornith artifact should validate before engine loading");
    let model_vocabulary_size = validated_artifact.config().vocabulary_size();
    let mut qwen3_5_engine = construct_small_chunk_text_engine(
        validated_artifact,
        &mlx_memory_limits,
        &model_directory,
        IMAGE_PAD_TOKEN_ID,
    );
    qwen3_5_engine
        .load()
        .await
        .expect("the engine should load the Ornith model with every kernel demoted");
    let request_id = RequestId::new(1_100);
    qwen3_5_engine
        .start_generation(
            TextInferenceRequest::new_sampling(
                request_id,
                LOCAL_AI_PROMPT_TOKEN_IDS.to_vec(),
                FORCED_FALLBACK_MAX_TOKENS as u16,
                1_000,
                1_000,
                Some(2_222),
            )
            .with_image_pad_token_id(IMAGE_PAD_TOKEN_ID),
        )
        .await
        .expect("the engine should accept one forced-fallback request");

    // Structural validity: the fallback path must produce at least one token
    // within the artifact's declared vocabulary and finish cleanly. Numeric
    // equivalence of the fallback route is proven per family by the direct-MLX
    // demotion parity journeys.
    let generated_token_count = count_tokens_within_vocabulary(
        &mut qwen3_5_engine,
        request_id,
        FORCED_FALLBACK_MAX_TOKENS,
        model_vocabulary_size,
        "forced-fallback",
    )
    .await;
    assert!(
        generated_token_count > 0,
        "the forced-fallback path must produce at least one token"
    );
}
