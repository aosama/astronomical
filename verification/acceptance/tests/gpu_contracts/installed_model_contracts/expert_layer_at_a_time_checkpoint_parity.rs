//! An expert model that reads a prompt one layer at a time must publish the
//! prompt-cache state it publishes chunk by chunk, bit for bit, and must
//! read each streamed layer once (issue #2221).
//!
//! One layer at a time, a streamed expert layer's complete page is read
//! once and held for every span of the prompt. This journey prefills one
//! Shakespeare prompt both ways from fresh state under a limit far below
//! the model's expert weights: first with nothing kept, so every layer is
//! streamed and held, then with the layers the prefill plan funds kept. The
//! published boundaries must fall at the same positions and hold identical
//! tensors, and with nothing kept the layer-at-a-time prefill must read
//! every expert of every layer exactly once where the chunk path reads them
//! once per chunk.

use std::time::Duration;

use astronomical_model_serving::{
    MemoryPhase, PerformanceAttribution, Qwen3_5ArtifactValidator, Qwen3_5StreamingModel,
};
use astronomical_runtime_integration::{MlxMemoryLimits, MlxRuntime};
use serial_test::serial;
use tokio::time::timeout;

#[path = "../../support/romeo_and_juliet.rs"]
mod romeo_and_juliet;

/// The engine's SSD-streaming chunk and a prompt-cache block that puts one
/// boundary inside a span and one at a span's end.
const CHUNK_TOKEN_COUNT: usize = 4096;
const BLOCK_TOKEN_COUNT: usize = 1792;
/// About 4,600 tokens: two spans, with boundaries at 1,792 and 3,584.
const PROMPT_WORD_COUNT: usize = 3000;
/// This limit defines a reproducible acceptance cell only, matching the
/// streaming scoreboard's row for this model; production code must not
/// hardwire it.
const MLX_MEMORY_LIMIT_BYTES: usize = 11 * 1024 * 1024 * 1024;
const CHECKPOINT_PARITY_BUDGET: Duration = Duration::from_secs(120);

#[tokio::test]
#[ignore = "loads the small_sparse_moe_streaming model and compares published prompt-cache state"]
#[serial]
async fn should_publish_the_chunk_path_prompt_cache_state_reading_experts_one_layer_at_a_time() {
    timeout(CHECKPOINT_PARITY_BUDGET, assert_checkpoint_parity())
        .await
        .expect("the checkpoint parity journey must finish within its budget");
}

async fn assert_checkpoint_parity() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let model_directory = crate::common::configured_small_sparse_moe_streaming_model_directory();
    let prompt_token_ids = two_span_prompt_token_ids(&model_directory);
    let validated_artifact = Qwen3_5ArtifactValidator::new()
        .validate(&model_directory, 64)
        .expect("the small sparse artifact should validate");
    let qwen3_5_config = validated_artifact.config().clone();
    let streaming_model = Qwen3_5StreamingModel::load(
        MlxRuntime::initialize(
            MlxMemoryLimits::new(MLX_MEMORY_LIMIT_BYTES, MLX_MEMORY_LIMIT_BYTES)
                .expect("the acceptance limit should be a valid MLX limit"),
        )
        .expect("the runtime should initialize"),
        validated_artifact,
        &model_directory,
        crate::common::standard_qwen3_5_model_chunking_configuration(),
    )
    .expect("the small sparse model should load on the streaming engine");
    streaming_model
        .materialize_target_weights()
        .expect("the model's kept weights should materialize");
    let streamed_boundaries = assert_parity_with_every_layer_streamed(
        &streaming_model,
        &qwen3_5_config,
        &prompt_token_ids,
    );
    assert_parity_after_a_span_is_run_again(
        &streaming_model,
        &qwen3_5_config,
        &prompt_token_ids,
        &streamed_boundaries,
    );
    assert_parity_with_planned_layers_kept(
        &streaming_model,
        &qwen3_5_config,
        &prompt_token_ids,
        &streamed_boundaries,
    );
}

/// A span that fails is run again and nothing before it is (issue #2221).
///
/// The engine redoes a span MLX could not fit. Here a span is failed on
/// purpose after it is submitted: once the second span of an early layer,
/// which puts the layer back to its first span because that one was not
/// confirmed yet, and once the first span of the last layer. Either way the
/// published state must be the undisturbed prefill's bit for bit, and the
/// only experts read twice are the failed layer's.
fn assert_parity_after_a_span_is_run_again(
    streaming_model: &Qwen3_5StreamingModel,
    qwen3_5_config: &astronomical_model_serving::Qwen3_5Config,
    prompt_token_ids: &[u32],
    streamed_boundaries: &PublishedBoundaries,
) {
    let layer_count = qwen3_5_config.layer_count() as usize;
    let experts_read_without_a_failure =
        u64::from(qwen3_5_config.layer_count()) * u64::from(qwen3_5_config.expert_count());
    for (failed_layer_index, failed_span_index) in [(1, 1), (layer_count - 1, 0)] {
        let expert_reads = ExpertReadCount::starting(streaming_model);
        let (redone_boundaries, redone_span_count) = streaming_model
            .prefill_layer_major_redoing_failed_spans(
                prompt_token_ids,
                CHUNK_TOKEN_COUNT,
                BLOCK_TOKEN_COUNT,
                Some((failed_layer_index, failed_span_index)),
                &mut crate::common::standard_request_decoder_state(qwen3_5_config),
            )
            .expect("a prefill whose failed span is run again should finish");
        assert_eq!(
            redone_span_count, 1,
            "one failed span must cost one redo: layer={failed_layer_index} span={failed_span_index}"
        );
        assert!(
            bit_patterns(&redone_boundaries) == bit_patterns(streamed_boundaries),
            "running a failed span again must publish the undisturbed state bit for bit: layer={failed_layer_index} span={failed_span_index}"
        );
        assert_eq!(
            expert_reads.since(streaming_model),
            experts_read_without_a_failure + u64::from(qwen3_5_config.expert_count()),
            "only the failed layer's experts are read again: layer={failed_layer_index} span={failed_span_index}"
        );
    }
    let (undisturbed_boundaries, redone_span_count) = streaming_model
        .prefill_layer_major_redoing_failed_spans(
            prompt_token_ids,
            CHUNK_TOKEN_COUNT,
            BLOCK_TOKEN_COUNT,
            Some((layer_count, 0)),
            &mut crate::common::standard_request_decoder_state(qwen3_5_config),
        )
        .expect("a prefill with no span to fail should finish");
    assert_eq!(
        redone_span_count, 0,
        "a span the prompt does not have cannot fail, so nothing is run again"
    );
    assert!(bit_patterns(&undisturbed_boundaries) == bit_patterns(streamed_boundaries));
    eprintln!("[expert-layer-at-a-time-parity] progress=span-redo layers_failed=2 redone_spans=2");
}

type PublishedBoundaries = Vec<(usize, std::collections::BTreeMap<String, Vec<f32>>)>;

/// Nothing kept: a model loaded directly has no residency plan, so its
/// cache admits nothing and every layer streams. Returns the chunk path's
/// boundaries for the second half to compare against.
fn assert_parity_with_every_layer_streamed(
    streaming_model: &Qwen3_5StreamingModel,
    qwen3_5_config: &astronomical_model_serving::Qwen3_5Config,
    prompt_token_ids: &[u32],
) -> PublishedBoundaries {
    let chunk_path_reads = ExpertReadCount::starting(streaming_model);
    let chunk_path_boundaries = streaming_model
        .prefill_chunk_by_chunk_with_prompt_cache_checkpoints(
            prompt_token_ids,
            CHUNK_TOKEN_COUNT,
            BLOCK_TOKEN_COUNT,
            &mut crate::common::standard_request_decoder_state(qwen3_5_config),
        )
        .expect("the chunk path should prefill");
    let chunk_path_expert_reads = chunk_path_reads.since(streaming_model);
    let layer_at_a_time_reads = ExpertReadCount::starting(streaming_model);
    let layer_at_a_time_boundaries = streaming_model
        .prefill_layer_major_with_prompt_cache_checkpoints(
            prompt_token_ids,
            CHUNK_TOKEN_COUNT,
            BLOCK_TOKEN_COUNT,
            &mut crate::common::standard_request_decoder_state(qwen3_5_config),
        )
        .expect("the layer-at-a-time prefill should run");
    let layer_at_a_time_expert_reads = layer_at_a_time_reads.since(streaming_model);
    let boundary_positions: Vec<usize> = chunk_path_boundaries
        .iter()
        .map(|(boundary_token, _)| *boundary_token)
        .collect();
    assert_eq!(
        boundary_positions,
        vec![BLOCK_TOKEN_COUNT, 2 * BLOCK_TOKEN_COUNT],
        "the chunk path should publish both boundaries"
    );
    assert!(
        bit_patterns(&layer_at_a_time_boundaries) == bit_patterns(&chunk_path_boundaries),
        "with every layer streamed the layer-at-a-time prefill must publish the chunk path's boundaries bit for bit"
    );
    assert_eq!(
        layer_at_a_time_expert_reads,
        u64::from(qwen3_5_config.layer_count()) * u64::from(qwen3_5_config.expert_count()),
        "one layer at a time, every expert of every layer is read exactly once"
    );
    assert!(
        chunk_path_expert_reads > layer_at_a_time_expert_reads,
        "the chunk path reads streamed layers once per chunk: chunk_path={chunk_path_expert_reads} layer_at_a_time={layer_at_a_time_expert_reads}"
    );
    eprintln!(
        "[expert-layer-at-a-time-parity] progress=1/2 kept=none prompt_tokens={} boundaries={boundary_positions:?} chunk_path_expert_reads={chunk_path_expert_reads} layer_at_a_time_expert_reads={layer_at_a_time_expert_reads}",
        prompt_token_ids.len()
    );
    chunk_path_boundaries
}

/// The layers the prefill plan funds are kept as they are read; the rest
/// stream. Fresh state each way, with the plan published as a request open
/// publishes it.
fn assert_parity_with_planned_layers_kept(
    streaming_model: &Qwen3_5StreamingModel,
    qwen3_5_config: &astronomical_model_serving::Qwen3_5Config,
    prompt_token_ids: &[u32],
    streamed_boundaries: &PublishedBoundaries,
) {
    let mut planned_boundaries = Vec::with_capacity(2);
    for reads_one_layer_at_a_time in [false, true] {
        streaming_model
            .refresh_phase_aware_expert_residency_plan_for_tests(
                MemoryPhase::Prefill,
                u64::try_from(prompt_token_ids.len()).unwrap_or(u64::MAX),
                u64::try_from(CHUNK_TOKEN_COUNT).unwrap_or(u64::MAX),
                &mut PerformanceAttribution::disabled(),
            )
            .expect("the request-open residency plan should publish");
        let mut request_decoder_state =
            crate::common::standard_request_decoder_state(qwen3_5_config);
        let published_boundaries = if reads_one_layer_at_a_time {
            streaming_model.prefill_layer_major_with_prompt_cache_checkpoints(
                prompt_token_ids,
                CHUNK_TOKEN_COUNT,
                BLOCK_TOKEN_COUNT,
                &mut request_decoder_state,
            )
        } else {
            streaming_model.prefill_chunk_by_chunk_with_prompt_cache_checkpoints(
                prompt_token_ids,
                CHUNK_TOKEN_COUNT,
                BLOCK_TOKEN_COUNT,
                &mut request_decoder_state,
            )
        }
        .expect("the planned prefill should run");
        planned_boundaries.push(published_boundaries);
    }
    let kept_expert_payload_bytes = streaming_model
        .expert_weight_memory_cache_statistics()
        .resident_payload_byte_count;
    assert!(
        kept_expert_payload_bytes > 0,
        "the published plan must keep at least one layer, or this half repeats the first"
    );
    assert!(
        bit_patterns(&planned_boundaries[1]) == bit_patterns(&planned_boundaries[0]),
        "with planned layers kept the layer-at-a-time prefill must publish the chunk path's boundaries bit for bit"
    );
    assert!(
        bit_patterns(&planned_boundaries[0]) == bit_patterns(streamed_boundaries),
        "keeping layers must not change the published state"
    );
    eprintln!(
        "[expert-layer-at-a-time-parity] progress=2/2 kept_expert_gb={:.2} tensors_per_boundary={}",
        kept_expert_payload_bytes as f64 / 1e9,
        streamed_boundaries[0].1.len()
    );
}

/// Experts read from storage since a starting point, from the cache's own
/// count of them.
struct ExpertReadCount {
    experts_read_before: u64,
}

impl ExpertReadCount {
    fn starting(streaming_model: &Qwen3_5StreamingModel) -> Self {
        Self {
            experts_read_before: streaming_model
                .expert_weight_memory_cache_statistics()
                .disk_page_load_count,
        }
    }

    fn since(&self, streaming_model: &Qwen3_5StreamingModel) -> u64 {
        streaming_model
            .expert_weight_memory_cache_statistics()
            .disk_page_load_count
            .saturating_sub(self.experts_read_before)
    }
}

fn two_span_prompt_token_ids(model_directory: &std::path::Path) -> Vec<u32> {
    let tokenizer = tokenizers::Tokenizer::from_file(model_directory.join("tokenizer.json"))
        .expect("the small sparse model's tokenizer should load");
    let prompt_token_ids = tokenizer
        .encode(romeo_and_juliet::summary_prompt(PROMPT_WORD_COUNT), false)
        .expect("the Shakespeare prompt should encode")
        .get_ids()
        .to_vec();
    assert!(
        (2 * BLOCK_TOKEN_COUNT..3 * BLOCK_TOKEN_COUNT).contains(&prompt_token_ids.len()),
        "the prompt must cross exactly two block boundaries; it has {} tokens",
        prompt_token_ids.len()
    );
    prompt_token_ids
}

/// Every snapshot value's bit pattern, so equality means bit for bit: an
/// equal NaN compares equal, and 0.0 differs from -0.0.
fn bit_patterns(
    published_boundaries: &PublishedBoundaries,
) -> Vec<(usize, Vec<(String, Vec<u32>)>)> {
    published_boundaries
        .iter()
        .map(|(boundary_token, snapshot_tensors)| {
            let tensor_bits = snapshot_tensors
                .iter()
                .map(|(tensor_name, values)| {
                    (
                        tensor_name.clone(),
                        values.iter().map(|value| value.to_bits()).collect(),
                    )
                })
                .collect();
            (*boundary_token, tensor_bits)
        })
        .collect()
}
