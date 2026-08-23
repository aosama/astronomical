use std::time::Instant;

use astronomical_model_serving::{Qwen3_5ArtifactValidator, Qwen3_5Config, Qwen3_5Model};
use astronomical_runtime_integration::MlxRuntime;

use super::super::qwen3_5::SAY_HI_PROMPT_TOKEN_IDS;
use super::expert_paging_decode::{bytes_to_gib, require_expert_paging_decode_completion};
use super::expert_paging_prefill_performance::prepare_reproduced_long_prompt_token_ids_for_model;
use super::expert_paging_representative_performance::{
    REPRESENTATIVE_INPUT_TOKEN_COUNT, REPRESENTATIVE_OUTPUT_TOKEN_COUNT,
    run_representative_performance_probe_for_loaded_model,
};

const PAGED_TEST_PROGRESS_LOG_PREFIX: &str = "[ornith-1.5-35b-a3b-oq8e-paged]";

#[tokio::test]
#[ignore = "loads configured Ornith-1.5-35B-A3B-oQ8e-mtp through expert paging"]
async fn should_generate_one_ornith_1_5_35b_a3b_eight_bit_token_with_bounded_expert_paging() {
    require_expert_paging_decode_completion(
        run_one_token_expert_paging_smoke_test(),
        PAGED_TEST_PROGRESS_LOG_PREFIX,
        "the Ornith 1.5 35B A3B eight-bit paged smoke test",
    )
    .await;
}

#[tokio::test]
#[ignore = "measures Ornith-1.5-35B-A3B-oQ8e-mtp with 1024 input and 500 output tokens"]
async fn should_measure_ornith_1_5_35b_a3b_eight_bit_with_1024_input_and_500_output_tokens() {
    require_expert_paging_decode_completion(
        run_representative_performance_probe(),
        PAGED_TEST_PROGRESS_LOG_PREFIX,
        "the Ornith 1.5 35B A3B eight-bit representative performance probe",
    )
    .await;
}

async fn run_one_token_expert_paging_smoke_test() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let (qwen3_5_model, config) = load_paged_model().await;
    let loaded_memory_snapshot = qwen3_5_model
        .runtime()
        .memory_snapshot()
        .expect("the smoke test should sample memory after model loading");
    eprintln!(
        "{PAGED_TEST_PROGRESS_LOG_PREFIX} status=progress phase=model_loaded mlx_active_gib={:.2} mlx_allocator_gib={:.2}",
        bytes_to_gib(loaded_memory_snapshot.active_memory_bytes() as u64),
        bytes_to_gib(loaded_memory_snapshot.allocator_cache_memory_bytes() as u64)
    );

    let mut request_decoder_state = crate::common::standard_request_decoder_state(&config);
    qwen3_5_model
        .prefill_chunck(
            &SAY_HI_PROMPT_TOKEN_IDS[..SAY_HI_PROMPT_TOKEN_IDS.len() - 1],
            0,
            &mut request_decoder_state,
        )
        .expect("the Ornith prompt prefix should materialize decoder state");
    let logits = qwen3_5_model
        .forward_chunk(
            &SAY_HI_PROMPT_TOKEN_IDS[SAY_HI_PROMPT_TOKEN_IDS.len() - 1..],
            (SAY_HI_PROMPT_TOKEN_IDS.len() - 1) as u32,
            &mut request_decoder_state,
        )
        .expect("the Ornith paged decode step should complete");
    let generated_token_id = qwen3_5_model
        .greedy_token_id(&logits)
        .expect("the Ornith logits should produce one greedy token");

    let expert_weight_memory_cache_statistics =
        qwen3_5_model.expert_weight_memory_cache_statistics();
    let final_memory_snapshot = qwen3_5_model
        .runtime()
        .memory_snapshot()
        .expect("the smoke test should sample final MLX memory");
    let expected_selected_expert_page_count = usize::try_from(
        config
            .layer_count()
            .checked_mul(config.experts_per_token())
            .expect("the selected expert page count should fit u32"),
    )
    .expect("the selected expert page count should fit usize");
    eprintln!(
        "{PAGED_TEST_PROGRESS_LOG_PREFIX} status=progress phase=final_memory generated_token_id={} cache_entries={} expected_selected_entries={} resident_payload_gib={:.2} maximum_resident_payload_gib={:.2} mlx_active_gib={:.2} mlx_allocator_gib={:.2} mlx_peak_gib={:.2}",
        generated_token_id,
        expert_weight_memory_cache_statistics.entry_count,
        expected_selected_expert_page_count,
        bytes_to_gib(expert_weight_memory_cache_statistics.resident_payload_byte_count),
        bytes_to_gib(expert_weight_memory_cache_statistics.maximum_resident_payload_byte_count),
        bytes_to_gib(final_memory_snapshot.active_memory_bytes() as u64),
        bytes_to_gib(final_memory_snapshot.allocator_cache_memory_bytes() as u64),
        bytes_to_gib(final_memory_snapshot.peak_memory_bytes() as u64),
    );
    assert!(
        expert_weight_memory_cache_statistics.entry_count > 0,
        "the smoke request should retain at least one expert page"
    );
    assert!(
        expert_weight_memory_cache_statistics.entry_count <= expected_selected_expert_page_count,
        "adaptive retention cannot contain more entries than the routed request loaded"
    );
    assert_ne!(
        expert_weight_memory_cache_statistics.maximum_resident_payload_byte_count,
        u64::MAX
    );
    assert!(
        expert_weight_memory_cache_statistics.resident_payload_byte_count
            <= expert_weight_memory_cache_statistics.maximum_resident_payload_byte_count
    );
    eprintln!("{PAGED_TEST_PROGRESS_LOG_PREFIX} status=success");
}

async fn run_representative_performance_probe() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let test_started_at = Instant::now();
    let model_directory = super::ornith_1_5_35b_a3b_eight_bit_model_directory();
    let prompt_token_ids = prepare_reproduced_long_prompt_token_ids_for_model(
        &model_directory,
        crate::common::ORNITH_MODEL_SWAP_SOURCE_MODEL_ID,
        REPRESENTATIVE_INPUT_TOKEN_COUNT,
        REPRESENTATIVE_OUTPUT_TOKEN_COUNT as u16,
    )
    .expect("the Ornith eight-bit prompt should prepare at the exact requested length");
    let (qwen3_5_model, config) = load_paged_model().await;
    run_representative_performance_probe_for_loaded_model(
        &qwen3_5_model,
        &config,
        &prompt_token_ids,
        PAGED_TEST_PROGRESS_LOG_PREFIX,
        test_started_at,
    );
}

async fn load_paged_model() -> (Qwen3_5Model, Qwen3_5Config) {
    let model_directory = super::ornith_1_5_35b_a3b_eight_bit_model_directory();
    eprintln!("{PAGED_TEST_PROGRESS_LOG_PREFIX} status=start phase=artifact_validation");
    let validated_artifact = Qwen3_5ArtifactValidator::new()
        .validate(&model_directory, 20_480)
        .expect("the Ornith 1.5 35B A3B eight-bit artifact should validate before paged loading");
    let config = validated_artifact.config().clone();
    eprintln!(
        "{PAGED_TEST_PROGRESS_LOG_PREFIX} status=progress phase=artifact_validated shards={} payload_bytes={} layers={} experts={} experts_per_token={} quantization_bits={} quantization_group_size={}",
        validated_artifact.shard_count(),
        validated_artifact.total_payload_bytes(),
        config.layer_count(),
        config.expert_count(),
        config.experts_per_token(),
        config.default_quantization_bits(),
        config.default_quantization_group_size()
    );

    let mlx_memory_limits =
        crate::common::sample_model_artifact_qualification_mlx_memory_limits().await;
    let runtime = MlxRuntime::initialize(mlx_memory_limits)
        .expect("the MLX runtime should initialize for Ornith expert paging");
    eprintln!("{PAGED_TEST_PROGRESS_LOG_PREFIX} status=progress phase=model_load");
    let qwen3_5_model = Qwen3_5Model::load(
        runtime,
        validated_artifact,
        &model_directory,
        false,
        crate::common::standard_qwen3_5_model_chunking_configuration(),
    )
    .expect("the Ornith 1.5 35B A3B eight-bit model should load with explicit expert streaming");
    (qwen3_5_model, config)
}
