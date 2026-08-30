//! User journey: not enough RAM to seat every expert. Ask Romeo twice.
//! Stay under the ceiling. The second ask must not pull more pages from disk.

use std::time::Duration;

use astronomical_ipc_protocol::{ExpertMemoryMode, RequestId};
use astronomical_model_serving::{
    CompleteResidencyHeadroomBoundary, Qwen3_5ArtifactValidator,
    mlx_ram_budget_model_geometry_from_validated_artifact,
};
use astronomical_runtime_integration::MlxMemoryLimits;
use serde_json::Value;
use tokio::time::timeout;

use crate::serving_acceptance::speculative_prefill::support::prepare_romeo_and_juliet_three_paragraph_summary_prompt;
use crate::serving_acceptance::support::performance_attribution::{
    counter_amount, create_attributed_engine, generation_report_for_request,
    load_engine_with_progress, read_attribution_report_documents, run_attributed_generation,
};

const PROMPT_TOKEN_COUNT: usize = 48;
const OUTPUT_TOKEN_COUNT: u16 = 32;
const COLD_REQUEST_ID: RequestId = RequestId::new(96_100);
const WARM_REQUEST_ID: RequestId = RequestId::new(96_101);

fn model_id() -> &'static str {
    crate::common::large_sparse_moe_model_id()
}

#[tokio::test]
#[ignore = "loads a sparse model under its paging ceiling and serves Romeo twice"]
async fn should_serve_romeo_twice_while_paging_without_exceeding_the_ram_ceiling() {
    timeout(Duration::from_secs(115), async {
        let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
        let model_directory =
            crate::common::configured_installed_model_directory_by_id(model_id());
        let validated_artifact = Qwen3_5ArtifactValidator::new()
            .validate(&model_directory, u32::from(OUTPUT_TOKEN_COUNT))
            .expect("the configured sparse artifact should validate");
        assert_eq!(validated_artifact.model_id(), model_id());
        let model_id = validated_artifact.model_id().to_owned();
        let (model_geometry, required_headroom_bytes) =
            mlx_ram_budget_model_geometry_from_validated_artifact(
                &validated_artifact,
                &model_directory,
            )
            .expect("the sparse artifact should expose disk RAM geometry");
        let residency_boundary = CompleteResidencyHeadroomBoundary::from_model_geometry(
            model_geometry,
            required_headroom_bytes,
        );
        let paging_ceiling_bytes = residency_boundary.paging_ceiling_bytes().expect(
            "this sparse artifact must have a ceiling that fits expert weights but not activation headroom",
        );
        let paging_ceiling_limit_bytes = usize::try_from(paging_ceiling_bytes)
            .expect("the paging ceiling should fit usize");
        eprintln!(
            "[paged-decode] status=progress phase=derived_ceiling expert_bytes={} headroom_bytes={required_headroom_bytes} paging_ceiling_bytes={paging_ceiling_bytes}",
            model_geometry.complete_expert_payload_bytes
        );
        let romeo_and_juliet_prompt = prepare_romeo_and_juliet_three_paragraph_summary_prompt(
            &model_directory,
            &model_id,
            COLD_REQUEST_ID,
            PROMPT_TOKEN_COUNT,
            OUTPUT_TOKEN_COUNT,
        );
        let temporary_attribution_directory = tempfile::tempdir()
            .expect("the paged decode journey should create an attribution directory");
        let attribution_log_path = temporary_attribution_directory
            .path()
            .join("paged-decode.jsonl");
        let mlx_memory_limits = MlxMemoryLimits::new(
            paging_ceiling_limit_bytes,
            paging_ceiling_limit_bytes,
        )
        .expect("the paging ceiling should be a valid MLX limit");
        let (mut qwen3_5_engine, end_of_sequence_token_ids) = create_attributed_engine(
            &model_directory,
            &attribution_log_path,
            &mlx_memory_limits,
            PROMPT_TOKEN_COUNT as u32,
        );

        eprintln!("[paged-decode 0/4] status=progress phase=model_load ETA_seconds=90");
        load_engine_with_progress(&mut qwen3_5_engine, "paged_decode_model_load").await;
        assert_ne!(
            qwen3_5_engine
                .expert_memory_mode_for_tests()
                .await
                .expect("the loaded model should expose its expert mode"),
            Some(ExpertMemoryMode::Resident),
            "the paging ceiling must not seat complete experts"
        );
        eprintln!("[paged-decode 1/4] status=progress phase=first_romeo");
        let first_generated_token_ids = run_attributed_generation(
            &mut qwen3_5_engine,
            COLD_REQUEST_ID,
            &romeo_and_juliet_prompt.prompt_token_ids,
            "paged_decode_first_romeo",
            OUTPUT_TOKEN_COUNT,
            &end_of_sequence_token_ids,
        )
        .await;
        eprintln!("[paged-decode 2/4] status=progress phase=second_romeo");
        let second_generated_token_ids = run_attributed_generation(
            &mut qwen3_5_engine,
            WARM_REQUEST_ID,
            &romeo_and_juliet_prompt.prompt_token_ids,
            "paged_decode_second_romeo",
            OUTPUT_TOKEN_COUNT,
            &end_of_sequence_token_ids,
        )
        .await;
        assert!(
            !first_generated_token_ids.is_empty(),
            "the first Romeo ask must emit at least one token"
        );
        assert_eq!(
            second_generated_token_ids, first_generated_token_ids,
            "the second Romeo ask must not change the continuation"
        );

        let attribution_report_documents =
            read_attribution_report_documents(&attribution_log_path);
        let first_generation_report = generation_report_for_request(
            &attribution_report_documents,
            COLD_REQUEST_ID.value(),
        );
        let second_generation_report = generation_report_for_request(
            &attribution_report_documents,
            WARM_REQUEST_ID.value(),
        );
        let first_disk_page_load_count = counter_amount(
            first_generation_report,
            "positional_file_read_call_count",
        );
        let second_disk_page_load_count = counter_amount(
            second_generation_report,
            "positional_file_read_call_count",
        );
        let first_memory_evidence =
            assert_memory_within_policy(first_generation_report, paging_ceiling_bytes);
        let second_memory_evidence =
            assert_memory_within_policy(second_generation_report, paging_ceiling_bytes);
        eprintln!(
            "[paged-decode 3/4] status=progress first_disk_page_load_count={first_disk_page_load_count} second_disk_page_load_count={second_disk_page_load_count} first_active_plus_allocator_bytes={} second_active_plus_allocator_bytes={} observed_peak_bytes={} allowed_peak_bytes={}",
            first_memory_evidence.active_plus_allocator_bytes,
            second_memory_evidence.active_plus_allocator_bytes,
            first_memory_evidence
                .peak_bytes
                .max(second_memory_evidence.peak_bytes),
            first_memory_evidence.allowed_peak_bytes,
        );
        assert!(
            second_disk_page_load_count <= first_disk_page_load_count,
            "the second Romeo ask must not read more expert pages than the first"
        );
        eprintln!("[paged-decode 4/4] status=success");
    })
    .await
    .expect("paged Romeo decode must finish within 115 seconds");
}

struct MemoryPolicyEvidence {
    active_plus_allocator_bytes: u64,
    peak_bytes: u64,
    allowed_peak_bytes: u64,
}

fn assert_memory_within_policy(
    generation_report: &Value,
    configured_mlx_memory_ceiling_bytes: u64,
) -> MemoryPolicyEvidence {
    let active_memory_bytes = generation_report["mlx_active_memory_bytes"]
        .as_u64()
        .expect("successful generation should report MLX active memory");
    let allocator_cache_memory_bytes = generation_report["mlx_allocator_cache_memory_bytes"]
        .as_u64()
        .expect("successful generation should report MLX allocator-cache memory");
    let active_plus_allocator_bytes = active_memory_bytes
        .checked_add(allocator_cache_memory_bytes)
        .expect("active and allocator-cache memory should fit u64");
    assert!(
        active_plus_allocator_bytes <= configured_mlx_memory_ceiling_bytes,
        "stable MLX residency {active_plus_allocator_bytes} exceeded the configured ceiling {configured_mlx_memory_ceiling_bytes}"
    );
    let peak_bytes = generation_report["mlx_peak_memory_bytes"]
        .as_u64()
        .expect("successful generation should report MLX peak memory");
    let allowed_peak_bytes = configured_mlx_memory_ceiling_bytes
        .saturating_add(configured_mlx_memory_ceiling_bytes / 100);
    assert!(
        peak_bytes <= allowed_peak_bytes,
        "MLX peak memory {peak_bytes} exceeded the one-percent transient policy limit {allowed_peak_bytes}"
    );
    MemoryPolicyEvidence {
        active_plus_allocator_bytes,
        peak_bytes,
        allowed_peak_bytes,
    }
}
