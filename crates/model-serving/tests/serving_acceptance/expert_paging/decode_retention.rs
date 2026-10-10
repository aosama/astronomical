//! Low-level retention contract for paged expert streaming on the large sparse MoE.
//!
//! The memory-ceiling sweep measured ~40 GB of request-time positional reads for a
//! hybrid cell whose un-retained expert payload is ~4.3 GB — roughly five times the
//! mandatory stream. This test isolates that defect below the journey layer: one
//! attributed in-process engine, two identical paged requests with a real
//! large-prompt/long-decode shape, and byte-level assertions that a warm request
//! must re-stream far less than the cold one pays for first touch.

use std::time::Duration;

use astronomical_ipc_protocol::{ExpertMemoryMode, RequestId};
use astronomical_model_serving::{
    CompleteResidencyHeadroomBoundary, Qwen3_5ArtifactValidator,
    mlx_ram_budget_model_geometry_from_validated_artifact,
};
use astronomical_runtime_integration::MlxMemoryLimits;
use serial_test::serial;
use tokio::time::timeout;

use crate::serving_acceptance::support::performance_attribution::{
    counter_amount, create_attributed_engine, generation_report_for_request,
    load_engine_with_progress, read_attribution_report_documents, run_attributed_generation,
};

const PROMPT_TOKEN_COUNT: usize = 2_000;
const OUTPUT_TOKEN_COUNT: u16 = 100;
const COLD_REQUEST_ID: RequestId = RequestId::new(96_300);
const WARM_REQUEST_ID: RequestId = RequestId::new(96_301);
const RETENTION_CONTRACT_TIMEOUT: Duration = Duration::from_secs(60);

fn model_id() -> &'static str {
    crate::common::large_sparse_moe_model_id()
}

#[tokio::test]
#[ignore = "loads the large sparse MoE in-process and proves warm paged requests re-stream far fewer expert bytes than cold ones"]
#[serial]
async fn should_not_restream_expert_pages_on_a_warm_paged_request() {
    timeout(RETENTION_CONTRACT_TIMEOUT, async {
        let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
        let model_directory =
            crate::common::configured_installed_model_directory_by_id(model_id());
        let validated_artifact = Qwen3_5ArtifactValidator::new()
            .validate(&model_directory, u32::from(OUTPUT_TOKEN_COUNT))
            .expect("the sparse artifact should validate for the retention contract");
        let (model_geometry, required_headroom_bytes) =
            mlx_ram_budget_model_geometry_from_validated_artifact(
                &validated_artifact,
                &model_directory,
            )
            .expect("the sparse artifact should expose RAM geometry");
        let residency_boundary = CompleteResidencyHeadroomBoundary::from_model_geometry(
            model_geometry,
            required_headroom_bytes,
        );
        let paging_ceiling_bytes = residency_boundary.paging_ceiling_bytes().expect(
            "the sparse artifact must have a ceiling that pages experts while serving",
        );
        let paging_ceiling_limit_bytes =
            usize::try_from(paging_ceiling_bytes).expect("the paging ceiling should fit usize");
        let romeo_and_juliet_prompt = crate::serving_acceptance::support::romeo_and_juliet::prepare_romeo_and_juliet_three_paragraph_summary_prompt(
            &model_directory,
            model_id(),
            COLD_REQUEST_ID,
            PROMPT_TOKEN_COUNT,
            OUTPUT_TOKEN_COUNT,
        );
        let temporary_attribution_directory = tempfile::tempdir()
            .expect("the retention contract should create an attribution directory");
        let attribution_log_path = temporary_attribution_directory.path().join("retention.jsonl");
        let mlx_memory_limits =
            MlxMemoryLimits::new(paging_ceiling_limit_bytes, paging_ceiling_limit_bytes)
                .expect("the paging ceiling should be a valid MLX limit");
        let (mut engine, end_of_sequence_token_ids) = create_attributed_engine(
            &model_directory,
            &attribution_log_path,
            &mlx_memory_limits,
            PROMPT_TOKEN_COUNT as u32,
        );
        eprintln!("[decode-retention] status=progress phase=model_load");
        load_engine_with_progress(&mut engine, "decode_retention_model_load").await;
        assert_ne!(
            engine
                .expert_memory_mode_for_tests()
                .await
                .expect("the loaded model should expose its expert mode"),
            Some(ExpertMemoryMode::Resident),
            "the retention contract requires the paging regime"
        );

        eprintln!("[decode-retention] status=progress phase=cold_request");
        let cold_generated_token_ids = run_attributed_generation(
            &mut engine,
            COLD_REQUEST_ID,
            &romeo_and_juliet_prompt,
            "decode_retention_cold_request",
            OUTPUT_TOKEN_COUNT,
            &end_of_sequence_token_ids,
        )
        .await;
        eprintln!("[decode-retention] status=progress phase=warm_request");
        let warm_generated_token_ids = run_attributed_generation(
            &mut engine,
            WARM_REQUEST_ID,
            &romeo_and_juliet_prompt,
            "decode_retention_warm_request",
            OUTPUT_TOKEN_COUNT,
            &end_of_sequence_token_ids,
        )
        .await;
        assert!(
            !cold_generated_token_ids.is_empty() && !warm_generated_token_ids.is_empty(),
            "both paged requests must generate tokens"
        );
        assert_eq!(
            warm_generated_token_ids, cold_generated_token_ids,
            "the warm request must produce the identical continuation"
        );

        let attribution_report_documents =
            read_attribution_report_documents(&attribution_log_path);
        let cold_report =
            generation_report_for_request(&attribution_report_documents, COLD_REQUEST_ID.value());
        let warm_report =
            generation_report_for_request(&attribution_report_documents, WARM_REQUEST_ID.value());
        let cold_positional_bytes = counter_amount(cold_report, "positional_file_read_byte_count");
        let warm_positional_bytes = counter_amount(warm_report, "positional_file_read_byte_count");
        eprintln!(
            "[decode-retention] status=evidence cold_positional_bytes={cold_positional_bytes} warm_positional_bytes={warm_positional_bytes} ratio={:.2}",
            warm_positional_bytes as f64 / cold_positional_bytes.max(1) as f64
        );

        assert!(
            warm_positional_bytes < cold_positional_bytes,
            "a warm paged request must read fewer expert bytes than the cold first touch: warm={warm_positional_bytes} cold={cold_positional_bytes}"
        );
        assert!(
            warm_positional_bytes <= cold_positional_bytes / 2,
            "expert retention is re-streaming: the warm request read {warm_positional_bytes} bytes, more than half the cold request's {cold_positional_bytes}"
        );
        eprintln!("[decode-retention] status=success");
    })
    .await
    .expect("the expert retention contract must finish within 60 seconds");
}
