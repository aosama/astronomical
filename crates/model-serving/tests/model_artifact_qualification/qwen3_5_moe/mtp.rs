use std::time::Duration;

use crate::model_artifact_qualification::qwen3_5::mtp_support::run_one_layer_mtp_head_forward_qualification;

mod benchmark;
mod benchmark_measurement;
mod engine_support;
mod lifecycle;
mod model_loading;

use engine_support::{generate_with_mtp_engine, performance_counter_amount};

const REJECTION_QUALIFICATION_OUTPUT_TOKEN_COUNT: u16 = 128;

#[tokio::test]
#[ignore = "loads a complete configured depth-one MTP artifact and evaluates its MTP head"]
async fn should_evaluate_the_configured_mtp_head_from_target_pre_normalization_hidden_states() {
    tokio::time::timeout(
        Duration::from_secs(120),
        run_one_layer_mtp_head_forward_qualification(
            super::configured_depth_one_mtp_model_artifact_directory(),
            "configured-moe-mtp-head",
        ),
    )
    .await
    .expect("the configured MTP head qualification should finish within 120 seconds");
}

#[tokio::test]
#[ignore = "loads a complete configured MTP artifact and forces target-verified rejection"]
async fn should_recover_from_a_rejected_depth_one_mtp_draft_without_operational_fallback() {
    tokio::time::timeout(
        Duration::from_secs(120),
        run_rejected_mtp_draft_qualification(),
    )
    .await
    .expect("the rejected MTP draft qualification should finish within 120 seconds");
}

async fn run_rejected_mtp_draft_qualification() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let model_directory = super::configured_depth_one_mtp_model_artifact_directory();
    eprintln!("[mtp-rejection] status=start ETA_seconds=120");
    let (generated_token_ids, generation_report) = generate_with_mtp_engine(
        &model_directory,
        REJECTION_QUALIFICATION_OUTPUT_TOKEN_COUNT,
        true,
    )
    .await;

    let admitted_attempt_count =
        performance_counter_amount(&generation_report, "mtp_admitted_attempt_count");
    let accepted_draft_count =
        performance_counter_amount(&generation_report, "mtp_accepted_draft_count");
    let rejected_draft_count =
        performance_counter_amount(&generation_report, "mtp_rejected_draft_count");
    let operational_fallback_count =
        performance_counter_amount(&generation_report, "mtp_operational_fallback_count");
    let memory_admission_fallback_count =
        performance_counter_amount(&generation_report, "mtp_memory_admission_fallback_count");
    assert!(!generated_token_ids.is_empty());
    assert!(admitted_attempt_count > 0);
    assert!(rejected_draft_count >= 1);
    assert_eq!(operational_fallback_count, 0);
    assert_eq!(memory_admission_fallback_count, 0);
    assert_eq!(
        accepted_draft_count + rejected_draft_count + operational_fallback_count,
        admitted_attempt_count,
    );
    eprintln!(
        "[mtp-rejection] status=success output_tokens={} admitted_attempts={} accepted_drafts={} rejected_drafts={}",
        generated_token_ids.len(),
        admitted_attempt_count,
        accepted_draft_count,
        rejected_draft_count,
    );
}
