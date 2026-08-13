const MLX_RUNTIME_MEMORY_POLICY_SOURCE: &str =
    include_str!("../../src/mlx_runtime/memory_policy.rs");
const QWEN_REQUEST_MEMORY_RELEASE_SOURCE: &str = include_str!(
    "../../../model-serving/src/qwen3_5/inference_execution/request_memory_release.rs"
);
const EXPERT_PAGING_LAYER_STREAMING_SOURCE: &str = include_str!(
    "../../../model-serving/src/qwen3_5_moe/expert_paging/expert_pager/rust_layer_streaming.rs"
);
const QWEN_PREFILL_ADVANCE_SOURCE: &str =
    include_str!("../../../model-serving/src/qwen3_5/inference_execution/prefill_advance.rs");
const QWEN_MTP_DECODE_SOURCE: &str =
    include_str!("../../../model-serving/src/qwen3_5/multi_token_prediction/decode.rs");
const QWEN_ADVANCE_GENERATION_SOURCE: &str =
    include_str!("../../../model-serving/src/qwen3_5/inference_execution/advance_generation.rs");
const QWEN_INJECT_INPUT_TOKENS_SOURCE: &str =
    include_str!("../../../model-serving/src/qwen3_5/inference_execution/inject_input_tokens.rs");
const QWEN_GENERATION_FINALIZATION_SOURCE: &str = include_str!(
    "../../../model-serving/src/qwen3_5/inference_execution/generation_finalization.rs"
);

#[test]
fn should_leave_mlx_per_buffer_wired_residency_disabled() {
    assert!(
        !MLX_RUNTIME_MEMORY_POLICY_SOURCE.contains("raw::mlx_set_wired_limit"),
        "production memory policy must not enable the MLX residency-set path that can panic IOGPU during allocation-pressure reclamation"
    );
}

#[test]
fn should_synchronize_gpu_stream_before_clearing_allocator_cache() {
    let synchronized_cleanup_start = MLX_RUNTIME_MEMORY_POLICY_SOURCE
        .find("pub fn synchronize_gpu_stream_and_clear_allocator_cache")
        .expect("the runtime must expose synchronized request-boundary cleanup");
    let synchronized_cleanup_source =
        &MLX_RUNTIME_MEMORY_POLICY_SOURCE[synchronized_cleanup_start..];
    let synchronization_position = synchronized_cleanup_source
        .find("self.synchronize_gpu_stream()?")
        .expect("request-boundary cleanup must synchronize the GPU stream");
    let allocator_cleanup_position = synchronized_cleanup_source
        .find("self.clear_allocator_cache()")
        .expect("request-boundary cleanup must clear reclaimable allocator memory");

    assert!(
        synchronization_position < allocator_cleanup_position,
        "request-boundary cleanup must synchronize before clearing allocator memory"
    );
}

#[test]
fn should_use_synchronized_cleanup_when_releasing_request_memory() {
    let request_cleanup_start = QWEN_REQUEST_MEMORY_RELEASE_SOURCE
        .find("fn release_request_memory")
        .expect("the Qwen engine must retain request-boundary cleanup");
    let request_cleanup_source = &QWEN_REQUEST_MEMORY_RELEASE_SOURCE[request_cleanup_start..];

    assert!(
        request_cleanup_source.contains("synchronize_gpu_stream_and_clear_allocator_cache"),
        "request-boundary cleanup must wait for one-token-ahead decode work"
    );
}

#[test]
fn should_synchronize_in_flight_expert_pages_before_reclaiming_allocator_memory() {
    assert!(
        EXPERT_PAGING_LAYER_STREAMING_SOURCE
            .contains("synchronize_gpu_stream_and_clear_allocator_cache"),
        "expert-page rejection recovery must synchronize in-flight GPU pages before clearing allocator memory"
    );
    assert!(
        EXPERT_PAGING_LAYER_STREAMING_SOURCE
            .contains("AllocationAdmissionDecision::ClearAllocatorCacheThenAdmit"),
        "expert paging must execute allocator cleanup only when memory policy requests it"
    );
}

#[test]
fn should_synchronize_completed_prefill_work_before_clearing_allocator_memory() {
    let prefill_execution_start = QWEN_PREFILL_ADVANCE_SOURCE
        .find(".execute_prompt_prefill_chunck(")
        .expect("prefill advance must execute prompt-processing chunks");
    let completed_forward_snapshot_start = QWEN_PREFILL_ADVANCE_SOURCE[prefill_execution_start..]
        .find("collect_completed_forward_memory_snapshot(")
        .map(|relative_snapshot_start| prefill_execution_start + relative_snapshot_start)
        .expect("prefill advance must sample memory after each completed prefill forward");
    let completed_prefill_cleanup_source =
        &QWEN_PREFILL_ADVANCE_SOURCE[prefill_execution_start..completed_forward_snapshot_start];

    assert!(
        completed_prefill_cleanup_source
            .contains("synchronize_gpu_stream_and_clear_allocator_cache"),
        "per-prefill cleanup must synchronize submitted GPU and Metal I/O work before releasing allocator memory"
    );
}

#[test]
fn should_not_sample_disabled_adaptive_memory_for_unpublished_forwards() {
    assert!(
        QWEN_INJECT_INPUT_TOKENS_SOURCE.contains("record_completed_adaptive_ram_growth("),
        "feedback forwards whose telemetry is not published must retain the record-only path"
    );
    assert!(
        !QWEN_INJECT_INPUT_TOKENS_SOURCE.contains("collect_completed_forward_memory_snapshot("),
        "feedback forwards must not perform fallible system sampling only to discard it"
    );

    let mtp_attempt_definition_start = QWEN_MTP_DECODE_SOURCE
        .find("fn attempt_prediction_proposal_and_verification")
        .expect("multi-token prediction module must expose MTP proposal and verification flow");
    let mtp_outcome_recording_call = QWEN_MTP_DECODE_SOURCE[mtp_attempt_definition_start..]
        .find("record_mtp_outcome(active_request")
        .map(|relative_recording_start| mtp_attempt_definition_start + relative_recording_start)
        .expect("MTP attempts must record acceptance outcomes");
    let mtp_attempt_source =
        &QWEN_MTP_DECODE_SOURCE[mtp_attempt_definition_start..mtp_outcome_recording_call];
    assert!(
        mtp_attempt_source.contains("Qwen3_5PredictionAcceptanceOutcome::OperationalFallback"),
        "MTP outcome matching must classify operational fallback distinctly"
    );

    let mtp_prediction_attempt_start = QWEN_ADVANCE_GENERATION_SOURCE
        .find("attempt_prediction_proposal_and_verification(")
        .expect("generation advance must invoke MTP proposal and verification");
    let mtp_match_start = QWEN_ADVANCE_GENERATION_SOURCE[mtp_prediction_attempt_start..]
        .find("match prediction_attempt_outcome")
        .map(|relative_match_start| mtp_prediction_attempt_start + relative_match_start)
        .expect("generation advance must branch on prediction attempt outcomes");
    // Search for the semantic call boundary rather than its rustfmt-sensitive
    // assignment layout so formatting cannot invalidate this structural contract.
    let target_only_decode_measurement_start = QWEN_ADVANCE_GENERATION_SOURCE[mtp_match_start..]
        .find("measure_adaptive_ram_growth_memory_admission(")
        .map(|relative_decode_start| mtp_match_start + relative_decode_start)
        .expect("generation advance must retain target-only decode fallback");
    let mtp_match_source =
        &QWEN_ADVANCE_GENERATION_SOURCE[mtp_match_start..target_only_decode_measurement_start];

    let successful_mtp_condition_position = mtp_match_source
        .find("if prediction_acceptance_outcome")
        .expect("MTP output must remain conditional on successful verification");
    let successful_mtp_snapshot_position = mtp_match_source
        .find("collect_completed_forward_memory_snapshot(")
        .expect("successful MTP output must publish its completed-forward snapshot");
    let successful_mtp_condition_source =
        &mtp_match_source[successful_mtp_condition_position..successful_mtp_snapshot_position];
    assert!(
        successful_mtp_condition_source.contains("!="),
        "MTP output must retain an acceptance check"
    );
    assert!(
        successful_mtp_condition_source
            .contains("Qwen3_5PredictionAcceptanceOutcome::OperationalFallback"),
        "MTP output must exclude operational fallback"
    );
    assert!(
        successful_mtp_condition_position < successful_mtp_snapshot_position,
        "MTP fallback must not collect a snapshot that target-only decode will replace"
    );

    let operational_fallback_decision_start = mtp_match_source
        .find("Ok((_prediction_acceptance_outcome, target_verification_was_attempted)) =>")
        .expect("MTP fallback must retain target-only decode branch");
    let fallback_error_start = mtp_match_source
        .find("Err(target_verification_error) =>")
        .expect("MTP fallback and verification errors must be surfaced");
    let target_only_fallback_branch =
        &mtp_match_source[operational_fallback_decision_start..fallback_error_start];
    assert!(
        !target_only_fallback_branch.contains("collect_completed_forward_memory_snapshot("),
        "target-only MTP fallback must not sample unpublished forwards before adaptive accounting"
    );
    assert!(
        target_only_fallback_branch.contains("record_completed_adaptive_ram_growth("),
        "MTP fallback must retain adaptive learning when admission is enabled"
    );

    let verification_error_source = &mtp_match_source[fallback_error_start..];
    assert!(
        verification_error_source.contains("return Err(target_verification_error);"),
        "target verification failures must continue through admission outcome recording"
    );
}

#[test]
fn should_collect_finalized_memory_independently_of_adaptive_admission() {
    let finalization_start = QWEN_GENERATION_FINALIZATION_SOURCE
        .find("fn collect_generation_finalization(")
        .expect("generation finalization must retain its memory publication boundary");
    let finalized_snapshot_start = QWEN_GENERATION_FINALIZATION_SOURCE[finalization_start..]
        .find("PerformanceOperation::FinalizedMlxMemorySnapshot")
        .map(|relative_snapshot_start| finalization_start + relative_snapshot_start)
        .expect("generation finalization must sample post-cleanup MLX memory");
    let finalization_admission_gate_source =
        &QWEN_GENERATION_FINALIZATION_SOURCE[finalization_start..finalized_snapshot_start];

    assert!(
        !finalization_admission_gate_source.contains("adaptive_ram_growth_guard_enabled"),
        "post-cleanup telemetry must not depend on adaptive admission being enabled"
    );
}
