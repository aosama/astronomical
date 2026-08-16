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
const QWEN_MTP_DECODE_ATTEMPT_SOURCE: &str =
    include_str!("../../../model-serving/src/qwen3_5/inference_execution/mtp_decode_attempt.rs");
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

    let mtp_prediction_attempt_start = QWEN_MTP_DECODE_ATTEMPT_SOURCE
        .find("attempt_prediction_proposal_and_verification(")
        .expect("generation advance must invoke MTP proposal and verification");
    let mtp_attempt_source = &QWEN_MTP_DECODE_ATTEMPT_SOURCE[mtp_prediction_attempt_start..];
    let operational_fallback_condition_position = mtp_attempt_source
        .find("if decision.is_operational_fallback()")
        .expect("MTP output must classify operational fallback through its typed decision");
    let successful_mtp_snapshot_position = mtp_attempt_source
        .find("collect_completed_forward_memory_snapshot(")
        .expect("successful MTP output must publish its completed-forward snapshot");
    let operational_fallback_source = &mtp_attempt_source
        [operational_fallback_condition_position..successful_mtp_snapshot_position];
    assert!(
        operational_fallback_source.contains("record_completed_adaptive_ram_growth("),
        "MTP fallback must retain adaptive learning when admission is enabled"
    );
    assert!(
        operational_fallback_source.contains("return Ok(None);"),
        "MTP operational fallback must continue through the ordinary target-only decode path"
    );
    assert!(
        !operational_fallback_source.contains("collect_completed_forward_memory_snapshot("),
        "target-only MTP fallback must not sample unpublished forwards before adaptive accounting"
    );
    assert!(
        operational_fallback_condition_position < successful_mtp_snapshot_position,
        "MTP fallback must be excluded before successful output samples memory"
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
