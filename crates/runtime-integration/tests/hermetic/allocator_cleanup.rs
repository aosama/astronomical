const MLX_RUNTIME_MEMORY_POLICY_SOURCE: &str =
    include_str!("../../src/mlx_runtime/memory_policy.rs");
const QWEN_MEMORY_ADMISSION_SOURCE: &str =
    include_str!("../../../model-serving/src/qwen3_5/inference_execution/memory_admission.rs");
const EXPERT_PAGING_MEMORY_BUDGET_SOURCE: &str =
    include_str!("../../../model-serving/src/expert_paging/memory_budget.rs");
const QWEN_PREFILL_ADVANCE_SOURCE: &str =
    include_str!("../../../model-serving/src/qwen3_5/inference_execution/prefill_advance.rs");
const QWEN_ADVANCE_GENERATION_SOURCE: &str =
    include_str!("../../../model-serving/src/qwen3_5/inference_execution/advance_generation.rs");
const QWEN_INJECT_INPUT_TOKENS_SOURCE: &str =
    include_str!("../../../model-serving/src/qwen3_5/inference_execution/inject_input_tokens.rs");
const QWEN_GENERATION_FINALIZATION_SOURCE: &str = include_str!(
    "../../../model-serving/src/qwen3_5/inference_execution/generation_finalization.rs"
);

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
    let request_cleanup_start = QWEN_MEMORY_ADMISSION_SOURCE
        .find("fn release_request_memory")
        .expect("the Qwen engine must retain request-boundary cleanup");
    let request_cleanup_source = &QWEN_MEMORY_ADMISSION_SOURCE[request_cleanup_start..];

    assert!(
        request_cleanup_source.contains("synchronize_gpu_stream_and_clear_allocator_cache"),
        "request-boundary cleanup must wait for one-token-ahead decode work"
    );
}

#[test]
fn should_synchronize_in_flight_expert_pages_before_reclaiming_allocator_memory() {
    let live_budget_check_start = EXPERT_PAGING_MEMORY_BUDGET_SOURCE
        .find("pub fn check(")
        .expect("the live expert paging budget must retain its checked admission method");
    let live_budget_snapshot_start = EXPERT_PAGING_MEMORY_BUDGET_SOURCE[live_budget_check_start..]
        .find("pub fn snapshot(")
        .map(|relative_snapshot_start| live_budget_check_start + relative_snapshot_start)
        .expect("the live expert paging budget must retain its snapshot method");
    let live_budget_check_source =
        &EXPERT_PAGING_MEMORY_BUDGET_SOURCE[live_budget_check_start..live_budget_snapshot_start];

    assert!(
        live_budget_check_source.contains("synchronize_gpu_stream_and_clear_allocator_cache"),
        "expert-page rejection recovery must synchronize in-flight GPU pages before clearing allocator memory"
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

    let mtp_outcome_start = QWEN_ADVANCE_GENERATION_SOURCE
        .find("record_mtp_outcome(active_request")
        .expect("generation advance must record the MTP verification outcome");
    let target_only_decode_start = QWEN_ADVANCE_GENERATION_SOURCE[mtp_outcome_start..]
        .find("let active_memory_bytes_before_growth = self.measure_adaptive_ram_growth_memory_admission(")
        .map(|relative_decode_start| mtp_outcome_start + relative_decode_start)
        .expect("generation advance must retain target-only decode fallback");
    let mtp_outcome_source =
        &QWEN_ADVANCE_GENERATION_SOURCE[mtp_outcome_start..target_only_decode_start];
    let successful_mtp_condition_position = mtp_outcome_source
        .find("mtp_prefix_acceptance_outcome != MtpPrefixAcceptanceOutcome::OperationalFallback")
        .expect("MTP output must remain conditional on successful verification");
    let published_mtp_snapshot_position = mtp_outcome_source
        .find("collect_completed_forward_memory_snapshot(")
        .expect("successful MTP output must publish its completed-forward snapshot");
    assert!(
        successful_mtp_condition_position < published_mtp_snapshot_position,
        "MTP fallback must not collect a snapshot that target-only decode will replace"
    );
    assert!(
        mtp_outcome_source.contains("record_completed_adaptive_ram_growth("),
        "MTP fallback must retain adaptive learning when admission is enabled"
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
