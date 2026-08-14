use std::fs;

use astronomical_ipc_protocol::{
    WorkerPersistentPromptCacheExpectedBlockHashPrefix, WorkerPersistentPromptCacheLookupOutcome,
    WorkerPersistentPromptCacheMissReason, WorkerPersistentPromptCacheRequestDiagnostics,
    WorkerPersistentPromptCacheStartupCleanupCategory,
    WorkerPersistentPromptCacheStartupCleanupEvidence,
};
use astronomical_supervisor::{GenerationPerformanceLog, GenerationPerformanceRecord};

#[test]
fn should_create_generation_performance_record_with_computed_throughput() {
    let (prefill_tps, generation_tps) = GenerationPerformanceRecord::compute_throughput(
        /* prompt_token_count */ 100_000, /* cached_token_count */ 50_000,
        /* generated_token_count */ 500, /* prefill_elapsed_millis */ 1_000,
        /* generation_elapsed_millis */ 5_000,
    );
    // 50,000 uncached tokens / 1.0 seconds = 50,000 tok/s
    assert_eq!(prefill_tps, Some(50_000.0));
    // 500 tokens / 5.0 seconds = 100 tok/s
    assert_eq!(generation_tps, Some(100.0));
}

#[test]
fn should_return_none_for_prefill_tok_per_second_when_fully_cached() {
    // When all tokens are cached (0 ms prefill elapsed), prefill TPS is None.
    let (prefill_tps, generation_tps) = GenerationPerformanceRecord::compute_throughput(
        /* prompt_token_count */ 10_000, /* cached_token_count */ 10_000,
        /* generated_token_count */ 200, /* prefill_elapsed_millis */ 0,
        /* generation_elapsed_millis */ 2_000,
    );
    assert!(
        prefill_tps.is_none(),
        "fully cached prompt should have None prefill TPS"
    );
    assert_eq!(generation_tps, Some(100.0)); // 200 tokens / 2.0 seconds
}

#[test]
fn should_return_none_for_generation_tok_per_second_when_zero_elapsed() {
    let (prefill_tps, generation_tps) = GenerationPerformanceRecord::compute_throughput(
        /* prompt_token_count */ 5_000, /* cached_token_count */ 0,
        /* generated_token_count */ 1, /* prefill_elapsed_millis */ 500,
        /* generation_elapsed_millis */ 0,
    );
    assert_eq!(prefill_tps, Some(10_000.0)); // 5,000 tokens / 0.5 seconds
    assert!(
        generation_tps.is_none(),
        "0 ms generation should have None generation TPS"
    );
}

#[test]
fn should_return_none_for_prefill_tps_when_uncached_tokens_is_zero() {
    // If prompt_token_count == cached_token_count, uncached = 0, so TPS is None
    // even if prefill_elapsed_millis > 0 (which shouldn't happen in practice, but
    // the math should be robust).
    let (prefill_tps, _) = GenerationPerformanceRecord::compute_throughput(
        /* prompt_token_count */ 1_000, /* cached_token_count */ 1_000,
        /* generated_token_count */ 100, /* prefill_elapsed_millis */ 500,
        /* generation_elapsed_millis */ 1_000,
    );
    assert!(
        prefill_tps.is_none(),
        "zero uncached tokens should produce None prefill TPS"
    );
}

#[test]
fn should_return_none_for_generation_tps_when_token_count_is_zero() {
    let (_, generation_tps) = GenerationPerformanceRecord::compute_throughput(
        /* prompt_token_count */ 1_000, /* cached_token_count */ 0,
        /* generated_token_count */ 0, /* prefill_elapsed_millis */ 100,
        /* generation_elapsed_millis */ 1_000,
    );
    assert!(
        generation_tps.is_none(),
        "zero generated tokens should produce None generation TPS"
    );
}

#[test]
fn should_open_and_append_to_performance_log() {
    let temp_dir = tempfile::tempdir().expect("test directory should be created");
    let mut performance_log =
        GenerationPerformanceLog::open(temp_dir.path()).expect("performance log should open");

    let record = GenerationPerformanceRecord {
        timestamp_millis: 1_700_000_000_000,
        request_id: 42,
        model_id: "test-model".to_owned(),
        prompt_token_count: 10_000,
        cached_token_count: 5_000,
        generated_token_count: 200,
        completion_reason: "end_of_sequence".to_owned(),
        prefill_elapsed_millis: 1_000,
        generation_elapsed_millis: 2_000,
        total_elapsed_millis: 3_500,
        time_to_first_output_millis: Some(1_750),
        generation_preparation_elapsed_millis: Some(250),
        first_decode_forward_elapsed_millis: Some(83),
        generation_preparation_expert_source_read_byte_count: 0,
        final_complete_expert_layer_count: Some(20),
        final_complete_expert_payload_bytes: Some(10_000_000_000),
        final_partial_expert_layer_count: Some(20),
        final_partial_expert_payload_bytes: Some(1_000_000_000),
        prefill_tok_per_second: Some(5_000.0),
        generation_tok_per_second: Some(100.0),
        mlx_peak_memory_bytes: Some(40_000_000_000),
        mlx_active_memory_bytes: Some(31_000_000_000),
        persistent_prompt_cache_diagnostics: Some(WorkerPersistentPromptCacheRequestDiagnostics {
            lookup_outcome: WorkerPersistentPromptCacheLookupOutcome::Hit,
            block_token_count: 2_048,
            complete_prompt_block_count: 5,
            maximum_restorable_block_count: 4,
            matched_sequence_state_block_count: 4,
            restored_block_count: 4,
            first_missing_sequence_state_block_index: None,
            miss_reason: None,
            expected_block_hash_prefix: None,
            startup_cleanup_evidence: Some(WorkerPersistentPromptCacheStartupCleanupEvidence {
                interrupted_transaction_recovery:
                    WorkerPersistentPromptCacheStartupCleanupCategory {
                        artifact_count: 1,
                        block_count: 0,
                        byte_count: 128,
                    },
                obsolete_format: WorkerPersistentPromptCacheStartupCleanupCategory {
                    artifact_count: 2,
                    block_count: 0,
                    byte_count: 256,
                },
                corrupt_current_format: WorkerPersistentPromptCacheStartupCleanupCategory {
                    artifact_count: 0,
                    block_count: 1,
                    byte_count: 512,
                },
                quota_eviction: WorkerPersistentPromptCacheStartupCleanupCategory {
                    artifact_count: 0,
                    block_count: 2,
                    byte_count: 1_024,
                },
            }),
            published_block_count: 1,
            allocator_bytes_cleared_for_publication: 4096,
            expert_bytes_reclaimed_for_publication: 8192,
            expert_bytes_reclaimed_for_restore: 16_384,
        }),
    };

    performance_log.record(&record);

    let performance_log_path = temp_dir.path().join("performance.jsonl");
    let contents =
        fs::read_to_string(&performance_log_path).expect("performance log file should be readable");
    let lines: Vec<&str> = contents.trim().lines().collect();
    assert_eq!(lines.len(), 1, "should have exactly one line");

    let parsed: serde_json::Value =
        serde_json::from_str(lines[0]).expect("performance log line should be valid JSON");
    assert_eq!(parsed["request_id"], 42);
    assert_eq!(parsed["model_id"], "test-model");
    assert_eq!(parsed["prompt_token_count"], 10_000);
    assert_eq!(parsed["cached_token_count"], 5_000);
    assert_eq!(parsed["generated_token_count"], 200);
    assert_eq!(parsed["completion_reason"], "end_of_sequence");
    assert_eq!(parsed["prefill_elapsed_millis"], 1_000);
    assert_eq!(parsed["generation_elapsed_millis"], 2_000);
    assert_eq!(parsed["total_elapsed_millis"], 3_500);
    assert_eq!(parsed["time_to_first_output_millis"], 1_750);
    assert_eq!(parsed["generation_preparation_elapsed_millis"], 250);
    assert_eq!(
        parsed["generation_preparation_expert_source_read_byte_count"],
        0
    );
    assert_eq!(parsed["prefill_tok_per_second"], 5_000.0);
    assert_eq!(parsed["generation_tok_per_second"], 100.0);
    assert_eq!(parsed["mlx_peak_memory_bytes"], 40_000_000_000_i64);
    assert_eq!(parsed["mlx_active_memory_bytes"], 31_000_000_000_i64);
    assert_eq!(
        parsed["persistent_prompt_cache_diagnostics"]["matched_sequence_state_block_count"],
        4
    );
    assert_eq!(
        parsed["persistent_prompt_cache_diagnostics"]["published_block_count"],
        1
    );
    assert_eq!(
        parsed["persistent_prompt_cache_diagnostics"]["expert_bytes_reclaimed_for_restore"],
        16_384
    );
    assert_eq!(
        parsed["persistent_prompt_cache_diagnostics"]["startup_cleanup_evidence"]["obsolete_format"]
            ["artifact_count"],
        2
    );
    let serialized_line = lines[0];
    assert!(!serialized_line.contains("/fictional/"));
    assert!(!serialized_line.contains("model_directory"));
}

#[test]
fn should_append_multiple_records_to_performance_log() {
    let temp_dir = tempfile::tempdir().expect("test directory should be created");
    let mut performance_log =
        GenerationPerformanceLog::open(temp_dir.path()).expect("performance log should open");

    let first_record = GenerationPerformanceRecord {
        timestamp_millis: 1_700_000_000_000,
        request_id: 1,
        model_id: "model-a".to_owned(),
        prompt_token_count: 1_000,
        cached_token_count: 0,
        generated_token_count: 50,
        completion_reason: "end_of_sequence".to_owned(),
        prefill_elapsed_millis: 200,
        generation_elapsed_millis: 1_000,
        total_elapsed_millis: 1_500,
        time_to_first_output_millis: Some(400),
        generation_preparation_elapsed_millis: Some(100),
        first_decode_forward_elapsed_millis: Some(10),
        generation_preparation_expert_source_read_byte_count: 0,
        final_complete_expert_layer_count: Some(1),
        final_complete_expert_payload_bytes: Some(1_000),
        final_partial_expert_layer_count: Some(1),
        final_partial_expert_payload_bytes: Some(100),
        prefill_tok_per_second: Some(5_000.0),
        generation_tok_per_second: Some(50.0),
        mlx_peak_memory_bytes: None,
        mlx_active_memory_bytes: None,
        persistent_prompt_cache_diagnostics: None,
    };

    let second_record = GenerationPerformanceRecord {
        timestamp_millis: 1_700_000_005_000,
        request_id: 2,
        model_id: "model-a".to_owned(),
        prompt_token_count: 50_000,
        cached_token_count: 49_000,
        generated_token_count: 500,
        completion_reason: "tool_calls".to_owned(),
        prefill_elapsed_millis: 100,
        generation_elapsed_millis: 10_000,
        total_elapsed_millis: 11_000,
        time_to_first_output_millis: Some(700),
        generation_preparation_elapsed_millis: Some(200),
        first_decode_forward_elapsed_millis: Some(15),
        generation_preparation_expert_source_read_byte_count: 0,
        final_complete_expert_layer_count: Some(2),
        final_complete_expert_payload_bytes: Some(2_000),
        final_partial_expert_layer_count: Some(2),
        final_partial_expert_payload_bytes: Some(200),
        prefill_tok_per_second: Some(10_000.0),
        generation_tok_per_second: Some(50.0),
        mlx_peak_memory_bytes: Some(35_000_000_000),
        mlx_active_memory_bytes: Some(28_000_000_000),
        persistent_prompt_cache_diagnostics: None,
    };

    performance_log.record(&first_record);
    performance_log.record(&second_record);

    let performance_log_path = temp_dir.path().join("performance.jsonl");
    let contents =
        fs::read_to_string(&performance_log_path).expect("performance log file should be readable");
    let lines: Vec<&str> = contents.trim().lines().collect();
    assert_eq!(lines.len(), 2, "should have exactly two lines");

    let first_parsed: serde_json::Value =
        serde_json::from_str(lines[0]).expect("first line should be valid JSON");
    assert_eq!(first_parsed["request_id"], 1);

    let second_parsed: serde_json::Value =
        serde_json::from_str(lines[1]).expect("second line should be valid JSON");
    assert_eq!(second_parsed["request_id"], 2);
    assert!(second_parsed["mlx_peak_memory_bytes"].is_number());
    assert!(second_parsed["mlx_active_memory_bytes"].is_number());
}

#[test]
fn should_serialize_null_for_optional_fields() {
    let record = GenerationPerformanceRecord {
        timestamp_millis: 1_700_000_000_000,
        request_id: 1,
        model_id: "model-a".to_owned(),
        prompt_token_count: 1_000,
        cached_token_count: 1_000,
        generated_token_count: 50,
        completion_reason: "end_of_sequence".to_owned(),
        prefill_elapsed_millis: 0,
        generation_elapsed_millis: 1_000,
        total_elapsed_millis: 1_500,
        time_to_first_output_millis: None,
        generation_preparation_elapsed_millis: None,
        first_decode_forward_elapsed_millis: None,
        generation_preparation_expert_source_read_byte_count: 0,
        final_complete_expert_layer_count: None,
        final_complete_expert_payload_bytes: None,
        final_partial_expert_layer_count: None,
        final_partial_expert_payload_bytes: None,
        prefill_tok_per_second: None,
        generation_tok_per_second: Some(50.0),
        mlx_peak_memory_bytes: None,
        mlx_active_memory_bytes: None,
        persistent_prompt_cache_diagnostics: Some(WorkerPersistentPromptCacheRequestDiagnostics {
            lookup_outcome: WorkerPersistentPromptCacheLookupOutcome::Miss,
            block_token_count: 2_048,
            complete_prompt_block_count: 1,
            maximum_restorable_block_count: 1,
            matched_sequence_state_block_count: 0,
            restored_block_count: 0,
            first_missing_sequence_state_block_index: Some(0),
            miss_reason: Some(WorkerPersistentPromptCacheMissReason::RootSequenceStateBlockMissing),
            expected_block_hash_prefix: Some(
                WorkerPersistentPromptCacheExpectedBlockHashPrefix::from_block_hash([1_u8; 32]),
            ),
            startup_cleanup_evidence: None,
            published_block_count: 1,
            allocator_bytes_cleared_for_publication: 0,
            expert_bytes_reclaimed_for_publication: 0,
            expert_bytes_reclaimed_for_restore: 0,
        }),
    };

    let json = serde_json::to_string(&record).expect("should serialize");
    assert!(
        json.contains("\"prefill_tok_per_second\":null"),
        "null TPS should serialize as null"
    );
    assert!(
        json.contains("\"mlx_peak_memory_bytes\":null"),
        "null MLX peak should serialize as null"
    );
    assert!(
        json.contains("\"mlx_active_memory_bytes\":null"),
        "null MLX active should serialize as null"
    );
    assert!(
        json.contains("\"generation_tok_per_second\":50.0"),
        "present TPS should serialize as number"
    );
}

#[test]
fn should_create_performance_log_in_nonexistent_directory() {
    let temp_dir = tempfile::tempdir().expect("test directory should be created");
    let log_dir = temp_dir.path().join("nested").join("logs");
    // GenerationPerformanceLog::open creates the file but not the directory.
    // The directory must exist. This test verifies the file is created when
    // the directory exists.
    fs::create_dir_all(&log_dir).expect("nested log directory should be created");
    let performance_log = GenerationPerformanceLog::open(&log_dir);
    assert!(
        performance_log.is_ok(),
        "performance log should open in nested directory"
    );
}
