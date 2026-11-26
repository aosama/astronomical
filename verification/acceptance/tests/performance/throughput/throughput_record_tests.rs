//! The throughput record: the worker build it names and the fields it prints.

use super::*;

#[test]
fn should_record_actual_worker_digest_and_dirty_source_state() {
    let mut run_context = TestRunContext {
        source_commit: Some("fixture-commit".to_owned()),
        source_is_dirty: true,
        ..TestRunContext::default()
    };
    run_context
        .workspace_binary_sha256
        .insert("astronomical-inference-worker".to_owned(), "ab".repeat(32));
    let build_identity = worker_build_identity(&run_context).expect("worker identity");
    assert_eq!(build_identity.source_commit, "fixture-commit");
    assert!(build_identity.source_is_dirty);
    assert_eq!(build_identity.executable_sha256, "ab".repeat(32));
}

#[test]
fn should_reject_missing_source_commit_or_worker_identity_before_measurement() {
    let mut run_context = TestRunContext::default();
    assert!(worker_build_identity(&run_context).is_err());
    run_context.source_commit = Some("fixture-commit".to_owned());
    assert!(worker_build_identity(&run_context).is_err());
}

fn fixture_throughput_record(configured_maximum_mlx_memory_bytes: Option<u64>) -> ThroughputRecord {
    ThroughputRecord {
        timestamp: "test-timestamp".to_owned(),
        model_id: "test-model".to_owned(),
        journey: ThroughputJourneyKind::ResidentMemoryCeilingProbe,
        decode_astro_expert_cache_policy: WorkerDecodeAstroExpertCachePolicy::Baseline,
        prefill_tokens_per_second: 0,
        decode_tokens_per_second: 0,
        git_commit: None,
        worker_build_identity: WorkerBuildIdentity {
            source_commit: "fixture-commit".to_owned(),
            source_is_dirty: false,
            executable_sha256: "ab".repeat(32),
        },
        machine: MachineSpecs {
            os: "macos",
            arch: "aarch64",
            cpu_cores: 1,
            cpu_model: None,
            total_memory_bytes: None,
            machine_mlx_memory_ceiling_bytes: None,
        },
        total_input_tokens: 0,
        total_output_tokens: 0,
        cached_tokens: 0,
        prefill_time_seconds: 0.0,
        decode_time_seconds: 0.0,
        configured_maximum_mlx_memory_bytes,
        completion_ordinal_in_worker: 2,
        memory_cell: None,
    }
}

#[test]
fn should_record_the_assigned_memory_ceiling_so_cells_sharing_a_kind_stay_distinct() {
    // Every engine-versus-ceiling cell records under one journey kind; the
    // assigned ceiling is the only field that tells a 24 GB line from a 19 GB one.
    let twenty_four_gigabyte_ceiling_bytes = 24_u64 << 30;
    let nineteen_gigabyte_ceiling_bytes = 19_u64 << 30;

    let twenty_four_gigabyte_line = throughput_record_json(&fixture_throughput_record(Some(
        twenty_four_gigabyte_ceiling_bytes,
    )));
    let nineteen_gigabyte_line = throughput_record_json(&fixture_throughput_record(Some(
        nineteen_gigabyte_ceiling_bytes,
    )));

    assert_eq!(
        twenty_four_gigabyte_line["configured_maximum_mlx_memory_bytes"].as_u64(),
        Some(twenty_four_gigabyte_ceiling_bytes)
    );
    assert_eq!(
        nineteen_gigabyte_line["configured_maximum_mlx_memory_bytes"].as_u64(),
        Some(nineteen_gigabyte_ceiling_bytes)
    );
    assert_eq!(
        twenty_four_gigabyte_line["journey"],
        nineteen_gigabyte_line["journey"]
    );
}

#[test]
fn should_omit_the_assigned_memory_ceiling_when_the_journey_used_the_default_serving_ceiling() {
    let default_ceiling_line = throughput_record_json(&fixture_throughput_record(None));

    assert!(
        default_ceiling_line
            .get("configured_maximum_mlx_memory_bytes")
            .is_none(),
        "a journey under the default serving ceiling names no ceiling"
    );
}

#[test]
fn should_record_a_zero_byte_assigned_ceiling_rather_than_dropping_it() {
    let zero_ceiling_line = throughput_record_json(&fixture_throughput_record(Some(0)));

    assert_eq!(
        zero_ceiling_line["configured_maximum_mlx_memory_bytes"].as_u64(),
        Some(0)
    );
}
