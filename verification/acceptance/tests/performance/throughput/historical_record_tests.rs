//! Regression coverage for the durable throughput-history ledger resolution.

use std::path::Path;

use super::*;

#[test]
fn should_anchor_the_default_history_ledger_to_the_workspace_root_regardless_of_launch_directory() {
    // The resolution used to be relative to the test process's working
    // directory, and the standard acceptance launcher starts binaries with the
    // package directory as their working directory, so measurements forked
    // into a doubled nested copy of the ledger path instead of the
    // source-controlled file. The compile-time manifest anchor makes the
    // absolute path itself the regression signal: a launch-directory-relative
    // resolution can never produce one.
    let resolved_ledger = resolve_history_log_path(None);

    assert!(
        resolved_ledger.is_absolute(),
        "the default ledger path must be anchored, not launch-directory relative: {resolved_ledger:?}"
    );

    let workspace_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..");
    let expected_ledger = workspace_root.join(DEFAULT_HISTORY_RELATIVE_PATH);
    assert_eq!(
        resolved_ledger, expected_ledger,
        "the default ledger must resolve to the source-controlled history file"
    );

    let canonical_ledger = resolved_ledger
        .canonicalize()
        .expect("the source-controlled history ledger should exist on disk");
    let canonical_ledger_text = canonical_ledger.to_string_lossy();
    assert!(
        !canonical_ledger_text.contains("verification/acceptance/verification/acceptance"),
        "the resolved ledger must not live in a doubled nested copy of the package tree: {canonical_ledger_text}"
    );
    assert!(
        canonical_ledger.is_file(),
        "the canonical ledger must be the existing source-controlled file: {canonical_ledger_text}"
    );
}

#[test]
fn should_prefer_the_explicit_history_path_over_the_workspace_default() {
    let override_ledger =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("target/selected-throughput-history.jsonl");
    let resolved_ledger = resolve_history_log_path(Some(&override_ledger));
    assert_eq!(
        resolved_ledger, override_ledger,
        "an explicit ledger override must take precedence over the anchored default"
    );
}

#[test]
fn should_preserve_spaces_in_the_explicit_history_path() {
    let configured_history_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target/measurement evidence/selected-throughput-history.jsonl");

    assert_eq!(
        resolve_history_log_path(Some(&configured_history_path)),
        configured_history_path
    );
}

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
        decode_expert_cache_policy: WorkerDecodeExpertCachePolicy::Baseline,
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
        "a default-ceiling line must stay byte-for-byte compatible with earlier history rows"
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
