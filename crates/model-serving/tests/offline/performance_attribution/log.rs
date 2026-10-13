use super::*;

#[test]
fn should_append_one_parseable_model_loading_json_record() {
    let temporary_log_directory = tempfile::tempdir().expect("temp log directory should exist");
    let performance_attribution_log_path = temporary_log_directory
        .path()
        .join("performance-attribution.jsonl");
    let mut performance_attribution = PerformanceAttribution::enabled();
    performance_attribution.record_completed_operation(
        PerformanceOperation::ArtifactValidation,
        std::time::Duration::from_nanos(5),
        std::time::Duration::from_nanos(25),
    );
    let performance_attribution_report = performance_attribution
        .finish_model_loading(model_loading_metadata(
            PerformanceAttributionOutcome::Success,
        ))
        .expect("enabled attribution should produce one model-loading report");
    let mut performance_attribution_log =
        PerformanceAttributionLog::open(&performance_attribution_log_path, true)
            .expect("enabled attribution log should open");

    performance_attribution_log
        .record(&performance_attribution_report)
        .expect("attribution report should be written");

    let performance_attribution_jsonl = std::fs::read_to_string(&performance_attribution_log_path)
        .expect("attribution log should be readable");
    let performance_attribution_json: Value =
        serde_json::from_str(performance_attribution_jsonl.trim())
            .expect("attribution report should be valid JSON");
    assert_eq!(performance_attribution_json["report_kind"], "model_loading");
    assert_eq!(performance_attribution_json["outcome"], "success");
    assert_eq!(
        performance_attribution_json["operations"][0]["operation"],
        "artifact_validation"
    );
    assert_eq!(
        performance_attribution_json["operations"][0]["total_elapsed_nanoseconds"],
        20
    );
}

#[test]
fn should_not_create_attribution_log_when_disabled() {
    let temporary_log_directory = tempfile::tempdir().expect("temp log directory should exist");
    let performance_attribution_log_path = temporary_log_directory
        .path()
        .join("performance-attribution.jsonl");

    let _performance_attribution_log =
        PerformanceAttributionLog::open(&performance_attribution_log_path, false)
            .expect("disabled attribution log should not need a file");

    assert!(!performance_attribution_log_path.exists());
}

#[test]
fn should_append_multiple_reports_to_one_jsonl_file() {
    let temporary_log_directory = tempfile::tempdir().expect("temp log directory should exist");
    let performance_attribution_log_path = temporary_log_directory
        .path()
        .join("performance-attribution.jsonl");
    let mut performance_attribution_log =
        PerformanceAttributionLog::open(&performance_attribution_log_path, true)
            .expect("enabled attribution log should open");

    for model_loading_outcome in [
        PerformanceAttributionOutcome::Success,
        PerformanceAttributionOutcome::Failed,
    ] {
        let performance_attribution_report = PerformanceAttribution::enabled()
            .finish_model_loading(model_loading_metadata(model_loading_outcome))
            .expect("enabled attribution should produce a model-loading report");
        performance_attribution_log
            .record(&performance_attribution_report)
            .expect("each attribution report should append successfully");
    }

    let performance_attribution_report_lines =
        std::fs::read_to_string(&performance_attribution_log_path)
            .expect("attribution log should be readable")
            .lines()
            .map(serde_json::from_str::<Value>)
            .collect::<Result<Vec<_>, _>>()
            .expect("every JSON Lines record should parse");

    assert_eq!(performance_attribution_report_lines.len(), 2);
    assert_eq!(
        performance_attribution_report_lines[0]["outcome"],
        "success"
    );
    assert_eq!(performance_attribution_report_lines[1]["outcome"], "failed");
}

#[test]
fn should_reopen_attribution_log_after_rotation_replaces_the_file() {
    let temporary_log_directory = tempfile::tempdir().expect("temp log directory should exist");
    let performance_attribution_log_path = temporary_log_directory
        .path()
        .join("performance-attribution.jsonl");
    let rotated_performance_attribution_log_path = temporary_log_directory
        .path()
        .join("performance-attribution.previous.jsonl");
    let mut performance_attribution_log =
        PerformanceAttributionLog::open(&performance_attribution_log_path, true)
            .expect("enabled attribution log should open");
    let first_performance_attribution_report = PerformanceAttribution::enabled()
        .finish_model_loading(model_loading_metadata(
            PerformanceAttributionOutcome::Success,
        ))
        .expect("enabled attribution should produce the first report");
    performance_attribution_log
        .record(&first_performance_attribution_report)
        .expect("first attribution report should be written");
    std::fs::rename(
        &performance_attribution_log_path,
        &rotated_performance_attribution_log_path,
    )
    .expect("attribution log should rotate");
    std::fs::File::create(&performance_attribution_log_path)
        .expect("replacement attribution log should exist");

    let second_performance_attribution_report = PerformanceAttribution::enabled()
        .finish_model_loading(model_loading_metadata(
            PerformanceAttributionOutcome::Failed,
        ))
        .expect("enabled attribution should produce the second report");
    performance_attribution_log
        .record(&second_performance_attribution_report)
        .expect("second attribution report should be written");

    let active_performance_attribution_log =
        std::fs::read_to_string(&performance_attribution_log_path)
            .expect("active attribution log should be readable");
    let active_performance_attribution_json: Value =
        serde_json::from_str(active_performance_attribution_log.trim())
            .expect("active attribution report should be valid JSON");
    assert_eq!(active_performance_attribution_json["outcome"], "failed");
    assert_eq!(
        std::fs::read_to_string(&rotated_performance_attribution_log_path)
            .expect("rotated attribution log should be readable")
            .lines()
            .count(),
        1
    );
}
