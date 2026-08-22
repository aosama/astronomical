//! Strict bundled catalog and supervisor attribution contracts.

use std::{
    fs, io,
    io::Write,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
};

use astronomical_supervisor::{
    DownloadCatalog, DownloadCatalogFamily, SupervisorPerformanceAttributionLog,
    SupervisorPerformanceMeasurement, SupervisorPerformanceOperation,
};

const VALID_REVISION: &str = "0123456789abcdef0123456789abcdef01234567";

#[test]
fn should_load_the_bundled_production_catalog() {
    let download_catalog = DownloadCatalog::load_bundled()
        .expect("the bundled production catalog should remain valid");

    assert_eq!(download_catalog.schema_version(), 1);
}

#[test]
fn should_package_flux_with_only_its_executable_diffusers_graph() {
    let download_catalog = DownloadCatalog::load_bundled()
        .expect("the bundled production catalog should remain valid");
    let flux_entry = download_catalog
        .entries()
        .iter()
        .find(|entry| entry.family() == DownloadCatalogFamily::Flux2Klein)
        .expect("the release catalog should include the executable FLUX profile");

    assert!(flux_entry.capabilities().supports_image_generation);
    for required_path in [
        "LICENSE.md",
        "model_index.json",
        "scheduler/scheduler_config.json",
        "text_encoder/config.json",
        "tokenizer/tokenizer.json",
        "transformer/diffusion_pytorch_model.safetensors",
        "vae/diffusion_pytorch_model.safetensors",
    ] {
        assert!(flux_entry.download_path_selection().includes(required_path));
    }
    assert!(
        !flux_entry
            .download_path_selection()
            .includes("flux-2-klein-4b.safetensors")
    );
    assert!(!flux_entry.download_path_selection().includes("editing.jpg"));
}

#[test]
fn should_accept_complete_entries_for_all_executable_families_in_authored_order() {
    let catalog_json = serde_json::json!({
        "schema_version": 1,
        "entries": [
            valid_entry("astronomical-test/example-qwen", "qwen3_5"),
            valid_entry("astronomical-test/example-laguna", "laguna"),
            valid_entry("astronomical-test/example-flux", "flux2_klein"),
        ]
    });

    let download_catalog = DownloadCatalog::parse_json(&catalog_json.to_string())
        .expect("complete fictional entries should be accepted");

    assert_eq!(download_catalog.entries().len(), 3);
    assert_eq!(
        download_catalog.entries()[0].huggingface_id(),
        "astronomical-test/example-qwen"
    );
    assert_eq!(
        download_catalog.entries()[1].huggingface_id(),
        "astronomical-test/example-laguna"
    );
    assert_eq!(
        download_catalog.entries()[2].huggingface_id(),
        "astronomical-test/example-flux"
    );
}

#[test]
fn should_accept_bounded_executable_payload_selection_and_reject_unsafe_rules() {
    let selected_entry = entry_with_overrides(serde_json::json!({
        "included_paths": ["LICENSE.md", "model_index.json", "transformer/"]
    }));
    let download_catalog =
        DownloadCatalog::parse_json(&catalog_with_entry(selected_entry).to_string())
            .expect("release-authored files and directory prefixes should be selectable");
    let selection = download_catalog.entries()[0].download_path_selection();
    assert!(selection.includes("LICENSE.md"));
    assert!(selection.includes("transformer/model.safetensors"));
    assert!(!selection.includes("alternate-model.safetensors"));

    for invalid_paths in [
        serde_json::json!([]),
        serde_json::json!(["../weights"]),
        serde_json::json!(["weights\\model.safetensors"]),
        serde_json::json!(["weights/", "weights/model.safetensors"]),
        serde_json::json!(["Transformer/", "transformer/"]),
    ] {
        let invalid_entry = entry_with_overrides(serde_json::json!({
            "included_paths": invalid_paths
        }));
        DownloadCatalog::parse_json(&catalog_with_entry(invalid_entry).to_string())
            .expect_err("unsafe, empty, colliding, or redundant selections must fail closed");
    }
}

#[test]
fn should_reject_invalid_catalog_document_shapes() {
    for invalid_catalog_json in [
        r#"{"entries":[]}"#,
        r#"{"schema_version":2,"entries":[]}"#,
        r#"{"schema_version":1}"#,
        r#"{"schema_version":1,"entries":{}}"#,
        r#"{"schema_version":1,"entries":[],"unexpected":true}"#,
        r#"{"schema_version":1,"schema_version":1,"entries":[]}"#,
        r#"{"schema_version":1,"entries":[]} trailing"#,
    ] {
        DownloadCatalog::parse_json(invalid_catalog_json)
            .expect_err("an invalid catalog document shape must fail closed");
    }
}

#[test]
fn should_reject_catalog_metadata_that_exceeds_startup_resource_bounds() {
    let oversized_catalog_document = " ".repeat(1_000_001);
    DownloadCatalog::parse_json(&oversized_catalog_document)
        .expect_err("catalog metadata beyond the startup byte bound must fail closed");

    let oversized_entry_count_catalog = serde_json::json!({
        "schema_version": 1,
        "entries": (0..1_025)
            .map(|entry_index| valid_entry(
                &format!("astronomical-test/example-{entry_index}"),
                "qwen3_5",
            ))
            .collect::<Vec<_>>()
    });
    DownloadCatalog::parse_json(&oversized_entry_count_catalog.to_string())
        .expect_err("catalog entry counts beyond the startup bound must fail closed");

    let oversized_display_name = "m".repeat(257);
    let oversized_display_name_catalog = catalog_with_entry(entry_with_overrides(
        serde_json::json!({"display_name": oversized_display_name}),
    ));
    DownloadCatalog::parse_json(&oversized_display_name_catalog.to_string())
        .expect_err("display names beyond the metadata bound must fail closed");
}

#[test]
fn should_reject_unsafe_or_non_hub_repository_identities() {
    for invalid_huggingface_id in [
        "",
        "single-component",
        "/model",
        "organization/",
        "organization/model/extra",
        " organization/model",
        "organization/model ",
        "organization/../model",
        "organization/model\\child",
        "organization/.model",
        "organization/model-",
        "organization/model--variant",
        "organization/model..variant",
        "organization/model.git",
        "organization/model name",
        "organization/modél",
    ] {
        let catalog_json = catalog_with_entry(valid_entry(invalid_huggingface_id, "qwen3_5"));
        DownloadCatalog::parse_json(&catalog_json.to_string())
            .expect_err("an unsafe or unsupported repository identity must fail closed");
    }

    let oversized_model_name = "m".repeat(97);
    let catalog_json = catalog_with_entry(valid_entry(
        &format!("astronomical-test/{oversized_model_name}"),
        "qwen3_5",
    ));
    DownloadCatalog::parse_json(&catalog_json.to_string())
        .expect_err("a repository name beyond the Hub bound must fail closed");
}

#[test]
fn should_reject_invalid_revision_family_visibility_size_and_display_name() {
    let invalid_entries = [
        entry_with_overrides(serde_json::json!({"revision": "main"})),
        entry_with_overrides(serde_json::json!({
            "revision": "0123456789abcdef0123456789abcdef0123456g"
        })),
        entry_with_overrides(serde_json::json!({
            "revision": "0123456789abcdef0123456789abcdef0123456A"
        })),
        entry_with_overrides(serde_json::json!({"family": "unsupported_family"})),
        entry_with_overrides(serde_json::json!({"public": false})),
        entry_with_overrides(serde_json::json!({"approximate_size_bytes": 0})),
        entry_with_overrides(
            serde_json::json!({"approximate_size_bytes": 9_007_199_254_740_992_u64}),
        ),
        entry_with_overrides(serde_json::json!({"display_name": "  "})),
        entry_with_overrides(serde_json::json!({"display_name": "Bad\nName"})),
    ];
    for invalid_entry in invalid_entries {
        let catalog_json = catalog_with_entry(invalid_entry);
        DownloadCatalog::parse_json(&catalog_json.to_string())
            .expect_err("invalid entry metadata must fail closed");
    }

    DownloadCatalog::parse_json(&format!(
        r#"{{"schema_version":1,"entries":[{{"huggingface_id":"astronomical-test/example-qwen","revision":"{VALID_REVISION}","display_name":"Example","family":"qwen3_5","approximate_size_bytes":4000000000}}]}}"#
    ))
    .expect_err("public must remain a required field");
}

#[test]
fn should_accept_the_largest_approximate_size_that_the_library_can_render_exactly() {
    let catalog_json = catalog_with_entry(entry_with_overrides(
        serde_json::json!({"approximate_size_bytes": 9_007_199_254_740_991_u64}),
    ));

    let download_catalog = DownloadCatalog::parse_json(&catalog_json.to_string())
        .expect("the maximum exactly renderable byte count should remain valid");

    assert_eq!(
        download_catalog.entries()[0].approximate_size_bytes(),
        9_007_199_254_740_991
    );
}

#[test]
fn should_reject_unknown_duplicate_and_case_colliding_entries() {
    let extra_field_entry = entry_with_overrides(serde_json::json!({"unexpected": true}));
    DownloadCatalog::parse_json(&catalog_with_entry(extra_field_entry).to_string())
        .expect_err("unknown entry fields must fail closed");

    let duplicated_entry = valid_entry("astronomical-test/example-qwen", "qwen3_5");
    let duplicate_catalog = serde_json::json!({
        "schema_version": 1,
        "entries": [duplicated_entry.clone(), duplicated_entry]
    });
    DownloadCatalog::parse_json(&duplicate_catalog.to_string())
        .expect_err("exact duplicate identities must fail closed");

    let case_collision_catalog = serde_json::json!({
        "schema_version": 1,
        "entries": [
            valid_entry("Astronomical-Test/Example-Qwen", "qwen3_5"),
            valid_entry("astronomical-test/example-qwen", "qwen3_5"),
        ]
    });
    DownloadCatalog::parse_json(&case_collision_catalog.to_string())
        .expect_err("case-insensitive destination collisions must fail closed");
}

#[test]
fn should_leave_the_supervisor_attribution_file_absent_when_disabled() {
    let log_directory = tempfile::tempdir().expect("a log directory should be created");
    let attribution_log = SupervisorPerformanceAttributionLog::open(log_directory.path(), false)
        .expect("disabled attribution should remain inert");

    let measured_catalog = attribution_log
        .measure_operation(
            SupervisorPerformanceOperation::LibraryCatalogLoad,
            || DownloadCatalog::parse_json(r#"{"schema_version":1,"entries":[]}"#),
            |catalog_outcome| match catalog_outcome {
                Ok(download_catalog) => SupervisorPerformanceMeasurement::successful_catalog_load(
                    download_catalog.entry_count(),
                ),
                Err(_) => SupervisorPerformanceMeasurement::failure(),
            },
        )
        .expect("disabled attribution should not affect the operation")
        .expect("the measured catalog should parse");

    assert_eq!(measured_catalog.entry_count(), 0);
    assert!(
        !log_directory
            .path()
            .join("supervisor-performance-attribution.jsonl")
            .exists()
    );
}

#[test]
fn should_record_success_and_failure_catalog_load_boundaries_when_enabled() {
    let log_directory = tempfile::tempdir().expect("a log directory should be created");
    let attribution_log = SupervisorPerformanceAttributionLog::open(log_directory.path(), true)
        .expect("enabled attribution should open its writer");

    let successful_catalog = attribution_log
        .measure_operation(
            SupervisorPerformanceOperation::LibraryCatalogLoad,
            || DownloadCatalog::parse_json(r#"{"schema_version":1,"entries":[]}"#),
            |catalog_outcome| match catalog_outcome {
                Ok(download_catalog) => SupervisorPerformanceMeasurement::successful_catalog_load(
                    download_catalog.entry_count(),
                ),
                Err(_) => SupervisorPerformanceMeasurement::failure(),
            },
        )
        .expect("successful attribution should be written");
    assert!(successful_catalog.is_ok());

    let failed_catalog = attribution_log
        .measure_operation(
            SupervisorPerformanceOperation::LibraryCatalogLoad,
            || DownloadCatalog::parse_json(r#"{"schema_version":2,"entries":[]}"#),
            |_| SupervisorPerformanceMeasurement::failure(),
        )
        .expect("failed operation attribution should still be written");
    assert!(failed_catalog.is_err());

    let attribution_text = fs::read_to_string(
        log_directory
            .path()
            .join("supervisor-performance-attribution.jsonl"),
    )
    .expect("the attribution log should remain readable");
    let attribution_records = attribution_text
        .lines()
        .map(|json_line| {
            serde_json::from_str::<serde_json::Value>(json_line)
                .expect("each attribution row should contain JSON")
        })
        .collect::<Vec<_>>();

    assert_eq!(attribution_records.len(), 2);
    assert_eq!(attribution_records[0]["operation"], "library_catalog_load");
    assert_eq!(attribution_records[0]["outcome"], "success");
    assert_eq!(attribution_records[0]["catalog_entry_count"], 0);
    assert_eq!(attribution_records[1]["outcome"], "failure");
    for attribution_record in attribution_records {
        let started_at = attribution_record["started_at_unix_millis"]
            .as_u64()
            .expect("the start timestamp should be unsigned");
        let ended_at = attribution_record["ended_at_unix_millis"]
            .as_u64()
            .expect("the end timestamp should be unsigned");
        assert!(ended_at >= started_at);
        assert!(attribution_record["elapsed_nanoseconds"].is_u64());
    }
}

#[test]
fn should_return_a_typed_io_error_when_an_enabled_attribution_write_fails() {
    let attribution_log = SupervisorPerformanceAttributionLog::from_writer(AlwaysFailingWriter);

    let attribution_error = attribution_log
        .measure_operation(
            SupervisorPerformanceOperation::LibraryCatalogLoad,
            || DownloadCatalog::parse_json(r#"{"schema_version":1,"entries":[]}"#),
            |_| SupervisorPerformanceMeasurement::successful_catalog_load(0),
        )
        .expect_err("required attribution write failures must remain typed");

    assert_eq!(attribution_error.kind(), io::ErrorKind::Other);
    assert!(
        attribution_error
            .to_string()
            .contains("intentional attribution failure")
    );
}

#[test]
fn should_return_a_typed_io_error_when_an_enabled_attribution_timestamp_fails() {
    let clock_call_count = Arc::new(AtomicUsize::new(0));
    let measured_operation_executed = Arc::new(AtomicBool::new(false));
    let clock_call_count_for_writer = Arc::clone(&clock_call_count);
    let measured_operation_executed_for_operation = Arc::clone(&measured_operation_executed);
    let attribution_log =
        SupervisorPerformanceAttributionLog::from_writer_and_clock(io::sink(), move || {
            if clock_call_count_for_writer.fetch_add(1, Ordering::SeqCst) == 0 {
                Ok(1_000)
            } else {
                Err(io::Error::other("intentional attribution clock failure"))
            }
        });

    let attribution_error = attribution_log
        .measure_operation(
            SupervisorPerformanceOperation::LibraryCatalogLoad,
            || {
                measured_operation_executed_for_operation.store(true, Ordering::SeqCst);
                DownloadCatalog::parse_json(r#"{"schema_version":1,"entries":[]}"#)
            },
            |_| SupervisorPerformanceMeasurement::successful_catalog_load(0),
        )
        .expect_err("required attribution timestamp failures must remain typed");

    assert!(measured_operation_executed.load(Ordering::SeqCst));
    assert_eq!(clock_call_count.load(Ordering::SeqCst), 2);
    assert_eq!(attribution_error.kind(), io::ErrorKind::Other);
    assert!(
        attribution_error
            .to_string()
            .contains("intentional attribution clock failure")
    );
}

struct AlwaysFailingWriter;

impl Write for AlwaysFailingWriter {
    fn write(&mut self, _buffer: &[u8]) -> io::Result<usize> {
        Err(io::Error::other("intentional attribution failure"))
    }

    fn flush(&mut self) -> io::Result<()> {
        Err(io::Error::other("intentional attribution failure"))
    }
}

fn valid_entry(huggingface_id: &str, family: &str) -> serde_json::Value {
    serde_json::json!({
        "huggingface_id": huggingface_id,
        "revision": VALID_REVISION,
        "display_name": "Example model",
        "family": family,
        "approximate_size_bytes": 4_000_000_000_u64,
        "public": true,
    })
}

fn entry_with_overrides(overrides: serde_json::Value) -> serde_json::Value {
    let mut catalog_entry = valid_entry("astronomical-test/example-qwen", "qwen3_5");
    for (field_name, field_value) in overrides
        .as_object()
        .expect("entry overrides should be an object")
    {
        catalog_entry
            .as_object_mut()
            .expect("the valid entry should be an object")
            .insert(field_name.clone(), field_value.clone());
    }
    catalog_entry
}

fn catalog_with_entry(catalog_entry: serde_json::Value) -> serde_json::Value {
    serde_json::json!({"schema_version": 1, "entries": [catalog_entry]})
}
