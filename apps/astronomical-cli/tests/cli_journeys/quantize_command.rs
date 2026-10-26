//! The `astronomical quantize` filesystem journeys: the `plan` and
//! `inspect` sub-verbs against documents written to a temporary directory.
//! No daemon, no GPU, no network. The pure parse assertions for the
//! `quantize` verb grammar and the default-build `convert` refusal are
//! unit tests under `src/arguments_tests/quantize_tests.rs` and
//! `src/quantize_command_tests.rs`.

use std::time::Duration;

use astronomical_cli::{QuantizeDependencies, run_quantize_inspect, run_quantize_plan};

use super::test_support::fresh_test_directory;

#[test]
#[ignore = "pending implementation of run_quantize_plan"]
fn should_allocate_from_a_sensitivity_document_and_print_the_bit_map() {
    let test_directory = fresh_test_directory("quantize", "plan");
    let sensitivity_path = test_directory.join("sensitivity.json");
    std::fs::write(
        &sensitivity_path,
        r#"{"base_model": "example/base", "candidate_bits": [4, 8], "group_size": 64,
            "metric": "kl_divergence_vs_reference",
            "layers": [
              {"layer_name": "model.layers.0.mlp.up_proj", "sensitivities": {"4": 0.9, "8": 0.1}, "param_count": 1000},
              {"layer_name": "model.layers.1.mlp.up_proj", "sensitivities": {"4": 0.2, "8": 0.1}, "param_count": 1000}
            ]}"#,
    )
    .expect("the fixture document should be writable");
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    let mut dependencies = QuantizeDependencies {
        stdout: &mut stdout,
        stderr: &mut stderr,
        progress_interval: Duration::from_secs(5),
    };
    let plan_arguments = astronomical_cli::QuantizePlanArguments {
        sensitivity_document_path: sensitivity_path,
        target_bits_per_weight: 6.0,
        candidate_bits: None,
        floor_components_per_block: 0,
        maximum_low_bit_block_run: 0,
        render_json: false,
    };

    run_quantize_plan(&plan_arguments, &mut dependencies).expect("the plan should succeed");

    let report = String::from_utf8(stdout).expect("the report is UTF-8");
    assert!(report.contains("model.layers.0.mlp.up_proj"));
    assert!(report.contains("Achieved bits per weight: 6.00"));
    let _ = std::fs::remove_dir_all(&test_directory);
}

#[test]
#[ignore = "pending implementation of run_quantize_inspect"]
fn should_report_a_directory_without_metadata_as_not_an_optiq_artifact() {
    let test_directory = fresh_test_directory("quantize", "inspect-empty");
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    let mut dependencies = QuantizeDependencies {
        stdout: &mut stdout,
        stderr: &mut stderr,
        progress_interval: Duration::from_secs(5),
    };
    let inspect_arguments = astronomical_cli::QuantizeInspectArguments {
        artifact_directory: test_directory.clone(),
        render_json: false,
    };

    run_quantize_inspect(&inspect_arguments, &mut dependencies)
        .expect("inspecting a plain directory succeeds with a disclosure");

    let report = String::from_utf8(stdout).expect("the report is UTF-8");
    assert!(report.contains("not an OptiQ-format artifact"));
    let _ = std::fs::remove_dir_all(&test_directory);
}
