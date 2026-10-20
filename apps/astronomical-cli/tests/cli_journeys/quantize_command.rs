//! The `astronomical quantize` journeys: argument parsing, the pure `plan`
//! and `inspect` sub-verbs against documents written to a temporary
//! directory, and the `convert` refusal in a default (MLX-free) build. No
//! daemon, no GPU, no network.

use std::ffi::OsString;
use std::time::Duration;

use astronomical_cli::{CliCommand, QuantizeCommand, QuantizeDependencies, parse_command};

use super::test_support::fresh_test_directory;

fn arguments(words: &[&str]) -> Vec<OsString> {
    std::iter::once("astronomical")
        .chain(words.iter().copied())
        .map(OsString::from)
        .collect()
}

#[test]
#[ignore = "pending implementation of parse_quantize_command"]
fn should_parse_a_plan_invocation_with_a_candidate_ladder() {
    let parsed = parse_command(arguments(&[
        "quantize",
        "plan",
        "--sensitivity",
        "sweep.json",
        "--target-bits-per-weight",
        "4.5",
        "--candidate-bits",
        "2,4,8",
        "--json",
    ]))
    .expect("the plan invocation should parse");

    match parsed {
        CliCommand::Quantize(QuantizeCommand::Plan(plan_arguments)) => {
            assert_eq!(plan_arguments.target_bits_per_weight, 4.5);
            assert_eq!(plan_arguments.candidate_bits, Some(vec![2, 4, 8]));
            assert!(plan_arguments.render_json);
        }
        other => panic!("expected a plan command, got {other:?}"),
    }
}

#[test]
#[ignore = "pending implementation of parse_quantize_command"]
fn should_reject_an_unknown_quantize_target() {
    let parse_outcome = parse_command(arguments(&["quantize", "shrink"]));

    assert!(matches!(
        parse_outcome,
        Err(astronomical_cli::UsageError::UnknownQuantizeTarget(target)) if target == "shrink"
    ));
}

#[test]
#[ignore = "pending implementation of parse_quantize_command"]
fn should_reject_an_unsupported_candidate_bit_width() {
    let parse_outcome = parse_command(arguments(&[
        "quantize",
        "plan",
        "--sensitivity",
        "sweep.json",
        "--target-bits-per-weight",
        "4.5",
        "--candidate-bits",
        "4,7",
    ]));

    assert!(matches!(
        parse_outcome,
        Err(astronomical_cli::UsageError::InvalidQuantizeValue {
            flag: "--candidate-bits",
            ..
        })
    ));
}

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

    astronomical_cli::run_quantize_plan(&plan_arguments, &mut dependencies)
        .expect("the plan should succeed");

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

    astronomical_cli::run_quantize_inspect(&inspect_arguments, &mut dependencies)
        .expect("inspecting a plain directory succeeds with a disclosure");

    let report = String::from_utf8(stdout).expect("the report is UTF-8");
    assert!(report.contains("not an OptiQ-format artifact"));
    let _ = std::fs::remove_dir_all(&test_directory);
}

#[cfg(not(feature = "quantize-convert"))]
#[test]
fn should_refuse_convert_in_a_default_build_and_name_the_feature() {
    let test_directory = fresh_test_directory("quantize", "convert-refused");
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    let mut dependencies = QuantizeDependencies {
        stdout: &mut stdout,
        stderr: &mut stderr,
        progress_interval: Duration::from_secs(5),
    };
    let convert_arguments = astronomical_cli::QuantizeConvertArguments {
        source_directory: test_directory.join("source"),
        output_directory: test_directory.join("output"),
        base_model_name: None,
        method: astronomical_cli::quantize_arguments::QuantizeMethodArgument::Static,
        reference: astronomical_cli::quantize_arguments::QuantizeReferenceArgument::Auto,
        target_bits_per_weight: 5.0,
        candidate_bits: vec![4, 8],
        group_size: 64,
        calibration_sequence_count: 24,
        calibration_mix_path: None,
        floor_components_per_block: 2,
        range_search_enabled: true,
        range_search_maximum_bits: 3,
        overwrite_output: false,
    };

    let convert_outcome =
        astronomical_cli::run_quantize_convert(&convert_arguments, &mut dependencies);

    let error_text = convert_outcome
        .expect_err("a default build refuses convert")
        .to_string();
    assert!(error_text.contains("quantize-convert"));
    let _ = std::fs::remove_dir_all(&test_directory);
}
