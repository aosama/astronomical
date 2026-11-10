//! Parsing of the `quantize` verb: the `plan` sub-grammar, the candidate
//! bit-width validation, and the unknown-target rejection. These are pure
//! in-memory parse assertions; the filesystem-driven `plan` and `inspect`
//! journeys live in `tests/integration/quantize.rs`.

use crate::arguments::CliCommand;
use crate::arguments::UsageError;
use crate::verbs::quantize::QuantizeCommand;

use super::parse;

#[test]

fn should_parse_a_plan_invocation_with_a_candidate_ladder() {
    let parsed = parse(&[
        "quantize",
        "plan",
        "--sensitivity",
        "sweep.json",
        "--target-bits-per-weight",
        "4.5",
        "--candidate-bits",
        "2,4,8",
        "--json",
    ])
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

fn should_reject_an_unknown_quantize_target() {
    let parse_outcome = parse(&["quantize", "shrink"]);

    assert!(matches!(
        parse_outcome,
        Err(UsageError::UnknownQuantizeTarget(target)) if target == "shrink"
    ));
}

#[test]

fn should_reject_an_unsupported_candidate_bit_width() {
    let parse_outcome = parse(&[
        "quantize",
        "plan",
        "--sensitivity",
        "sweep.json",
        "--target-bits-per-weight",
        "4.5",
        "--candidate-bits",
        "4,7",
    ]);

    assert!(matches!(
        parse_outcome,
        Err(UsageError::InvalidQuantizeValue {
            flag: "--candidate-bits",
            ..
        })
    ));
}

#[test]
fn should_parse_a_convert_invocation_with_optiq_defaults() {
    let parsed = parse(&[
        "quantize", "convert", "source", "--output", "out", "--method", "static",
    ])
    .expect("the convert invocation should parse");

    match parsed {
        CliCommand::Quantize(QuantizeCommand::Convert(convert_arguments)) => {
            assert_eq!(
                convert_arguments.source_directory,
                std::path::PathBuf::from("source")
            );
            assert_eq!(
                convert_arguments.output_directory,
                std::path::PathBuf::from("out")
            );
            assert_eq!(
                convert_arguments.method,
                crate::verbs::quantize::QuantizeMethodArgument::Static
            );
            assert_eq!(convert_arguments.candidate_bits, vec![4, 8]);
            assert_eq!(convert_arguments.group_size, 64);
            assert_eq!(convert_arguments.calibration_sequence_count, 24);
            assert!(convert_arguments.range_search_enabled);
            assert!(!convert_arguments.overwrite_output);
        }
        other => panic!("expected a convert command, got {other:?}"),
    }
}

#[test]
fn should_require_an_output_directory_for_convert() {
    let parse_outcome = parse(&["quantize", "convert", "source"]);

    assert!(matches!(
        parse_outcome,
        Err(UsageError::MissingValue("--output"))
    ));
}

#[test]
fn should_reject_a_repeated_convert_flag() {
    let parse_outcome = parse(&[
        "quantize", "convert", "source", "--output", "a", "--output", "b",
    ]);

    assert!(matches!(
        parse_outcome,
        Err(UsageError::RepeatedArgument("--output"))
    ));
}

#[test]
fn should_reject_a_non_positive_target() {
    let parse_outcome = parse(&[
        "quantize",
        "plan",
        "--sensitivity",
        "sweep.json",
        "--target-bits-per-weight",
        "0",
    ]);

    assert!(matches!(
        parse_outcome,
        Err(UsageError::InvalidQuantizeValue {
            flag: "--target-bits-per-weight",
            ..
        })
    ));
}

#[test]
fn should_reject_a_missing_quantize_sub_verb() {
    assert!(matches!(
        parse(&["quantize"]),
        Err(UsageError::MissingCommand)
    ));
}

#[test]
fn should_parse_an_evaluate_invocation_with_optiq_defaults() {
    let parsed = parse(&["quantize", "evaluate", "artifact", "--reference", "base"])
        .expect("the evaluate invocation should parse");

    match parsed {
        CliCommand::Quantize(QuantizeCommand::Evaluate(evaluate_arguments)) => {
            assert_eq!(
                evaluate_arguments.artifact_directory,
                std::path::PathBuf::from("artifact")
            );
            assert_eq!(
                evaluate_arguments.reference_directory,
                std::path::PathBuf::from("base")
            );
            assert_eq!(evaluate_arguments.prompt_count, 64);
            assert_eq!(evaluate_arguments.sequence_length_tokens, 256);
            assert!(!evaluate_arguments.render_json);
        }
        other => panic!("expected an evaluate command, got {other:?}"),
    }
}

#[test]
fn should_parse_explicit_evaluate_sizes_and_the_json_switch() {
    let parsed = parse(&[
        "quantize",
        "evaluate",
        "artifact",
        "--reference",
        "base",
        "--prompts",
        "8",
        "--sequence-length",
        "128",
        "--json",
    ])
    .expect("the evaluate invocation should parse");

    match parsed {
        CliCommand::Quantize(QuantizeCommand::Evaluate(evaluate_arguments)) => {
            assert_eq!(evaluate_arguments.prompt_count, 8);
            assert_eq!(evaluate_arguments.sequence_length_tokens, 128);
            assert!(evaluate_arguments.render_json);
        }
        other => panic!("expected an evaluate command, got {other:?}"),
    }
}

#[test]
fn should_reject_an_evaluate_invocation_without_a_reference() {
    let parse_outcome = parse(&["quantize", "evaluate", "artifact"]);

    assert!(matches!(
        parse_outcome,
        Err(UsageError::MissingValue("--reference"))
    ));
}

#[test]
fn should_reject_an_evaluation_of_zero_prompts() {
    let parse_outcome = parse(&[
        "quantize",
        "evaluate",
        "artifact",
        "--reference",
        "base",
        "--prompts",
        "0",
    ]);

    assert!(matches!(
        parse_outcome,
        Err(UsageError::InvalidQuantizeValue {
            flag: "--prompts",
            ..
        })
    ));
}

#[test]
fn should_reject_an_evaluate_invocation_naming_the_reference_twice() {
    let parse_outcome = parse(&[
        "quantize",
        "evaluate",
        "artifact",
        "--reference",
        "base",
        "--reference",
        "other",
    ]);

    assert!(matches!(
        parse_outcome,
        Err(UsageError::RepeatedArgument("--reference"))
    ));
}
