//! Parsing of the `quantize` verb: the `plan` sub-grammar, the candidate
//! bit-width validation, and the unknown-target rejection. These are pure
//! in-memory parse assertions; the filesystem-driven `plan` and `inspect`
//! journeys live in `tests/integration/quantize.rs`.

use crate::arguments::CliCommand;
use crate::arguments::UsageError;
use crate::verbs::quantize::QuantizeCommand;

use super::parse;

#[test]
#[ignore = "pending implementation of parse_quantize_command"]
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
#[ignore = "pending implementation of parse_quantize_command"]
fn should_reject_an_unknown_quantize_target() {
    let parse_outcome = parse(&["quantize", "shrink"]);

    assert!(matches!(
        parse_outcome,
        Err(UsageError::UnknownQuantizeTarget(target)) if target == "shrink"
    ));
}

#[test]
#[ignore = "pending implementation of parse_quantize_command"]
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
