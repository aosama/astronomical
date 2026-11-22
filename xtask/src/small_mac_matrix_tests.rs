//! Unit tests for `cargo xtask small-mac-matrix`: the simulated limits, the
//! options, reading a chat answer and the table. No test touches an instance.

use crate::small_mac_matrix::{
    CellOutcome, MatrixCell, parse_chat_outcome, parse_matrix_options, render_matrix,
    simulated_memory_limit_gb,
};

fn arguments(argument_texts: &[&str]) -> Vec<String> {
    let mut owned_arguments = Vec::with_capacity(argument_texts.len());
    for argument_text in argument_texts {
        owned_arguments.push((*argument_text).to_owned());
    }
    owned_arguments
}

#[test]
fn should_simulate_two_thirds_of_a_macs_memory_rounded_down() {
    assert_eq!(simulated_memory_limit_gb(8), 5);
    assert_eq!(simulated_memory_limit_gb(16), 10);
    assert_eq!(simulated_memory_limit_gb(24), 16);
    assert_eq!(simulated_memory_limit_gb(32), 21);
}

#[test]
fn should_default_to_development_and_the_four_small_macs() {
    let matrix_options = parse_matrix_options(&[]).expect("no flags is valid");
    assert_eq!(matrix_options.base_url, "http://127.0.0.1:6733");
    assert_eq!(matrix_options.mac_memory_gb, [8, 16, 24, 32]);
    assert!(matrix_options.model_ids.is_empty());
}

#[test]
fn should_take_named_models_and_macs() {
    let matrix_options = parse_matrix_options(&arguments(&[
        "--model", "a", "--model", "b", "--mac-gb", "16",
    ]))
    .expect("named models and Macs are valid");
    assert_eq!(matrix_options.model_ids, ["a", "b"]);
    assert_eq!(matrix_options.mac_memory_gb, [16]);
}

#[test]
fn should_refuse_a_mac_smaller_than_any_sold_and_malformed_flags() {
    for bad_arguments in [
        vec!["--mac-gb", "4"],
        vec!["--mac-gb", "lots"],
        vec!["--mac-gb"],
        vec!["--model"],
        vec!["--fast"],
    ] {
        assert!(
            parse_matrix_options(&arguments(&bad_arguments)).is_err(),
            "{bad_arguments:?}"
        );
    }
}

#[test]
fn should_read_a_reply_an_error_code_and_an_empty_answer() {
    assert_eq!(
        parse_chat_outcome(r#"{"choices":[{"message":{"content":"Thou art"}}]}"#, 3.25),
        CellOutcome::Replied {
            seconds_to_reply: 3.25
        }
    );
    assert_eq!(
        parse_chat_outcome(
            r#"{"error":{"message":"x","code":"model_load_failed"}}"#,
            1.0
        ),
        CellOutcome::Failed {
            error_code: "model_load_failed".to_owned()
        }
    );
    assert_eq!(
        parse_chat_outcome("", 1.0),
        CellOutcome::Failed {
            error_code: "no_answer".to_owned()
        }
    );
}

#[test]
fn should_render_each_outcome_and_say_what_the_simulation_leaves_out() {
    let matrix_text = render_matrix(&[
        MatrixCell {
            model_id: "small-model".to_owned(),
            mac_memory_gb: 8,
            memory_limit_gb: 5,
            outcome: CellOutcome::Replied {
                seconds_to_reply: 4.26,
            },
        },
        MatrixCell {
            model_id: "large-model".to_owned(),
            mac_memory_gb: 8,
            memory_limit_gb: 5,
            outcome: CellOutcome::LimitRefused,
        },
    ]);
    assert!(matrix_text.contains("replied in 4.3 s"));
    assert!(matrix_text.contains("memory limit refused"));
    assert!(matrix_text.contains("necessary, not sufficient"));
}
