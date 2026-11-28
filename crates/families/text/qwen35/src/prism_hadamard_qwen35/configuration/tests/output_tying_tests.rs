//! Output tying: a pack's two `tie_word_embeddings` and Prism's `tied_output`
//! must agree.

use super::{FoldedPackConfigError, parse_config, shipped_shape_config_json};

fn config_with_tying(
    pack_ties_word_embeddings: bool,
    text_ties_word_embeddings: bool,
    pack_declares_tied_output: Option<bool>,
) -> serde_json::Value {
    let mut document = shipped_shape_config_json();
    document["tie_word_embeddings"] = serde_json::json!(pack_ties_word_embeddings);
    document["text_config"]["tie_word_embeddings"] = serde_json::json!(text_ties_word_embeddings);
    if let Some(declares_tied_output) = pack_declares_tied_output {
        document["tied_output"] = serde_json::json!(declares_tied_output);
    }
    document
}

#[test]
fn should_accept_a_tied_pack_whose_three_declarations_agree() {
    let config = parse_config(&config_with_tying(true, true, Some(true)))
        .expect("a tied pack that says so everywhere should parse");
    assert!(config.has_tied_output());
}

#[test]
fn should_read_an_untied_pack_without_the_tied_output_marker() {
    let config = parse_config(&config_with_tying(false, false, None))
        .expect("the 9B and 27B packs omit the marker");
    assert!(!config.has_tied_output());
}

#[test]
fn should_refuse_tying_declarations_that_disagree() {
    for (pack_ties, text_ties, declares_tied_output) in [
        (true, true, None),
        (true, false, Some(true)),
        (false, true, Some(true)),
        (false, false, Some(true)),
    ] {
        let error = parse_config(&config_with_tying(
            pack_ties,
            text_ties,
            declares_tied_output,
        ))
        .expect_err("disagreeing tying declarations must be refused");
        assert!(
            matches!(error, FoldedPackConfigError::OutputTyingDisagreement { .. }),
            "{error}"
        );
    }
}
