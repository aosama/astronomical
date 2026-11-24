//! Golden-vector tests against the reference Laya implementation.
//!
//! The fixture (`src/laya_golden_reference_vectors.json`) was produced by
//! running `tests/fixtures/generate_laya_golden_reference_vectors.py` against
//! the reference package named in the fixture's `reference` field. It pins the
//! JSON text, the token layout and budget rules, and the calibration maths.
//! Regenerating it needs that package; the fixture is checked in so these
//! tests need neither Python nor a model.

use crate::answer::{answer_confidence, distribution_confidence};
use crate::calibration::{
    BucketTemperature, KindTemperatures, TemperatureTable, calibrated_probabilities,
    temperature_bucket_key,
};
use crate::laya_rendering::{
    laya_option_texts, laya_state_text, laya_truncates_state_from_the_left,
};
use crate::laya_sequence::{
    LayaQuestionSequence, LayaSequenceError, LayaSequenceLimits, LayaSpecialTokens,
    build_laya_question_sequence, encode_laya_state,
};
use crate::ordered_json::{OrderedJsonMember, OrderedJsonValue};
use crate::python_json_text::{JsonTextStyle, python_json_text};
use crate::question::DecisionQuestionKind;
use crate::system_one_questions::parse_system_one_questions;
use crate::test_support::{WordHashTokenEncoder, assert_close, ordered_json};

/// The reference computes probabilities from the same decimal logits in
/// 64-bit floats, while this crate receives them as 32-bit model outputs.
const PROBABILITY_TOLERANCE: f64 = 1e-6;

fn golden_vectors() -> OrderedJsonValue {
    ordered_json(include_str!("laya_golden_reference_vectors.json"))
}

fn golden_field<'a>(vector: &'a OrderedJsonValue, key: &str) -> &'a OrderedJsonValue {
    vector
        .member(key)
        .unwrap_or_else(|| panic!("the golden vector has a `{key}` field"))
}

fn text_field<'a>(vector: &'a OrderedJsonValue, key: &str) -> &'a str {
    golden_field(vector, key)
        .as_text()
        .unwrap_or_else(|| panic!("golden field `{key}` is a string"))
}

fn array_field<'a>(vector: &'a OrderedJsonValue, key: &str) -> &'a [OrderedJsonValue] {
    golden_field(vector, key)
        .as_array()
        .unwrap_or_else(|| panic!("golden field `{key}` is an array"))
}

fn numeric_value(value: &OrderedJsonValue) -> f64 {
    match value {
        OrderedJsonValue::SignedInteger(integer) => *integer as f64,
        OrderedJsonValue::UnsignedInteger(integer) => *integer as f64,
        OrderedJsonValue::Float(float) => *float,
        other_value => panic!("expected a number, got {}", other_value.type_name()),
    }
}

fn count_in_field(vector: &OrderedJsonValue, key: &str) -> usize {
    numeric_value(golden_field(vector, key)) as usize
}

fn counts_of(values: &[OrderedJsonValue]) -> Vec<usize> {
    let mut counts = Vec::with_capacity(values.len());
    for value in values {
        counts.push(numeric_value(value) as usize);
    }
    counts
}

fn golden_special_tokens(vectors: &OrderedJsonValue) -> LayaSpecialTokens {
    let tokenizer = golden_field(vectors, "tokenizer");
    LayaSpecialTokens {
        cls_token_id: count_in_field(tokenizer, "cls_token_id") as u32,
        sep_token_id: count_in_field(tokenizer, "sep_token_id") as u32,
        mask_token_id: count_in_field(tokenizer, "mask_token_id") as u32,
        mask_token_text: text_field(tokenizer, "mask_token_text").to_owned(),
    }
}

#[test]
fn should_write_json_text_exactly_as_the_reference_does() {
    let vectors = golden_vectors();
    for json_text_vector in array_field(&vectors, "json_text") {
        let value = golden_field(json_text_vector, "value");
        // A string value is a state passed through as written, not JSON text.
        if value.as_text().is_some() {
            continue;
        }
        assert_eq!(
            python_json_text(value, JsonTextStyle::PYTHON_DEFAULT),
            text_field(json_text_vector, "python_default")
        );
        assert_eq!(
            python_json_text(value, JsonTextStyle::COMPACT_SORTED_KEYS),
            text_field(json_text_vector, "compact_sorted_keys")
        );
    }
}

#[test]
fn should_lay_out_every_golden_question_exactly_as_the_reference_does() {
    let vectors = golden_vectors();
    let special_tokens = golden_special_tokens(&vectors);
    let sequence_vectors = array_field(&vectors, "sequences");
    assert!(
        sequence_vectors.len() >= 15,
        "the fixture covers every layout rule"
    );
    for sequence_vector in sequence_vectors {
        check_sequence_vector(sequence_vector, &special_tokens);
    }
}

fn check_sequence_vector(sequence_vector: &OrderedJsonValue, special_tokens: &LayaSpecialTokens) {
    let case_name = text_field(sequence_vector, "name");
    let questions = OrderedJsonValue::Object(vec![OrderedJsonMember {
        key: case_name.to_owned(),
        value: golden_field(sequence_vector, "question").clone(),
    }]);
    let question = parse_system_one_questions(&questions)
        .unwrap_or_else(|error| panic!("case {case_name}: {error}"))
        .remove(0);
    let state = golden_field(sequence_vector, "state");

    let mut expected_option_texts = Vec::new();
    for option_text in array_field(sequence_vector, "option_texts") {
        expected_option_texts.push(option_text.as_text().expect("a string").to_owned());
    }
    assert_eq!(
        laya_option_texts(&question),
        expected_option_texts,
        "case {case_name}"
    );
    assert_eq!(
        laya_state_text(state),
        text_field(sequence_vector, "state_text"),
        "case {case_name}"
    );

    let state_token_ids = encode_laya_state(&WordHashTokenEncoder, special_tokens, state)
        .expect("the stand-in encoder never fails");
    let sequence_result = build_laya_question_sequence(
        &WordHashTokenEncoder,
        special_tokens,
        LayaSequenceLimits {
            maximum_sequence_tokens: count_in_field(sequence_vector, "max_len"),
            head_token_budget: count_in_field(sequence_vector, "head_max_len"),
        },
        &question,
        &state_token_ids,
        laya_truncates_state_from_the_left(state),
    );
    let expected_marker_positions = counts_of(array_field(sequence_vector, "marker_positions"));
    if *golden_field(sequence_vector, "options_fit") == OrderedJsonValue::Boolean(false) {
        let Err(LayaSequenceError::OptionsDoNotFit {
            surviving_marker_count,
            ..
        }) = sequence_result
        else {
            panic!("case {case_name}: the reference dropped markers, so this must be rejected");
        };
        assert_eq!(surviving_marker_count, expected_marker_positions.len());
        return;
    }
    let sequence = sequence_result.unwrap_or_else(|error| panic!("case {case_name}: {error}"));
    assert_eq!(
        sequence.option_marker_positions, expected_marker_positions,
        "case {case_name}"
    );
    assert_sequence_matches_the_reference(&sequence, sequence_vector, case_name);
}

fn assert_sequence_matches_the_reference(
    sequence: &LayaQuestionSequence,
    sequence_vector: &OrderedJsonValue,
    case_name: &str,
) {
    let mut actual_token_ids = Vec::with_capacity(sequence.token_ids.len());
    for token_id in &sequence.token_ids {
        actual_token_ids.push(*token_id as usize);
    }
    assert_eq!(
        actual_token_ids,
        counts_of(array_field(sequence_vector, "token_ids")),
        "case {case_name}"
    );
    let expected_option_token_cap = match golden_field(sequence_vector, "option_token_cap") {
        OrderedJsonValue::Null => None,
        cap_value => Some(numeric_value(cap_value) as usize),
    };
    assert_eq!(
        sequence.option_token_cap, expected_option_token_cap,
        "case {case_name}"
    );
    assert_eq!(
        sequence.distinct_option_count,
        count_in_field(sequence_vector, "distinct_option_count"),
        "case {case_name}"
    );
    assert_eq!(
        sequence.state_token_count,
        count_in_field(sequence_vector, "state_token_count"),
        "case {case_name}"
    );
    assert_eq!(
        sequence.state_tokens_used,
        count_in_field(sequence_vector, "state_tokens_used"),
        "case {case_name}"
    );
}

fn golden_temperature_table(vectors: &OrderedJsonValue) -> TemperatureTable {
    let calibration_table = golden_field(vectors, "calibration_table");
    let kind_temperatures = array_field(calibration_table, "temperature");
    let mut bucket_temperatures = Vec::new();
    let by_options = golden_field(calibration_table, "temperature_by_options")
        .as_object()
        .expect("an object");
    for bucket_member in by_options {
        bucket_temperatures.push(BucketTemperature {
            bucket_key: bucket_member.key.clone(),
            temperature: numeric_value(&bucket_member.value),
        });
    }
    // A Laya checkpoint lists its per-kind temperatures as choice, score, noul.
    TemperatureTable::from_raw_temperatures(
        KindTemperatures {
            choice: numeric_value(&kind_temperatures[0]),
            score: numeric_value(&kind_temperatures[1]),
            noul: numeric_value(&kind_temperatures[2]),
        },
        &bucket_temperatures,
    )
}

#[test]
fn should_calibrate_every_golden_case_exactly_as_the_reference_does() {
    let vectors = golden_vectors();
    let temperature_table = golden_temperature_table(&vectors);
    let calibration_vectors = array_field(&vectors, "calibration");
    assert!(
        calibration_vectors.len() >= 9,
        "the fixture covers every bucket"
    );
    for calibration_vector in calibration_vectors {
        let kind = DecisionQuestionKind::from_wire_name(text_field(calibration_vector, "type"))
            .expect("a known kind");
        let mut option_logits = Vec::new();
        for logit in array_field(calibration_vector, "logits") {
            option_logits.push(numeric_value(logit) as f32);
        }
        let option_count = option_logits.len();
        assert_eq!(
            temperature_bucket_key(kind, option_count),
            text_field(calibration_vector, "bucket_key")
        );
        let temperature = temperature_table.temperature_for(kind, option_count);
        assert_close(
            temperature,
            numeric_value(golden_field(calibration_vector, "temperature")),
            0.0,
        );
        let probabilities = calibrated_probabilities(&option_logits, temperature);
        let expected_probabilities = array_field(calibration_vector, "probabilities");
        assert_eq!(probabilities.len(), expected_probabilities.len());
        for (probability, expected_probability) in probabilities.iter().zip(expected_probabilities)
        {
            assert_close(
                *probability,
                numeric_value(expected_probability),
                PROBABILITY_TOLERANCE,
            );
        }
        assert_close(
            answer_confidence(&probabilities),
            numeric_value(golden_field(calibration_vector, "answer_confidence")),
            PROBABILITY_TOLERANCE,
        );
        assert_close(
            distribution_confidence(&probabilities),
            numeric_value(golden_field(calibration_vector, "distribution_confidence")),
            PROBABILITY_TOLERANCE,
        );
    }
}
