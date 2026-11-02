//! Unit tests for mix parsing, rendering, and chunking with a whitespace
//! tokenizer. The Romeo and Juliet fixture is the only prose used for the
//! chunking contracts, per the repository's test-input rule.

use super::*;
use crate::error::QuantizationError;
use crate::progress::SilentProgressReporter;

/// Splits on whitespace and hashes each word to a stable id.
struct WhitespaceTokenizer;

impl CalibrationTokenizer for WhitespaceTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<u32>, QuantizationError> {
        Ok(text
            .split_whitespace()
            .map(|word| {
                word.bytes().fold(17_u32, |hash, byte| {
                    hash.wrapping_mul(31).wrapping_add(u32::from(byte))
                })
            })
            .collect())
    }
}

const ROMEO_AND_JULIET_EXCERPT: &str =
    include_str!("../../tests/fixtures/romeo_and_juliet_excerpt.txt");

#[test]
fn should_parse_the_bundled_mix_into_forty_samples_across_six_domains() {
    let samples =
        CalibrationMix::parse_jsonl(BUNDLED_OPTIQ_MIX_JSONL).expect("the bundled mix should parse");

    assert_eq!(samples.len(), 40);
    let mut domains: Vec<&str> = samples
        .iter()
        .map(|sample| sample.domain.as_str())
        .collect();
    domains.sort_unstable();
    domains.dedup();
    assert_eq!(
        domains,
        vec!["agent", "code", "instruct", "prose", "thought", "tool"]
    );
}

#[test]
fn should_render_chat_samples_with_role_prefixes() {
    let sample = CalibrationSample {
        domain: "thought".to_owned(),
        text: None,
        messages: Some(vec![
            CalibrationMessage {
                role: "user".to_owned(),
                content: "Who is Mercutio?".to_owned(),
            },
            CalibrationMessage {
                role: "assistant".to_owned(),
                content: "Romeo's friend.".to_owned(),
            },
        ]),
        tools: None,
    };

    assert_eq!(
        sample.render_text().as_deref(),
        Some("[user]: Who is Mercutio?\n\n[assistant]: Romeo's friend.")
    );
}

#[test]
fn should_cut_the_token_stream_into_fixed_length_sequences_deterministically() {
    let samples = vec![CalibrationSample {
        domain: "prose".to_owned(),
        text: Some(ROMEO_AND_JULIET_EXCERPT.to_owned()),
        messages: None,
        tools: None,
    }];
    let mut silent_reporter = SilentProgressReporter;

    let first_mix = CalibrationMix::from_samples(
        samples.clone(),
        &WhitespaceTokenizer,
        4,
        32,
        DEFAULT_SHUFFLE_SEED,
        &mut silent_reporter,
    )
    .expect("the excerpt is long enough for four sequences of 32 words");
    let second_mix = CalibrationMix::from_samples(
        samples,
        &WhitespaceTokenizer,
        4,
        32,
        DEFAULT_SHUFFLE_SEED,
        &mut silent_reporter,
    )
    .expect("the second load should succeed");

    assert_eq!(first_mix.samples().len(), 4);
    assert!(
        first_mix
            .samples()
            .iter()
            .all(|sample| sample.token_ids.len() == 32)
    );
    assert_eq!(first_mix.samples(), second_mix.samples());
}

#[test]
fn should_refuse_a_mix_shorter_than_one_sequence() {
    let samples = vec![CalibrationSample {
        domain: "prose".to_owned(),
        text: Some("Two households both alike in dignity".to_owned()),
        messages: None,
        tools: None,
    }];
    let mut silent_reporter = SilentProgressReporter;

    let mix_outcome = CalibrationMix::from_samples(
        samples,
        &WhitespaceTokenizer,
        1,
        512,
        DEFAULT_SHUFFLE_SEED,
        &mut silent_reporter,
    );

    assert!(matches!(
        mix_outcome,
        Err(QuantizationError::CalibrationMixTooShort { .. })
    ));
}

fn prose_sample(text: &str) -> CalibrationSample {
    CalibrationSample {
        domain: "prose".to_owned(),
        text: Some(text.to_owned()),
        messages: None,
        tools: None,
    }
}

#[test]
fn should_render_messages_over_text_when_a_sample_carries_both() {
    let mut sample = prose_sample("ignored");
    sample.messages = Some(vec![CalibrationMessage {
        role: "user".to_owned(),
        content: "Wherefore art thou Romeo?".to_owned(),
    }]);

    assert_eq!(
        sample.render_text().as_deref(),
        Some("[user]: Wherefore art thou Romeo?")
    );
}

#[test]
fn should_name_the_line_of_a_malformed_jsonl_entry() {
    let parse_outcome =
        CalibrationMix::parse_jsonl("{\"domain\": \"prose\", \"text\": \"a\"}\n\nnot json\n");

    match parse_outcome {
        Err(QuantizationError::CalibrationMixUnreadable { description, .. }) => {
            assert!(description.starts_with("line 3:"), "{description}");
        }
        other => panic!("expected an unreadable mix, got {other:?}"),
    }
}

#[test]
fn should_refuse_a_stream_of_exactly_one_sequence_as_optiq_does() {
    // OptiQ's offsets stop short of `total - L`, so exactly L tokens yield
    // no sequence at all.
    let mut silent_reporter = SilentProgressReporter;

    let mix_outcome = CalibrationMix::from_samples(
        vec![prose_sample("Two households both alike in dignity")],
        &WhitespaceTokenizer,
        1,
        6,
        DEFAULT_SHUFFLE_SEED,
        &mut silent_reporter,
    );

    assert!(matches!(
        mix_outcome,
        Err(QuantizationError::CalibrationMixTooShort {
            available_token_count: 6,
            ..
        })
    ));
}

#[test]
fn should_report_an_unreadable_mix_file_and_an_unloadable_tokenizer() {
    let directory = tempfile::tempdir().expect("a temporary directory should be creatable");
    let mut silent_reporter = SilentProgressReporter;

    let mix_outcome = CalibrationMix::from_jsonl_file(
        &directory.path().join("missing.jsonl"),
        &WhitespaceTokenizer,
        1,
        8,
        DEFAULT_SHUFFLE_SEED,
        &mut silent_reporter,
    );
    let tokenizer_outcome = HuggingFaceCalibrationTokenizer::from_tokenizer_file(
        &directory.path().join("tokenizer.json"),
    );

    assert!(matches!(
        mix_outcome,
        Err(QuantizationError::CalibrationMixUnreadable { .. })
    ));
    assert!(matches!(
        tokenizer_outcome,
        Err(QuantizationError::TokenizerFailure { .. })
    ));
}
