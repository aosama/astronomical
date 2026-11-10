use super::{bundled_evaluation_prompts, evaluation_prompts_from, tokenized_evaluation_prompts};
use crate::calibration::{CalibrationMessage, CalibrationSample, CalibrationTokenizer};
use crate::error::QuantizationError;

/// One token per whitespace-separated word, numbered from 1.
struct WordCountingTokenizer;

impl CalibrationTokenizer for WordCountingTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<u32>, QuantizationError> {
        let mut token_ids = Vec::new();
        for (word_index, _) in text.split_whitespace().enumerate() {
            token_ids.push(word_index as u32 + 1);
        }
        Ok(token_ids)
    }
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
fn should_render_a_chat_as_role_prefixed_turns_and_prose_as_itself() {
    let chat_sample = CalibrationSample {
        domain: "thought".to_owned(),
        text: None,
        messages: Some(vec![
            CalibrationMessage {
                role: "user".to_owned(),
                content: "Wherefore art thou Romeo?".to_owned(),
            },
            CalibrationMessage {
                role: "assistant".to_owned(),
                content: "Deny thy father.".to_owned(),
            },
        ]),
        tools: None,
    };

    let prompts = evaluation_prompts_from(&[prose_sample("Two households"), chat_sample], 2);

    assert_eq!(
        prompts,
        vec![
            "Two households".to_owned(),
            "[user]: Wherefore art thou Romeo?\n\n[assistant]: Deny thy father.".to_owned(),
        ]
    );
}

#[test]
fn should_cycle_through_the_samples_when_more_prompts_are_requested() {
    let prompts = evaluation_prompts_from(&[prose_sample("first"), prose_sample("second")], 5);

    assert_eq!(prompts, vec!["first", "second", "first", "second", "first"]);
}

#[test]
fn should_return_no_prompts_for_no_samples_or_a_zero_request() {
    assert!(evaluation_prompts_from(&[], 4).is_empty());
    assert!(evaluation_prompts_from(&[prose_sample("first")], 0).is_empty());
}

#[test]
fn should_take_the_requested_number_of_prompts_from_the_bundled_mix() {
    let prompts = bundled_evaluation_prompts(64).expect("the bundled mix should parse");

    assert_eq!(prompts.len(), 64);
    // Forty samples in the mix, so the forty-first prompt is the first again.
    assert_eq!(prompts[40], prompts[0]);
}

#[test]
fn should_truncate_each_prompt_and_drop_one_that_encodes_to_nothing() {
    let prompts = vec![
        "But soft what light through yonder window breaks".to_owned(),
        "   ".to_owned(),
        "It is the east".to_owned(),
    ];

    let tokenized = tokenized_evaluation_prompts(&prompts, &WordCountingTokenizer, 5)
        .expect("the prompts should encode");

    assert_eq!(tokenized.len(), 2);
    assert_eq!(tokenized[0].token_ids, vec![1, 2, 3, 4, 5]);
    assert_eq!(tokenized[1].token_ids, vec![1, 2, 3, 4]);
}
