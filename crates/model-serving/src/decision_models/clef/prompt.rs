//! The tokens Clef reads for one request, laid out as
//! `joint_schema_model.py`'s `encode_record` lays them out.
//!
//! One sequence holds everything: a system prompt, the state, a schema that
//! lists every question with its options, and an assistant suffix. Each
//! piece is tokenized on its own and the pieces are joined, exactly as the
//! reference joins them, so a question's instructions and each option's text
//! occupy known spans; the head pools the backbone's states over those spans.
//! Text is tokenized as the reference does, without escaping: caller text is
//! read the same way Clef was trained to read it.
//!
//! The whole sequence may be at most 16,384 tokens. A longer state is cut
//! from its end to fit, as the reference cuts it, and the answers say so.

use astronomical_decision_contract::clef_rendering::{
    ClefQuestionText, clef_question_text, clef_render,
};
use astronomical_decision_contract::{DecisionFailure, DecisionQuestion, OrderedJsonValue};

use super::tokenizer::ClefTokenizer;

/// `encode_record`'s default `max_length`: the whole sequence, schema included.
pub(crate) const CLEF_MAXIMUM_SEQUENCE_TOKENS: usize = 16_384;
/// `SYSTEM_PROMPT`.
const CLEF_SYSTEM_PROMPT: &str = "Read the complete state and schema. Decide every field jointly. Each answer must be exactly one of that field's allowed options.";
const CLEF_ASSISTANT_SUFFIX: &str =
    "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:";

/// A `[start, end)` run of token positions.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct TokenSpan {
    pub(crate) start: usize,
    pub(crate) end: usize,
}

/// One question's place in the sequence.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ClefQuestionLayout {
    pub(crate) question_text: ClefQuestionText,
    pub(crate) instructions_span: TokenSpan,
    /// Each option's span, in the order Clef scores the options.
    pub(crate) option_spans: Vec<TokenSpan>,
}

/// One request's whole sequence.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ClefSequence {
    pub(crate) token_ids: Vec<u32>,
    pub(crate) questions: Vec<ClefQuestionLayout>,
    /// The state's whole length in tokens, and how much of it was kept.
    pub(crate) state_token_count: usize,
    pub(crate) state_tokens_read: usize,
}

pub(crate) fn build_clef_sequence(
    clef_tokenizer: &ClefTokenizer,
    state: &OrderedJsonValue,
    questions: &[DecisionQuestion],
) -> Result<ClefSequence, DecisionFailure> {
    let (schema_token_ids, schema_questions) = encode_schema(clef_tokenizer, questions)?;
    let prefix_token_ids = clef_tokenizer.encode(&format!(
        "<|im_start|>system\n{CLEF_SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nSTATE:\n"
    ))?;
    let suffix_token_ids = clef_tokenizer.encode(CLEF_ASSISTANT_SUFFIX)?;
    let state_token_ids = clef_tokenizer.encode(&clef_render(state))?;
    let fixed_token_count =
        prefix_token_ids.len() + schema_token_ids.len() + suffix_token_ids.len();
    if fixed_token_count > CLEF_MAXIMUM_SEQUENCE_TOKENS {
        return Err(DecisionFailure::invalid_request(&format!(
            "the questions need {fixed_token_count} tokens before any state, over Clef's \
             {CLEF_MAXIMUM_SEQUENCE_TOKENS}-token limit: ask fewer questions or shorten them"
        )));
    }
    let state_tokens_read = state_token_ids
        .len()
        .min(CLEF_MAXIMUM_SEQUENCE_TOKENS - fixed_token_count);
    let schema_offset = prefix_token_ids.len() + state_tokens_read;
    let mut token_ids = prefix_token_ids;
    token_ids.extend(&state_token_ids[..state_tokens_read]);
    token_ids.extend(&schema_token_ids);
    token_ids.extend(&suffix_token_ids);
    let mut question_layouts = schema_questions;
    for question_layout in &mut question_layouts {
        question_layout.instructions_span =
            shifted(question_layout.instructions_span, schema_offset);
        for option_span in &mut question_layout.option_spans {
            *option_span = shifted(*option_span, schema_offset);
        }
    }
    Ok(ClefSequence {
        token_ids,
        questions: question_layouts,
        state_token_count: state_token_ids.len(),
        state_tokens_read,
    })
}

/// The schema's tokens, with spans relative to the schema's start.
fn encode_schema(
    clef_tokenizer: &ClefTokenizer,
    questions: &[DecisionQuestion],
) -> Result<(Vec<u32>, Vec<ClefQuestionLayout>), DecisionFailure> {
    let mut schema_token_ids = clef_tokenizer.encode("\n\nSCHEMA FIELDS:\n")?;
    let mut question_layouts = Vec::with_capacity(questions.len());
    for (question_index, question) in questions.iter().enumerate() {
        schema_token_ids.extend(clef_tokenizer.encode(&format!(
            "\nFIELD {}\nID: {}\nTYPE: {}\nINSTRUCTION: ",
            question_index + 1,
            question.name,
            question.kind().wire_name()
        ))?);
        let question_text = clef_question_text(question);
        let instructions_start = schema_token_ids.len();
        schema_token_ids.extend(clef_tokenizer.encode(&question_text.instructions_text)?);
        let instructions_span = TokenSpan {
            start: instructions_start,
            end: schema_token_ids.len(),
        };
        schema_token_ids.extend(clef_tokenizer.encode("\nALLOWED OPTIONS:\n")?);
        let mut option_spans = Vec::with_capacity(question_text.options.len());
        for (option_index, option_text) in question_text.options.iter().enumerate() {
            schema_token_ids
                .extend(clef_tokenizer.encode(&format!("OPTION {}: ", option_index + 1))?);
            let option_start = schema_token_ids.len();
            schema_token_ids.extend(clef_tokenizer.encode(&option_text.semantics_text)?);
            option_spans.push(TokenSpan {
                start: option_start,
                end: schema_token_ids.len(),
            });
            schema_token_ids.extend(clef_tokenizer.encode("\n")?);
        }
        schema_token_ids.extend(clef_tokenizer.encode("END FIELD\n")?);
        question_layouts.push(ClefQuestionLayout {
            question_text,
            instructions_span,
            option_spans,
        });
    }
    Ok((schema_token_ids, question_layouts))
}

const fn shifted(token_span: TokenSpan, offset: usize) -> TokenSpan {
    TokenSpan {
        start: token_span.start + offset,
        end: token_span.end + offset,
    }
}

#[cfg(test)]
#[path = "prompt_tests.rs"]
mod tests;
