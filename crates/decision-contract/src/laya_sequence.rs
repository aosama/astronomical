//! Lays one Laya question out as the token sequence the encoder reads.
//!
//! The layout is `[CLS] <type> question: <instructions> [SEP] [MASK] option
//! [MASK] option … [SEP] state [SEP]`. The decision head reads one logit at
//! each `[MASK]`, so the marker positions are returned with the tokens.
//!
//! Every budget rule below is the reference implementation's. They decide
//! which tokens of a long option, instruction or state the model never sees,
//! so matching them token for token is what keeps answers equal to the
//! reference on inputs that do not fit.

use thiserror::Error;

use crate::laya_rendering::{
    LayaRenderingError, laya_instruction_text, laya_option_texts, laya_state_text,
};
use crate::ordered_json::OrderedJsonValue;
use crate::question::DecisionQuestion;
use crate::token_encoder::{DecisionTokenEncoder, TokenEncodingError};

/// The most tokens of one option's text that are kept, before any shared-budget cut.
const OPTION_TEXT_TOKEN_LIMIT: usize = 48;
/// The head budget the options must leave free before they are cut evenly.
const MINIMUM_FREE_HEAD_TOKENS: i64 = 16;
/// The fewest tokens an option keeps (marker included) when options are cut evenly.
const MINIMUM_TOKENS_PER_CUT_OPTION: i64 = 4;
/// The fewest instruction tokens kept however many options there are.
const MINIMUM_INSTRUCTION_TOKENS: i64 = 8;

/// The special tokens of a Laya checkpoint's tokenizer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LayaSpecialTokens {
    pub cls_token_id: u32,
    pub sep_token_id: u32,
    pub mask_token_id: u32,
    /// The mask token as text, such as `[MASK]`. It is removed from every
    /// piece of request text so a request cannot plant a marker of its own.
    pub mask_token_text: String,
}

/// The two token budgets a Laya checkpoint was trained with.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LayaSequenceLimits {
    /// The longest sequence the checkpoint reads (`max_len`).
    pub maximum_sequence_tokens: usize,
    /// How much of that the instruction and options may take (`head_max_len`).
    pub head_token_budget: usize,
}

/// One question laid out for the encoder.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LayaQuestionSequence {
    pub token_ids: Vec<u32>,
    /// The position of each option's `[MASK]`, in canonical option order.
    pub option_marker_positions: Vec<usize>,
    /// The per-option token cap applied when the options overran the head
    /// budget, marker included; `None` when no option was cut for that reason.
    pub option_token_cap: Option<usize>,
    /// How many options still have a token span unlike every other option's.
    /// Fewer than the option count means the cut made options indistinguishable.
    pub distinct_option_count: usize,
    /// How many tokens the whole state encodes to.
    pub state_token_count: usize,
    /// How many of those the sequence had room for.
    pub state_tokens_used: usize,
}

/// Why a question cannot be laid out.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum LayaSequenceError {
    #[error(transparent)]
    Rendering(#[from] LayaRenderingError),
    #[error(transparent)]
    TokenEncoding(#[from] TokenEncodingError),
    #[error(
        "question {question:?}: only {surviving_marker_count} of its {option_count} option markers fit in {maximum_sequence_tokens} tokens with {head_token_budget} spent on the question; use fewer options"
    )]
    OptionsDoNotFit {
        question: String,
        surviving_marker_count: usize,
        option_count: usize,
        maximum_sequence_tokens: usize,
        head_token_budget: usize,
    },
}

/// Encodes a request's state once, for reuse across every question about it.
pub fn encode_laya_state(
    token_encoder: &dyn DecisionTokenEncoder,
    special_tokens: &LayaSpecialTokens,
    state: &OrderedJsonValue,
) -> Result<Vec<u32>, TokenEncodingError> {
    let state_text = laya_state_text(state).replace(&special_tokens.mask_token_text, " ");
    token_encoder.encode_without_special_tokens(&state_text)
}

/// The instruction and options of one question, before the state is appended.
struct LayaQuestionHead {
    token_ids: Vec<u32>,
    option_marker_positions: Vec<usize>,
    option_token_cap: Option<usize>,
    distinct_option_count: usize,
}

/// Lays out one question against an already encoded state.
pub fn build_laya_question_sequence(
    token_encoder: &dyn DecisionTokenEncoder,
    special_tokens: &LayaSpecialTokens,
    limits: LayaSequenceLimits,
    question: &DecisionQuestion,
    state_token_ids: &[u32],
    truncate_state_from_the_left: bool,
) -> Result<LayaQuestionSequence, LayaSequenceError> {
    let head = build_question_head(token_encoder, special_tokens, limits, question)?;
    let option_count = head.option_marker_positions.len();
    let mut token_ids = head.token_ids;
    let kept_state_token_ids = state_tokens_that_fit(
        state_token_ids,
        limits.maximum_sequence_tokens,
        token_ids.len(),
        truncate_state_from_the_left,
    );
    token_ids.extend_from_slice(kept_state_token_ids);
    token_ids.push(special_tokens.sep_token_id);
    token_ids.truncate(limits.maximum_sequence_tokens);
    let mut surviving_marker_positions = Vec::with_capacity(option_count);
    for marker_position in head.option_marker_positions {
        if marker_position < limits.maximum_sequence_tokens {
            surviving_marker_positions.push(marker_position);
        }
    }
    if surviving_marker_positions.len() != option_count {
        // A dropped marker means an option has no logit to read. Answering
        // over the remaining options would silently change the question.
        return Err(LayaSequenceError::OptionsDoNotFit {
            question: question.name.clone(),
            surviving_marker_count: surviving_marker_positions.len(),
            option_count,
            maximum_sequence_tokens: limits.maximum_sequence_tokens,
            head_token_budget: limits.head_token_budget,
        });
    }
    Ok(LayaQuestionSequence {
        token_ids,
        option_marker_positions: surviving_marker_positions,
        option_token_cap: head.option_token_cap,
        distinct_option_count: head.distinct_option_count,
        state_token_count: state_token_ids.len(),
        state_tokens_used: kept_state_token_ids.len(),
    })
}

/// The part of the state the sequence has room for after its head.
///
/// One position is held back for the closing separator. A state cut from the
/// left keeps its end, and with no room at all keeps nothing.
fn state_tokens_that_fit(
    state_token_ids: &[u32],
    maximum_sequence_tokens: usize,
    head_token_count: usize,
    truncate_state_from_the_left: bool,
) -> &[u32] {
    let state_room = maximum_sequence_tokens.saturating_sub(head_token_count + 1);
    if truncate_state_from_the_left {
        &state_token_ids[state_token_ids.len().saturating_sub(state_room)..]
    } else {
        &state_token_ids[..state_room.min(state_token_ids.len())]
    }
}

fn build_question_head(
    token_encoder: &dyn DecisionTokenEncoder,
    special_tokens: &LayaSpecialTokens,
    limits: LayaSequenceLimits,
    question: &DecisionQuestion,
) -> Result<LayaQuestionHead, LayaSequenceError> {
    let mask_token_text = special_tokens.mask_token_text.as_str();
    // The reference strips the mask text from the instructions alone. Doing
    // it on the whole line is the same thing, because the "<kind> question: "
    // prefix this crate adds never contains a tokenizer's mask text.
    let instruction_text = laya_instruction_text(question)?.replace(mask_token_text, " ");
    let mut instruction_token_ids =
        token_encoder.encode_without_special_tokens(&instruction_text)?;
    let mut option_spans = encode_option_spans(token_encoder, special_tokens, question)?;
    let head_token_budget = i64::try_from(limits.head_token_budget).unwrap_or(i64::MAX);
    let mut free_head_tokens = head_token_budget - total_span_tokens(&option_spans);
    let mut option_token_cap = None;
    if free_head_tokens < MINIMUM_FREE_HEAD_TOKENS {
        // The options overran the budget: give each the same share so no
        // option is dropped whole while another keeps its full text.
        let option_count = i64::try_from(option_spans.len()).unwrap_or(i64::MAX).max(1);
        let even_share = (head_token_budget - MINIMUM_FREE_HEAD_TOKENS).div_euclid(option_count);
        let tokens_per_option =
            usize::try_from(even_share.max(MINIMUM_TOKENS_PER_CUT_OPTION)).unwrap_or(usize::MAX);
        for option_span in &mut option_spans {
            option_span.truncate(tokens_per_option);
        }
        option_token_cap = Some(tokens_per_option);
        free_head_tokens = head_token_budget - total_span_tokens(&option_spans);
    }
    let instruction_token_limit =
        usize::try_from(free_head_tokens.max(MINIMUM_INSTRUCTION_TOKENS)).unwrap_or(usize::MAX);
    instruction_token_ids.truncate(instruction_token_limit);

    let mut token_ids = vec![special_tokens.cls_token_id];
    token_ids.extend_from_slice(&instruction_token_ids);
    token_ids.push(special_tokens.sep_token_id);
    let mut option_marker_positions = Vec::with_capacity(option_spans.len());
    for option_span in &option_spans {
        option_marker_positions.push(token_ids.len());
        token_ids.extend_from_slice(option_span);
    }
    token_ids.push(special_tokens.sep_token_id);
    Ok(LayaQuestionHead {
        token_ids,
        option_marker_positions,
        option_token_cap,
        distinct_option_count: count_distinct_spans(&option_spans),
    })
}

/// One span per option: its `[MASK]` marker followed by the option's text tokens.
fn encode_option_spans(
    token_encoder: &dyn DecisionTokenEncoder,
    special_tokens: &LayaSpecialTokens,
    question: &DecisionQuestion,
) -> Result<Vec<Vec<u32>>, TokenEncodingError> {
    let option_texts = laya_option_texts(question);
    let mut option_spans = Vec::with_capacity(option_texts.len());
    for option_text in option_texts {
        // The leading space makes the first word tokenize as it would
        // mid-sentence, which is how the checkpoints saw options in training.
        let spaced_option_text = format!(
            " {}",
            option_text.replace(&special_tokens.mask_token_text, " ")
        );
        let mut option_text_token_ids =
            token_encoder.encode_without_special_tokens(&spaced_option_text)?;
        option_text_token_ids.truncate(OPTION_TEXT_TOKEN_LIMIT);
        let mut option_span = Vec::with_capacity(option_text_token_ids.len() + 1);
        option_span.push(special_tokens.mask_token_id);
        option_span.extend_from_slice(&option_text_token_ids);
        option_spans.push(option_span);
    }
    Ok(option_spans)
}

fn total_span_tokens(option_spans: &[Vec<u32>]) -> i64 {
    let mut total_tokens = 0_usize;
    for option_span in option_spans {
        total_tokens += option_span.len();
    }
    i64::try_from(total_tokens).unwrap_or(i64::MAX)
}

fn count_distinct_spans(option_spans: &[Vec<u32>]) -> usize {
    let mut distinct_spans: Vec<&Vec<u32>> = Vec::with_capacity(option_spans.len());
    for option_span in option_spans {
        if !distinct_spans.contains(&option_span) {
            distinct_spans.push(option_span);
        }
    }
    distinct_spans.len()
}

#[cfg(test)]
#[path = "laya_sequence_tests.rs"]
mod tests;
