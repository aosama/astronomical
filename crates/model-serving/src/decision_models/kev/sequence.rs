//! The tokens Kev reads for one request, laid out as `kev/model.py`'s
//! `encode` and `rows_of` lay them out for serving.
//!
//! The state is `<|fim_prefix|>` and the rendered state. Each question is a
//! row continuing the state: `<|fim_middle|>` and its instructions, each
//! option as `<|box_start|> option <|box_end|>`, then `<|fim_suffix|>`. The
//! head reads the row's last token (the decision token) and each option's
//! closing token.

use std::collections::HashSet;

use astronomical_decision_contract::kev_rendering::{kev_question_text, kev_render};
use astronomical_decision_contract::{DecisionFailure, DecisionQuestion, OrderedJsonValue};

use super::tokenizer::KevTokenizer;

/// The longest state the reference serves, its state token included
/// (`SERVE_MAX_STATE`). A longer state is refused, as the reference refuses
/// it, rather than cut: an answer from part of a document would look like
/// an answer about all of it.
pub(crate) const KEV_MAXIMUM_STATE_TOKENS: usize = 65_536;
/// The longest row, state included (`SERVE_MAX_BRANCH`): at least 8,192
/// tokens for each question beyond the longest state.
pub(crate) const KEV_MAXIMUM_ROW_TOKENS: usize = KEV_MAXIMUM_STATE_TOKENS + 8_192;

/// One question's row.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct KevQuestionRow {
    pub(crate) token_ids: Vec<u32>,
    /// The offsets, within the row, of each option's closing token.
    pub(crate) option_closing_offsets: Vec<usize>,
    /// How many options read as token sequences no other option shares.
    pub(crate) distinguishable_option_count: usize,
}

impl KevQuestionRow {
    /// The decision token is always the row's last.
    pub(crate) fn decision_offset(&self) -> usize {
        self.token_ids.len() - 1
    }
}

/// One question's whole row as the backbone reads it, state included, with
/// absolute readout offsets: the shape a reference implementation that runs
/// each question as one causal row records, so the two compare directly.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KevRowLayout {
    pub token_ids: Vec<u32>,
    pub option_closing_offsets: Vec<usize>,
}

/// One request's state and rows.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct KevRequestSequence {
    pub(crate) state_token_ids: Vec<u32>,
    pub(crate) question_rows: Vec<KevQuestionRow>,
}

impl KevRequestSequence {
    /// Every question's row joined to the state.
    pub(crate) fn row_layouts(&self) -> Vec<KevRowLayout> {
        let state_length = self.state_token_ids.len();
        let mut row_layouts = Vec::with_capacity(self.question_rows.len());
        for question_row in &self.question_rows {
            let mut token_ids = self.state_token_ids.clone();
            token_ids.extend(&question_row.token_ids);
            let mut option_closing_offsets =
                Vec::with_capacity(question_row.option_closing_offsets.len());
            for option_closing_offset in &question_row.option_closing_offsets {
                option_closing_offsets.push(state_length + option_closing_offset);
            }
            row_layouts.push(KevRowLayout {
                token_ids,
                option_closing_offsets,
            });
        }
        row_layouts
    }
}

pub(crate) fn build_kev_sequence(
    kev_tokenizer: &KevTokenizer,
    state: &OrderedJsonValue,
    questions: &[DecisionQuestion],
) -> Result<KevRequestSequence, DecisionFailure> {
    let mut state_token_ids = vec![kev_tokenizer.control_tokens.state_token_id];
    state_token_ids.extend(kev_tokenizer.encode_caller_text(&kev_render(state))?);
    if state_token_ids.len() > KEV_MAXIMUM_STATE_TOKENS {
        return Err(DecisionFailure::invalid_request(&format!(
            "state is {} tokens, over the {KEV_MAXIMUM_STATE_TOKENS}-token limit (the state token included): \
             shorten the document or split it across requests",
            state_token_ids.len()
        )));
    }
    let mut question_rows = Vec::with_capacity(questions.len());
    for question in questions {
        let question_row = build_question_row(kev_tokenizer, question)?;
        if state_token_ids.len() + question_row.token_ids.len() > KEV_MAXIMUM_ROW_TOKENS {
            return Err(DecisionFailure::invalid_request(&format!(
                "question {:?} is {} tokens with a {}-token state, over the {KEV_MAXIMUM_ROW_TOKENS}-token row limit",
                question.name,
                question_row.token_ids.len(),
                state_token_ids.len()
            )));
        }
        question_rows.push(question_row);
    }
    Ok(KevRequestSequence {
        state_token_ids,
        question_rows,
    })
}

fn build_question_row(
    kev_tokenizer: &KevTokenizer,
    question: &DecisionQuestion,
) -> Result<KevQuestionRow, DecisionFailure> {
    let control_tokens = kev_tokenizer.control_tokens;
    let question_text = kev_question_text(question);
    let mut token_ids = vec![control_tokens.question_token_id];
    token_ids.extend(kev_tokenizer.encode_caller_text(&question_text.instructions_text)?);
    let mut option_closing_offsets = Vec::with_capacity(question_text.option_texts.len());
    let mut distinct_option_tokens = HashSet::new();
    for option_text in &question_text.option_texts {
        let option_token_ids = kev_tokenizer.encode_caller_text(option_text)?;
        token_ids.push(control_tokens.option_opening_token_id);
        token_ids.extend(&option_token_ids);
        token_ids.push(control_tokens.option_closing_token_id);
        option_closing_offsets.push(token_ids.len() - 1);
        distinct_option_tokens.insert(option_token_ids);
    }
    token_ids.push(control_tokens.decision_token_id);
    Ok(KevQuestionRow {
        token_ids,
        option_closing_offsets,
        distinguishable_option_count: distinct_option_tokens.len(),
    })
}
