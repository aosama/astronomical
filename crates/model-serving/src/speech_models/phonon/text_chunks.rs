//! Text into the token chunks Phonon reads, as the reference runtime
//! prepares them: whitespace collapsed, the first letter capitalised, a full
//! stop added when the text ends in a letter or digit, and long text split at
//! sentence ends into chunks of at most 50 tokens.
//!
//! The chunk limit is the model's own: a chunk much longer than the speech it
//! learnt to say in one go makes it hurry and skip words.

use super::text_tokenizer::PhononTokenizer;
use crate::speech_models::speech_model_error::SpeechModelError;

/// The most tokens one chunk is prompted with.
pub(crate) const MAXIMUM_TOKENS_PER_CHUNK: usize = 50;
/// Marks after which a sentence ends.
const SENTENCE_ENDS: [&str; 4] = [".", "!", "...", "?"];
/// Texts of this many words or fewer get three extra frames after the end of
/// speech instead of one, so a short reply is not clipped.
const SHORT_TEXT_WORD_LIMIT: usize = 4;

/// One chunk ready to prompt with.
#[derive(Debug, Eq, PartialEq)]
pub(crate) struct TextChunk {
    pub(crate) token_ids: Vec<i32>,
    /// Frames to keep generating after the model signals the end of speech.
    pub(crate) frames_after_end: usize,
}

/// One sentence and how many tokens it takes.
struct TokenizedSentence {
    token_count: usize,
    sentence: String,
}

/// `text` as prompt chunks, sentences kept together while they fit.
pub(crate) fn prompt_chunks(
    tokenizer: &PhononTokenizer,
    text: &str,
) -> Result<Vec<TextChunk>, SpeechModelError> {
    let prepared = prepared_text(text);
    if prepared.is_empty() {
        return Ok(Vec::new());
    }
    let tokenized_sentences = sentences(tokenizer, &prepared)?;
    let chunk_texts = packed_chunk_texts(tokenized_sentences, MAXIMUM_TOKENS_PER_CHUNK);
    chunk_texts
        .iter()
        .map(|chunk_text| prompt_chunk(tokenizer, chunk_text))
        .collect()
}

/// Sentences joined greedily while the joined token count stays within
/// `maximum_tokens`; a sentence longer than that stands alone.
fn packed_chunk_texts(
    tokenized_sentences: Vec<TokenizedSentence>,
    maximum_tokens: usize,
) -> Vec<String> {
    let mut chunk_texts: Vec<String> = Vec::new();
    let mut current_text = String::new();
    let mut current_token_count = 0;
    for TokenizedSentence {
        token_count,
        sentence,
    } in tokenized_sentences
    {
        let would_overflow = current_token_count + token_count > maximum_tokens;
        if !current_text.is_empty() && would_overflow {
            chunk_texts.push(std::mem::take(&mut current_text));
            current_token_count = 0;
        }
        if !current_text.is_empty() {
            current_text.push(' ');
        }
        current_text.push_str(sentence.trim());
        current_token_count += token_count;
    }
    if !current_text.is_empty() {
        chunk_texts.push(current_text);
    }
    chunk_texts
}

fn prompt_chunk(
    tokenizer: &PhononTokenizer,
    chunk_text: &str,
) -> Result<TextChunk, SpeechModelError> {
    let prepared = prepared_text(chunk_text);
    let is_short_text = prepared.split_whitespace().count() <= SHORT_TEXT_WORD_LIMIT;
    let token_ids = tokenizer
        .encode(&prepared)?
        .into_iter()
        .map(|token_id| token_id as i32)
        .collect();
    Ok(TextChunk {
        token_ids,
        frames_after_end: if is_short_text { 3 } else { 1 },
    })
}

/// A sentence ends after a run of end-mark tokens.
fn sentences(
    tokenizer: &PhononTokenizer,
    prepared: &str,
) -> Result<Vec<TokenizedSentence>, SpeechModelError> {
    let token_ids = tokenizer.encode(prepared)?;
    let mut sentence_starts = vec![0];
    let mut did_previous_end_sentence = false;
    for (token_index, token_id) in token_ids.iter().enumerate() {
        let does_end_sentence = SENTENCE_ENDS.contains(&tokenizer.decode(&[*token_id])?.trim());
        if !does_end_sentence && did_previous_end_sentence {
            sentence_starts.push(token_index);
        }
        did_previous_end_sentence = does_end_sentence;
    }
    sentence_starts.push(token_ids.len());
    let mut tokenized_sentences = Vec::with_capacity(sentence_starts.len());
    for sentence_bounds in sentence_starts.windows(2) {
        let sentence_tokens = &token_ids[sentence_bounds[0]..sentence_bounds[1]];
        tokenized_sentences.push(TokenizedSentence {
            token_count: sentence_tokens.len(),
            sentence: tokenizer.decode(sentence_tokens)?,
        });
    }
    Ok(tokenized_sentences)
}

/// Whitespace collapsed, first letter capitalised, and a full stop added
/// after a final letter or digit.
pub(crate) fn prepared_text(text: &str) -> String {
    let collapsed = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut characters = collapsed.chars();
    let mut prepared = match characters.next() {
        Some(first_character) => first_character
            .to_uppercase()
            .chain(characters)
            .collect::<String>(),
        None => return String::new(),
    };
    if prepared.chars().last().is_some_and(char::is_alphanumeric) {
        prepared.push('.');
    }
    prepared
}

#[cfg(test)]
#[path = "text_chunks_tests.rs"]
mod tests;
