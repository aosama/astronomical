//! Long text cut into pieces a speech model can read in one pass (Kokoro
//! reads at most 510 phonemes; its engine passes that limit). Cutting at sentence ends
//! keeps each piece's intonation whole; a sentence that is itself too long is
//! cut after a comma, and failing that between words.

use crate::phonemizer::english_phonemes;

/// `text` as phoneme strings of at most `maximum_phonemes_per_chunk`
/// characters each, in reading order, sentences kept together where they fit.
pub fn english_phoneme_chunks(text: &str, maximum_phonemes_per_chunk: usize) -> Vec<String> {
    let mut phoneme_chunks: Vec<String> = Vec::new();
    let mut current_chunk = String::new();
    for sentence in sentences(text) {
        let sentence_phonemes = english_phonemes(&sentence);
        if sentence_phonemes.is_empty() {
            continue;
        }
        for sentence_piece in pieces_within(&sentence_phonemes, maximum_phonemes_per_chunk) {
            let joined_length = current_chunk.chars().count() + 1 + sentence_piece.chars().count();
            if !current_chunk.is_empty() && joined_length > maximum_phonemes_per_chunk {
                phoneme_chunks.push(std::mem::take(&mut current_chunk));
            }
            if !current_chunk.is_empty() {
                current_chunk.push(' ');
            }
            current_chunk.push_str(&sentence_piece);
        }
    }
    if !current_chunk.is_empty() {
        phoneme_chunks.push(current_chunk);
    }
    phoneme_chunks
}

/// Sentences of `text`, each ending after ".", "!", "?" or a line break.
fn sentences(text: &str) -> Vec<String> {
    let mut found_sentences = Vec::new();
    let mut current_sentence = String::new();
    let mut characters = text.chars().peekable();
    while let Some(character) = characters.next() {
        current_sentence.push(character);
        let does_end_sentence =
            ".!?".contains(character) && characters.peek().is_none_or(|next| next.is_whitespace());
        if does_end_sentence || character == '\n' {
            found_sentences.push(std::mem::take(&mut current_sentence));
        }
    }
    found_sentences.push(current_sentence);
    found_sentences.retain(|sentence| !sentence.trim().is_empty());
    found_sentences
}

/// One sentence's phonemes cut to fit: after the last comma that fits, else
/// at the last space that fits, else (one unbroken run) at the limit itself.
fn pieces_within(phonemes: &str, maximum_phonemes_per_chunk: usize) -> Vec<String> {
    let mut remaining: Vec<char> = phonemes.chars().collect();
    let mut pieces = Vec::new();
    while remaining.len() > maximum_phonemes_per_chunk {
        let fitting_phonemes = &remaining[..maximum_phonemes_per_chunk];
        let cut_after = fitting_phonemes
            .iter()
            .rposition(|sound| *sound == ',')
            .map(|comma_position| comma_position + 1)
            .or_else(|| fitting_phonemes.iter().rposition(|sound| *sound == ' '))
            .filter(|cut_position| *cut_position > 0)
            .unwrap_or(maximum_phonemes_per_chunk);
        let piece: String = remaining[..cut_after].iter().collect();
        pieces.push(piece.trim().to_owned());
        remaining = remaining[cut_after..]
            .iter()
            .copied()
            .skip_while(|sound| *sound == ' ')
            .collect();
    }
    if !remaining.is_empty() {
        pieces.push(remaining.into_iter().collect());
    }
    pieces
}

#[cfg(test)]
#[path = "phoneme_chunks_tests.rs"]
mod tests;
