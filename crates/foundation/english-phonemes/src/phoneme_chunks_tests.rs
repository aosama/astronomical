use super::{english_phoneme_chunks, pieces_within, sentences};

const ROMEO_LINES: &str = "But soft, what light through yonder window breaks? It is the east, and Juliet is the sun. \
Arise, fair sun, and kill the envious moon, who is already sick and pale with grief.";

#[test]
fn should_keep_short_text_in_one_chunk() {
    let chunks = english_phoneme_chunks("Good night, good night!", 510);
    assert_eq!(chunks.len(), 1);
}

#[test]
fn should_cut_at_sentence_ends_when_the_limit_is_small() {
    let chunks = english_phoneme_chunks(ROMEO_LINES, 60);
    assert!(chunks.len() > 1);
    for chunk in &chunks {
        assert!(chunk.chars().count() <= 60, "{chunk}");
    }
}

#[test]
fn should_split_sentences_after_end_marks_and_line_breaks() {
    assert_eq!(
        sentences("One. Two!\nThree"),
        ["One.", " Two!", "\n", "Three"]
            .iter()
            .filter(|s| !s.trim().is_empty())
            .map(|s| s.to_string())
            .collect::<Vec<_>>()
    );
}

// Boundary: a run with no comma or space is cut exactly at the limit.
#[test]
fn should_cut_an_unbroken_run_at_the_limit() {
    let pieces = pieces_within("abcdefghij", 4);
    assert_eq!(pieces, ["abcd", "efgh", "ij"]);
}

#[test]
fn should_prefer_a_comma_over_a_space_when_cutting() {
    let pieces = pieces_within("ab, cd ef gh", 9);
    assert_eq!(pieces[0], "ab,");
}

// Negative: no text, no chunks.
#[test]
fn should_return_no_chunks_for_empty_text() {
    assert!(english_phoneme_chunks("", 510).is_empty());
}
