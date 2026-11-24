use super::prepared_text;

#[test]
fn should_capitalise_and_close_a_sentence() {
    assert_eq!(
        prepared_text("good night, good night"),
        "Good night, good night."
    );
}

#[test]
fn should_keep_existing_end_punctuation() {
    assert_eq!(
        prepared_text("wherefore art thou Romeo?"),
        "Wherefore art thou Romeo?"
    );
}

#[test]
fn should_collapse_line_breaks_and_runs_of_spaces() {
    assert_eq!(
        prepared_text("  parting is\n such   sweet\r\nsorrow "),
        "Parting is such sweet sorrow."
    );
}

// Negative: blank text prepares to nothing.
#[test]
fn should_prepare_blank_text_to_nothing() {
    assert_eq!(prepared_text(" \n "), "");
}

use super::{TokenizedSentence, packed_chunk_texts};

fn sentence(token_count: usize, text: &str) -> TokenizedSentence {
    TokenizedSentence {
        token_count,
        sentence: text.to_owned(),
    }
}

#[test]
fn should_join_sentences_while_they_fit() {
    let packed = packed_chunk_texts(
        vec![sentence(20, "But soft."), sentence(30, "It is the east.")],
        50,
    );
    assert_eq!(packed, ["But soft. It is the east."]);
}

// Boundary: one token over the limit starts a new chunk.
#[test]
fn should_start_a_new_chunk_one_token_over_the_limit() {
    let packed = packed_chunk_texts(
        vec![sentence(20, "But soft."), sentence(31, "It is the east.")],
        50,
    );
    assert_eq!(packed, ["But soft.", "It is the east."]);
}

// Edge: a sentence longer than the limit stands alone rather than being cut.
#[test]
fn should_keep_an_overlong_sentence_whole() {
    let packed = packed_chunk_texts(
        vec![sentence(80, "Arise, fair sun."), sentence(5, "Good night.")],
        50,
    );
    assert_eq!(packed, ["Arise, fair sun.", "Good night."]);
}

#[test]
fn should_pack_nothing_from_no_sentences() {
    assert!(packed_chunk_texts(Vec::new(), 50).is_empty());
}
