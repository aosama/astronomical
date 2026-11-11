use super::{TextTokenKind, text_tokens};

fn token_texts(text: &str) -> Vec<String> {
    text_tokens(text)
        .into_iter()
        .map(|token| token.text)
        .collect()
}

#[test]
fn should_split_words_numbers_and_punctuation() {
    assert_eq!(
        token_texts("Romeo, it's 2026!"),
        ["Romeo", ",", "it's", "2026", "!"]
    );
}

#[test]
fn should_record_which_tokens_a_space_follows() {
    let tokens = text_tokens("But soft, what light");
    let spaced: Vec<bool> = tokens
        .iter()
        .map(|token| token.is_followed_by_space)
        .collect();
    assert_eq!(spaced, [true, false, true, true, false]);
}

#[test]
fn should_keep_dotted_abbreviations_whole() {
    assert_eq!(token_texts("the U.S. army"), ["the", "U.S.", "army"]);
}

// Edge: a dot between two words is a sentence end, not an abbreviation.
#[test]
fn should_not_join_two_words_across_a_dot() {
    assert_eq!(token_texts("end.Next"), ["end", ".", "Next"]);
}

#[test]
fn should_turn_a_spaced_dash_into_a_pause_and_drop_a_joining_hyphen() {
    assert_eq!(token_texts("wait - now"), ["wait", "—", "now"]);
    assert_eq!(token_texts("well-known"), ["well", "known"]);
}

#[test]
fn should_drop_characters_without_a_sound() {
    assert_eq!(
        token_texts("**bold** `code` /path"),
        ["bold", "code", "path"]
    );
}

#[test]
fn should_classify_money_and_negative_numbers() {
    let tokens = text_tokens("$5 and -3");
    let kinds: Vec<TextTokenKind> = tokens.into_iter().map(|token| token.kind).collect();
    assert_eq!(
        kinds,
        [
            TextTokenKind::Symbol,
            TextTokenKind::Number,
            TextTokenKind::Word,
            TextTokenKind::Number
        ]
    );
}

#[test]
fn should_return_nothing_for_empty_text() {
    assert!(text_tokens("   ").is_empty());
}
