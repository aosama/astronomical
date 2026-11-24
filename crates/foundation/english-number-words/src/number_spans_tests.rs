use super::{number_span_end, starts_signed_number};

fn characters(text: &str) -> Vec<char> {
    text.chars().collect()
}

#[test]
fn should_span_digits_with_separators_between_them() {
    let text = characters("1,234.5 apples");
    assert_eq!(number_span_end(&text, 0), 7);
}

#[test]
fn should_include_an_ending_but_not_a_longer_word() {
    assert_eq!(number_span_end(&characters("21st"), 0), 4);
    assert_eq!(number_span_end(&characters("1990s"), 0), 5);
    assert_eq!(number_span_end(&characters("3rdly"), 0), 1);
}

// Edge: a trailing separator is punctuation, not part of the number.
#[test]
fn should_stop_before_a_trailing_comma_or_point() {
    assert_eq!(number_span_end(&characters("2026."), 0), 4);
    assert_eq!(number_span_end(&characters("7, then"), 0), 1);
}

#[test]
fn should_start_a_negative_number_only_at_a_word_start() {
    assert!(starts_signed_number(&characters("-3"), 0));
    assert!(starts_signed_number(&characters("a -3"), 2));
    assert!(!starts_signed_number(&characters("a-3"), 1));
    assert!(!starts_signed_number(&characters("- 3"), 0));
}
