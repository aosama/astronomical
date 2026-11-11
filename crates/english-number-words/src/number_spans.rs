//! Where a number begins and ends in running text.

/// The endings a number may carry and still be read as a number.
const NUMBER_ENDINGS: [&str; 6] = ["st", "nd", "rd", "th", "s", "'s"];

/// Whether a minus sign at `position` starts a negative number: it is
/// followed by a digit and stands at the start of a word.
pub fn starts_signed_number(characters: &[char], position: usize) -> bool {
    characters[position] == '-'
        && characters
            .get(position + 1)
            .is_some_and(char::is_ascii_digit)
        && (position == 0 || characters[position - 1].is_whitespace())
}

/// The end of the number starting at `start`: digits with commas or points
/// between them, then one ending ("st", "s", ...) that is not the start of a
/// longer word.
pub fn number_span_end(characters: &[char], start: usize) -> usize {
    let mut digits_end = start + 1;
    while digits_end < characters.len() {
        let character = characters[digits_end];
        let is_separator_between_digits = (character == ',' || character == '.')
            && characters
                .get(digits_end + 1)
                .is_some_and(char::is_ascii_digit);
        if character.is_ascii_digit() || is_separator_between_digits {
            digits_end += 1;
        } else {
            break;
        }
    }
    digits_end + ending_length(characters, digits_end)
}

fn ending_length(characters: &[char], digits_end: usize) -> usize {
    for ending in NUMBER_ENDINGS {
        let ending_characters: Vec<char> = ending.chars().collect();
        let after_ending = digits_end + ending_characters.len();
        let is_whole_ending = characters[digits_end..].starts_with(&ending_characters)
            && !characters
                .get(after_ending)
                .is_some_and(|next| next.is_alphabetic());
        if is_whole_ending {
            return ending_characters.len();
        }
    }
    0
}

#[cfg(test)]
#[path = "number_spans_tests.rs"]
mod tests;
