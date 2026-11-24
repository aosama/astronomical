//! The function-name alphabet every tool surface accepts.

/// ASCII letters, digits, `_`, and `-`: the Anthropic `^[a-zA-Z0-9_-]`
/// tool-name rule, and the name characters the worker's tool-call parser
/// round-trips.
pub(crate) fn is_valid_function_name(function_name: &str) -> bool {
    if function_name.is_empty() {
        return false;
    }
    for name_byte in function_name.bytes() {
        if !(name_byte.is_ascii_alphanumeric() || matches!(name_byte, b'_' | b'-')) {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::is_valid_function_name;

    #[test]
    fn should_accept_letters_digits_underscores_and_hyphens() {
        assert!(is_valid_function_name("mcp__server__read-file_2"));
    }

    #[test]
    fn should_accept_a_single_character_name() {
        assert!(is_valid_function_name("r"));
    }

    #[test]
    fn should_reject_empty_names_spaces_and_punctuation() {
        assert!(!is_valid_function_name(""));
        assert!(!is_valid_function_name("read file"));
        assert!(!is_valid_function_name("read.file"));
    }
}
