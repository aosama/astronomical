//! The report identifiers of the operation and counter catalogs. Serialized
//! reports are keyed by these strings, so each must be unique and spelled the
//! same way as every other.

use std::collections::HashSet;

use super::*;

/// Lowercase words joined by single underscores. Digits may follow the first
/// letter, as in `jangtq2_codebook_page_decode`.
fn is_snake_case(identifier: &str) -> bool {
    let Some(first_character) = identifier.chars().next() else {
        return false;
    };
    first_character.is_ascii_lowercase()
        && !identifier.ends_with('_')
        && !identifier.contains("__")
        && identifier.chars().all(|character| {
            character.is_ascii_lowercase() || character.is_ascii_digit() || character == '_'
        })
}

#[test]
fn should_accept_lowercase_words_joined_by_single_underscores() {
    assert!(is_snake_case("prompt_token_count"));
    assert!(is_snake_case("jangtq2_activation_rotation_hadamard32"));
    assert!(is_snake_case("a"));
}

#[test]
fn should_reject_identifiers_that_are_not_snake_case() {
    for malformed_identifier in [
        "",
        "PromptTokenCount",
        "prompt-token-count",
        "prompt token count",
        "_prompt_token_count",
        "prompt_token_count_",
        "prompt__token_count",
        "2nd_prompt_token_count",
    ] {
        assert!(
            !is_snake_case(malformed_identifier),
            "{malformed_identifier:?} should not read as snake case"
        );
    }
}

#[test]
fn should_give_every_operation_a_unique_report_identifier() {
    let mut catalogued_identifiers = HashSet::new();

    for performance_operation in PerformanceOperation::ALL {
        let identifier = performance_operation.identifier();
        assert!(
            catalogued_identifiers.insert(identifier),
            "duplicate operation identifier {identifier}"
        );
    }

    assert_eq!(
        catalogued_identifiers.len(),
        PerformanceOperation::COUNT,
        "the identifier set must cover exactly the reserved operation storage"
    );
}

#[test]
fn should_write_every_operation_identifier_in_snake_case() {
    for performance_operation in PerformanceOperation::ALL {
        let identifier = performance_operation.identifier();
        assert!(
            is_snake_case(identifier),
            "operation identifier {identifier:?} should be snake case"
        );
    }
}

#[test]
fn should_write_every_counter_identifier_in_snake_case() {
    for performance_counter in PerformanceCounter::ALL {
        let identifier = performance_counter.identifier();
        assert!(
            is_snake_case(identifier),
            "counter identifier {identifier:?} should be snake case"
        );
    }
}

// An operation and a counter never share a report section, but the same
// string on both would still make a reader of the raw JSON assume one meaning.
#[test]
fn should_keep_operation_and_counter_identifiers_distinct_from_each_other() {
    let operation_identifiers: HashSet<&str> = PerformanceOperation::ALL
        .into_iter()
        .map(PerformanceOperation::identifier)
        .collect();

    for performance_counter in PerformanceCounter::ALL {
        let identifier = performance_counter.identifier();
        assert!(
            !operation_identifiers.contains(identifier),
            "{identifier} names both an operation and a counter"
        );
    }
}
