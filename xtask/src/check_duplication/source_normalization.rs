//! Turns a Rust source file into the code lines the duplication measure
//! compares: blank lines and `//` comment lines dropped, whitespace trimmed,
//! and model-family names masked.
//!
//! Masking exists so two copies that differ only by the family they serve
//! (`KevEngine` against `ClefEngine`, `KokoroSpeechFamily` against
//! `PhononSpeechFamily`) are counted as the copies they are. Without it the
//! measure would miss most of the duplication issue #2473 sets out to remove.

/// What every family spelling below is replaced with.
pub(crate) const FAMILY_NAME_MASK: &str = "FAMILY";

/// Family spellings, in CamelCase, snake_case and the SCREAMING_SNAKE form
/// the image families use for constants. The order is the match order at each
/// position, so a compound such as `Qwen3_5Text` must come before its
/// parts: it then becomes one mask, as it did in the measurement the issue's
/// figures come from, rather than two masks side by side.
pub(crate) const FAMILY_NAME_SPELLINGS: [&str; 24] = [
    "Qwen3_5Text",
    "qwen3_5_text",
    "Streaming",
    "streaming",
    "Flux2Klein",
    "QwenImage21",
    "flux2_klein",
    "qwen_image_21",
    "FLUX2_KLEIN",
    "QWEN_IMAGE_21",
    "Kokoro",
    "Phonon",
    "kokoro",
    "phonon",
    "Kev",
    "Clef",
    "Laya",
    "kev",
    "clef",
    "laya",
    "PrismHadamardQwen35",
    "prism_hadamard_qwen35",
    "Qwen3_5",
    "qwen3_5",
];

/// A line shorter than this, in characters, carries too little to call two
/// copies of it duplication (`Ok(())`, `} else {`, `.await?;`).
const MINIMUM_SUBSTANTIAL_LINE_CHARACTERS: usize = 13;

/// The code lines of one source file, in order, ready to compare.
pub(crate) fn normalize_code_lines(source_text: &str) -> Vec<String> {
    let mut code_lines = Vec::new();
    for source_line in source_text.lines() {
        let trimmed_line = source_line.trim();
        if trimmed_line.is_empty() || trimmed_line.starts_with("//") {
            continue;
        }
        code_lines.push(mask_family_names(trimmed_line));
    }
    code_lines
}

/// Replaces every family spelling in `code_line` with [`FAMILY_NAME_MASK`],
/// scanning left to right and taking the first spelling in list order that
/// matches at each position.
pub(crate) fn mask_family_names(code_line: &str) -> String {
    let mut masked_line = String::with_capacity(code_line.len());
    let mut unscanned_text = code_line;
    while let Some(next_character) = unscanned_text.chars().next() {
        match matching_family_spelling(unscanned_text) {
            Some(family_spelling) => {
                masked_line.push_str(FAMILY_NAME_MASK);
                unscanned_text = &unscanned_text[family_spelling.len()..];
            }
            None => {
                masked_line.push(next_character);
                unscanned_text = &unscanned_text[next_character.len_utf8()..];
            }
        }
    }
    masked_line
}

/// The family spelling `unscanned_text` starts with, if any. Every spelling
/// starts with one of a dozen ASCII letters, so the first byte rules out most
/// positions before any spelling is compared; the debug build the hooks run
/// would otherwise spend seconds on comparisons that cannot match.
fn matching_family_spelling(unscanned_text: &str) -> Option<&'static str> {
    let first_byte = *unscanned_text.as_bytes().first()?;
    if !b"QqRrSsFfKkPpCcLl".contains(&first_byte) {
        return None;
    }
    // The closure is the form clippy's `manual_find` lint asks for here.
    FAMILY_NAME_SPELLINGS
        .into_iter()
        .find(|family_spelling| unscanned_text.starts_with(family_spelling))
}

/// Whether a normalized line says enough to count toward a repeated block:
/// long enough, and more than closing brackets and separators.
pub(crate) fn is_substantial_code_line(code_line: &str) -> bool {
    !is_only_brackets_and_separators(code_line)
        && code_line.chars().count() >= MINIMUM_SUBSTANTIAL_LINE_CHARACTERS
}

fn is_only_brackets_and_separators(code_line: &str) -> bool {
    for line_character in code_line.chars() {
        if !line_character.is_whitespace() && !"()[]{};,".contains(line_character) {
            return false;
        }
    }
    true
}
