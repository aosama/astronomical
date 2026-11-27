//! No C++ or Metal program text inside a Rust file.
//!
//! Every Metal kernel lives in a `.metal` file and C++ in its own files, so a
//! Rust string literal that holds program statements is code hidden from the
//! compilers, formatters and checks that read those files: it was how kernels
//! were written before issue #2712, by gluing string pieces together. This
//! check lexes every Rust file (comments are dropped by the lexer, strings
//! inside macro calls such as `format!` are kept, doc comments are skipped)
//! and refuses a literal that has both a program statement (a line ending in
//! `;` or `{`, or starting with `}`) and a C++ or Metal word. A line ending
//! in `}` alone is not a statement: Rust format strings end that way
//! (`"… has {actual_dtype}"`). A single word used as a search
//! key, or a sentence that names a Metal concept, is not a statement and
//! passes.

use std::fs;
use std::path::Path;
use std::str::FromStr;

use proc_macro2::{Delimiter, TokenStream, TokenTree};

use super::rust_file_walk::rust_files_under;

/// The folders whose Rust is searched.
const SEARCHED_ROOTS: [&str; 4] = ["apps", "crates", "verification", "xtask"];

/// The one file allowed to hold such text: the port of MLX's
/// `write_signature`, whose whole job is writing the signature lines MLX's
/// own C++ writes, checked byte for byte against MLX by the
/// `metal_kernel_wrapper_parity` contract.
const ALLOWED_FILES: [&str; 1] = ["crates/foundation/metal-kernel-declarations/src/mlx_wrapper.rs"];

/// Words that only C++ or Metal programs use. A statement line with one of
/// them is program text; ordinary Rust strings never have both. A word that
/// ends in a letter or digit must stand alone (`uint3`, not `uint32`).
/// `bfloat16` is not among them: every model configuration names it.
const PROGRAM_WORDS: [&str; 25] = [
    "threadgroup",
    "device ",
    "constant ",
    "simd_",
    "simdgroup",
    "metal::",
    "[[kernel]]",
    "[[buffer",
    "#include",
    "#define",
    "#pragma",
    "template <",
    "template<",
    "std::",
    "namespace ",
    "typedef ",
    "extern \"C\"",
    "constexpr",
    "static_cast<",
    "reinterpret_cast<",
    "uint3",
    "float4",
    "half4",
    "thread_position_in_grid",
    "thread_index_in_",
];

/// Whether a string literal's value is C++ or Metal program text.
pub(crate) fn is_program_text(literal_value: &str) -> bool {
    let mut has_statement_line = false;
    for literal_line in literal_value.lines() {
        has_statement_line |= is_statement_line(literal_line.trim());
    }
    if !has_statement_line {
        return false;
    }
    for program_word in PROGRAM_WORDS {
        if contains_program_word(literal_value, program_word) {
            return true;
        }
    }
    false
}

fn is_statement_line(trimmed_line: &str) -> bool {
    trimmed_line.len() > 1
        && (trimmed_line.ends_with(';')
            || trimmed_line.ends_with('{')
            || trimmed_line.starts_with('}'))
}

/// Whether `program_word` occurs in `text`, standing alone when the word
/// ends in a letter or digit.
fn contains_program_word(text: &str, program_word: &str) -> bool {
    let word_ends_in_name_character = program_word
        .chars()
        .last()
        .is_some_and(|last_character| last_character.is_ascii_alphanumeric());
    for (match_index, _) in text.match_indices(program_word) {
        let following_character = text[match_index + program_word.len()..].chars().next();
        let is_followed_by_name_character = following_character
            .is_some_and(|character| character.is_ascii_alphanumeric() || character == '_');
        if !word_ends_in_name_character || !is_followed_by_name_character {
            return true;
        }
    }
    false
}

/// Every Rust file line, as `path:line`, whose string literal holds program text.
pub(crate) fn embedded_program_text_sites(repository_root: &Path) -> Vec<String> {
    let mut program_text_sites = Vec::new();
    for (relative_path, rust_file) in
        rust_files_under(repository_root, &SEARCHED_ROOTS, &ALLOWED_FILES)
    {
        let Ok(rust_text) = fs::read_to_string(&rust_file) else {
            continue;
        };
        for line_number in program_text_lines(&rust_text) {
            program_text_sites.push(format!("{relative_path}:{line_number}"));
        }
    }
    program_text_sites
}

/// The lines of `rust_text` where a string literal holding program text
/// starts. A file the lexer cannot read is reported at line 1, so a check
/// that cannot look is never silent.
pub(crate) fn program_text_lines(rust_text: &str) -> Vec<usize> {
    let Ok(token_stream) = TokenStream::from_str(rust_text) else {
        return vec![1];
    };
    let mut program_text_lines = Vec::new();
    collect_program_text_lines(token_stream, &mut program_text_lines);
    program_text_lines.sort_unstable();
    program_text_lines.dedup();
    program_text_lines
}

fn collect_program_text_lines(token_stream: TokenStream, program_text_lines: &mut Vec<usize>) {
    for token_tree in token_stream {
        match token_tree {
            TokenTree::Group(token_group) => {
                if !is_doc_attribute(&token_group) {
                    collect_program_text_lines(token_group.stream(), program_text_lines);
                }
            }
            TokenTree::Literal(literal) => {
                let Ok(string_literal) = syn::parse_str::<syn::LitStr>(&literal.to_string()) else {
                    continue;
                };
                if is_program_text(&string_literal.value()) {
                    program_text_lines.push(literal.span().start().line);
                }
            }
            TokenTree::Ident(_) | TokenTree::Punct(_) => {}
        }
    }
}

/// `#[doc = "…"]`, which the lexer makes of a doc comment: prose for
/// readers, which may show code examples.
fn is_doc_attribute(token_group: &proc_macro2::Group) -> bool {
    if token_group.delimiter() != Delimiter::Bracket {
        return false;
    }
    matches!(token_group.stream().into_iter().next(), Some(TokenTree::Ident(attribute_name)) if attribute_name == "doc")
}
