use std::fs;
use std::path::Path;

use super::embedded_program_text::{
    embedded_program_text_sites, is_program_text, program_text_lines,
};

const METAL_KERNEL_BODY: &str = include_str!("program_text_fixtures/metal_kernel_body.txt");
const CPP_HELPER: &str = include_str!("program_text_fixtures/cpp_helper.txt");
const RUST_FILE_WITH_PROGRAM_TEXT: &str =
    include_str!("program_text_fixtures/rust_file_with_program_text.txt");

fn write_file(file_path: &Path, file_text: &str) {
    fs::create_dir_all(file_path.parent().expect("a parent")).expect("the folder is created");
    fs::write(file_path, file_text).expect("the file is written");
}

#[test]
fn should_recognize_a_metal_kernel_body_and_a_cpp_helper() {
    assert!(is_program_text(METAL_KERNEL_BODY));
    assert!(is_program_text(CPP_HELPER));
}

#[test]
fn should_pass_a_sentence_that_names_a_metal_word_without_a_statement() {
    assert!(!is_program_text("the threadgroup holds 256 threads"));
    assert!(!is_program_text("simd_max merges the lanes' lists"));
}

#[test]
fn should_pass_a_statement_without_any_program_word() {
    assert!(!is_program_text("row {}; column {}"));
    assert!(!is_program_text("Y[0] = X[0];"));
}

#[test]
fn should_pass_an_empty_string_and_a_lone_brace() {
    assert!(!is_program_text(""));
    assert!(!is_program_text("}"));
}

#[test]
fn should_find_program_text_in_raw_strings_and_macro_calls_but_not_in_comments_or_docs() {
    assert_eq!(program_text_lines(RUST_FILE_WITH_PROGRAM_TEXT), [7, 15]);
}

#[test]
fn should_report_a_file_the_lexer_cannot_read_at_its_first_line() {
    assert_eq!(program_text_lines("fn broken( {"), [1]);
}

#[test]
fn should_name_each_site_and_skip_the_allowed_file_and_build_output() {
    let repository = tempfile::tempdir().expect("a temporary folder");
    write_file(
        &repository.path().join("crates/a/src/kernel.rs"),
        RUST_FILE_WITH_PROGRAM_TEXT,
    );
    write_file(
        &repository
            .path()
            .join("crates/foundation/metal-kernel-declarations/src/mlx_wrapper.rs"),
        RUST_FILE_WITH_PROGRAM_TEXT,
    );
    write_file(
        &repository.path().join("crates/a/target/debug/generated.rs"),
        RUST_FILE_WITH_PROGRAM_TEXT,
    );
    write_file(
        &repository.path().join("xtask/src/clean.rs"),
        "fn clean() {}\n",
    );
    assert_eq!(
        embedded_program_text_sites(repository.path()),
        ["crates/a/src/kernel.rs:7", "crates/a/src/kernel.rs:15"]
    );
}

#[test]
fn should_pass_an_error_message_that_ends_in_a_format_placeholder() {
    assert!(!is_program_text(
        "packed code words must use uint32, got {packed_dtype:?}"
    ));
    assert!(!is_program_text(
        "must use float16, bfloat16, or float32 but has {actual_dtype}"
    ));
}

#[test]
fn should_count_a_short_program_word_only_when_it_stands_alone() {
    assert!(!is_program_text("the uint32 index; "));
    assert!(is_program_text(include_str!(
        "program_text_fixtures/standalone_uint3_statement.txt"
    )));
}
