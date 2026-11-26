//! Finds Rust attributes that silence the function-length lint: an `allow` or
//! `expect` naming `clippy::too_many_lines`, or naming `clippy::pedantic`, the
//! lint group it belongs to, since allowing the group allows the lint too.
//!
//! The scan reads text, not syntax. A line inside a multi-line string literal
//! that happens to start with such an attribute is reported as well; a test
//! that needs that text builds it with `concat!` so no line starts with it.

/// Lint paths whose `allow` or `expect` lets a long function through.
const SILENCING_LINT_PATHS: [&str; 2] = ["clippy::too_many_lines", "clippy::pedantic"];

/// The one-based line numbers where an attribute that allows or expects the
/// function-length lint starts. An attribute may span several lines
/// (`#[expect(\n clippy::too_many_lines,\n reason = ..)]`), so the lines of
/// one attribute are read together until its brackets close.
pub(crate) fn exemption_line_numbers(source_text: &str) -> Vec<usize> {
    let source_lines: Vec<&str> = source_text.lines().collect();
    let mut exemption_lines = Vec::new();
    let mut line_index = 0;
    while line_index < source_lines.len() {
        if !is_attribute_start(source_lines[line_index].trim_start()) {
            line_index += 1;
            continue;
        }
        let attribute_end_index = attribute_end_line_index(&source_lines, line_index);
        let attribute_text = source_lines[line_index..=attribute_end_index].join("\n");
        if is_function_length_exemption(&attribute_text) {
            exemption_lines.push(line_index + 1);
        }
        line_index = attribute_end_index + 1;
    }
    exemption_lines
}

fn is_attribute_start(trimmed_line: &str) -> bool {
    trimmed_line.starts_with("#[") || trimmed_line.starts_with("#![")
}

/// The index of the line on which the attribute starting at `start_index`
/// closes its outermost bracket, or the last line when it never closes.
/// Brackets inside string literals and after a `//` comment are not counted,
/// so `#[error("range [{a}, {b})")]` closes on its own line.
fn attribute_end_line_index(source_lines: &[&str], start_index: usize) -> usize {
    let mut open_bracket_depth: i64 = 0;
    for (line_index, source_line) in source_lines.iter().enumerate().skip(start_index) {
        open_bracket_depth += bracket_depth_change(source_line);
        if open_bracket_depth <= 0 {
            return line_index;
        }
    }
    source_lines.len().saturating_sub(1)
}

/// Opening minus closing square brackets on one line, ignoring those inside
/// a `"…"` string literal (with `\"` escapes) and everything after `//`.
fn bracket_depth_change(source_line: &str) -> i64 {
    let mut depth_change = 0;
    let mut is_inside_string = false;
    let mut previous_character = '\0';
    for source_character in source_line.chars() {
        if is_inside_string {
            if source_character == '"' && previous_character != '\\' {
                is_inside_string = false;
            }
        } else {
            match source_character {
                '"' => is_inside_string = true,
                '/' if previous_character == '/' => break,
                '[' => depth_change += 1,
                ']' => depth_change -= 1,
                _ => {}
            }
        }
        previous_character = source_character;
    }
    depth_change
}

/// `allow` and `expect` silence a lint; `warn` and `deny` do not, and a
/// `cfg_attr` wrapping either of the first two silences it in that build.
fn is_function_length_exemption(attribute_text: &str) -> bool {
    names_silencing_lint(attribute_text)
        && (attribute_text.contains("allow(") || attribute_text.contains("expect("))
}

fn names_silencing_lint(attribute_text: &str) -> bool {
    for lint_path in SILENCING_LINT_PATHS {
        if attribute_text.contains(lint_path) {
            return true;
        }
    }
    false
}
