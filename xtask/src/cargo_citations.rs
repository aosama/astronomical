//! `cargo xtask check-cargo-citations`: every cargo command cited in tracked
//! Markdown must be a stock cargo subcommand or an alias defined in
//! `.cargo/config.toml`, so documentation cannot drift onto retired command
//! names the way the release procedure once did (issue #1248).
//!
//! Typed port of the former
//! `verification/tooling_contracts/test-cargo-citation-contract.sh` (issue
//! #2635), keeping its repository-sweep output lines and its failure exit.
//! Only fenced code blocks and inline backtick spans are citations: prose such
//! as "cargo runs the build" is not a command anyone would copy.
//!
//! A citation may name a family of aliases instead of one: `cargo measure-...`
//! or `cargo measure-*` stands for "any alias that starts with `measure-`",
//! and holds while at least one alias does. The shell check could not express
//! that and read the placeholder as a command called `measure`.

use std::path::Path;
use std::process::Command;

const CARGO_CONFIG_PATH: &str = ".cargo/config.toml";
const CITATION_LEAD: &str = "cargo ";
const FENCE_MARKER: &str = "```";
/// What may follow a cited word to make it a prefix: three dots, the
/// typographic ellipsis an editor may substitute for them, or a glob star.
const PREFIX_PLACEHOLDERS: [&str; 3] = ["...", "…", "*"];

/// Stock subcommands, and the plugins this repository's pinned toolchain
/// provides (cargo-about through install-verification-tools, clippy through
/// rustup).
const CITABLE_STOCK_COMMANDS: [&str; 27] = [
    "build",
    "check",
    "clean",
    "doc",
    "fetch",
    "fmt",
    "help",
    "init",
    "install",
    "login",
    "logout",
    "metadata",
    "new",
    "owner",
    "package",
    "publish",
    "report",
    "run",
    "rustc",
    "rustdoc",
    "search",
    "test",
    "tree",
    "update",
    "version",
    "yank",
    "verify-project",
];
const CITABLE_PLUGIN_COMMANDS: [&str; 2] = ["about", "clippy"];

/// One cargo command cited in Markdown.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum CitedCommand {
    /// `cargo build`: one command, which must exist.
    Exact(String),
    /// `cargo measure-...`: any alias starting with this text.
    AliasFamily(String),
}

pub fn run(repository_root: &Path, arguments: &[String]) -> Result<(), String> {
    if !arguments.is_empty() {
        return Err("usage: cargo xtask check-cargo-citations".to_owned());
    }
    let config_text = std::fs::read_to_string(repository_root.join(CARGO_CONFIG_PATH))
        .map_err(|read_error| format!("cannot read {CARGO_CONFIG_PATH}: {read_error}"))?;
    let alias_names = alias_names(&config_text);
    println!("[cargo-citation-contract] phase=repository-sweep status=start");
    let mut citation_count = 0;
    let mut violations = Vec::new();
    for markdown_path in tracked_markdown_paths(repository_root)? {
        let markdown_text = std::fs::read_to_string(repository_root.join(&markdown_path))
            .map_err(|read_error| format!("cannot read {markdown_path}: {read_error}"))?;
        for cited_command in cited_commands(&markdown_text) {
            citation_count += 1;
            if !is_citable(&cited_command, &alias_names) {
                violations.push(violation_line(&markdown_path, &cited_command));
            }
        }
    }
    if !violations.is_empty() {
        eprintln!(
            "Error: {} markdown cargo citation(s) do not resolve:",
            violations.len()
        );
        for violation in &violations {
            eprintln!("{violation}");
        }
        return Err("[cargo-citation-contract] phase=repository-sweep status=failed".to_owned());
    }
    println!(
        "[cargo-citation-contract] phase=repository-sweep status=success citations={citation_count}"
    );
    Ok(())
}

/// Every key of the `[alias]` table. The aliases are one string each (the
/// nested-config contract in `xtask/tests` says why), so each key line reads
/// `name = "..."`.
pub(crate) fn alias_names(config_text: &str) -> Vec<String> {
    let mut is_inside_alias_table = false;
    let mut found_alias_names = Vec::new();
    for config_line in config_text.lines() {
        if config_line.starts_with('[') {
            is_inside_alias_table = config_line.trim_end() == "[alias]";
            continue;
        }
        if !is_inside_alias_table {
            continue;
        }
        if let Some((alias_name, _expansion)) = config_line.split_once(" = \"")
            && is_command_word(alias_name)
        {
            found_alias_names.push(alias_name.to_owned());
        }
    }
    found_alias_names
}

fn is_command_word(candidate_word: &str) -> bool {
    starts_with_lowercase_letter(candidate_word)
        && candidate_word.chars().all(is_command_word_character)
}

fn starts_with_lowercase_letter(text: &str) -> bool {
    matches!(text.chars().next(), Some('a'..='z'))
}

fn is_command_word_character(word_character: char) -> bool {
    word_character.is_ascii_lowercase() || word_character.is_ascii_digit() || word_character == '-'
}

fn ends_a_command_word(word_character: char) -> bool {
    !is_command_word_character(word_character)
}

/// Every cargo command cited in a fenced code block or an inline backtick
/// span, in document order. A fence opens and closes only on a line that
/// starts with three backticks, as the shell check read it.
pub(crate) fn cited_commands(markdown_text: &str) -> Vec<CitedCommand> {
    let mut is_inside_fence = false;
    let mut found_citations = Vec::new();
    for markdown_line in markdown_text.lines() {
        if markdown_line.starts_with(FENCE_MARKER) {
            is_inside_fence = !is_inside_fence;
            continue;
        }
        if is_inside_fence {
            found_citations.extend(citations_in_code(markdown_line));
            continue;
        }
        for code_span in inline_code_spans(markdown_line) {
            found_citations.extend(citations_in_code(code_span));
        }
    }
    found_citations
}

/// The text inside each single-backtick span of one line; an unclosed
/// backtick opens no span.
pub(crate) fn inline_code_spans(markdown_line: &str) -> Vec<&str> {
    let mut found_spans = Vec::new();
    let mut remaining_line = markdown_line;
    while let Some(span_open) = remaining_line.find('`') {
        let after_open = &remaining_line[span_open + 1..];
        let Some(span_length) = after_open.find('`') else {
            break;
        };
        if span_length > 0 {
            found_spans.push(&after_open[..span_length]);
        }
        remaining_line = &after_open[span_length + 1..];
    }
    found_spans
}

/// Each `cargo WORD` in a piece of code. `cargo` must start a word, so
/// `xcargo build` or `~/.cargo build` cites nothing.
pub(crate) fn citations_in_code(code_text: &str) -> Vec<CitedCommand> {
    let mut found_citations = Vec::new();
    let mut search_offset = 0;
    while let Some(lead_position) = code_text[search_offset..].find(CITATION_LEAD) {
        let lead_start = search_offset + lead_position;
        let word_start = lead_start + CITATION_LEAD.len();
        search_offset = word_start;
        if starts_a_word(code_text, lead_start)
            && let Some(cited_command) = citation_at(&code_text[word_start..])
        {
            found_citations.push(cited_command);
        }
    }
    found_citations
}

fn starts_a_word(code_text: &str, lead_start: usize) -> bool {
    match code_text[..lead_start].chars().next_back() {
        None => true,
        Some(preceding) => {
            !(preceding.is_ascii_alphanumeric() || matches!(preceding, '-' | '_' | '.' | '/'))
        }
    }
}

/// The command a citation names, read from the text after `cargo `. A word
/// followed by a placeholder is an alias family; otherwise trailing hyphens
/// are dropped, and a word shorter than two letters (a flag's value, a
/// version) is not a command.
fn citation_at(text_after_lead: &str) -> Option<CitedCommand> {
    if !starts_with_lowercase_letter(text_after_lead) {
        return None;
    }
    let word_length = text_after_lead
        .find(ends_a_command_word)
        .unwrap_or(text_after_lead.len());
    let cited_word = &text_after_lead[..word_length];
    for placeholder in PREFIX_PLACEHOLDERS {
        if text_after_lead[word_length..].starts_with(placeholder) {
            return Some(CitedCommand::AliasFamily(cited_word.to_owned()));
        }
    }
    let exact_word = cited_word.trim_end_matches('-');
    if exact_word.len() < 2 {
        return None;
    }
    Some(CitedCommand::Exact(exact_word.to_owned()))
}

/// An exact citation must name a stock command or an alias; a family must
/// match at least one alias, or it describes commands that do not exist.
pub(crate) fn is_citable(cited_command: &CitedCommand, alias_names: &[String]) -> bool {
    match cited_command {
        CitedCommand::Exact(command_word) => {
            CITABLE_STOCK_COMMANDS.contains(&command_word.as_str())
                || CITABLE_PLUGIN_COMMANDS.contains(&command_word.as_str())
                || alias_names.contains(command_word)
        }
        CitedCommand::AliasFamily(alias_prefix) => {
            for alias_name in alias_names {
                if alias_name.starts_with(alias_prefix.as_str()) {
                    return true;
                }
            }
            false
        }
    }
}

pub(crate) fn violation_line(markdown_path: &str, cited_command: &CitedCommand) -> String {
    match cited_command {
        CitedCommand::Exact(command_word) => {
            format!("{markdown_path} cites unknown cargo command \"{command_word}\"")
        }
        CitedCommand::AliasFamily(alias_prefix) => format!(
            "{markdown_path} cites cargo alias family \"{alias_prefix}...\" but no alias starts with \"{alias_prefix}\""
        ),
    }
}

fn tracked_markdown_paths(repository_root: &Path) -> Result<Vec<String>, String> {
    let ls_files_output = Command::new("git")
        .arg("-C")
        .arg(repository_root)
        .args(["ls-files", "*.md"])
        .output()
        .map_err(|launch_error| format!("cannot run git ls-files: {launch_error}"))?;
    if !ls_files_output.status.success() {
        return Err(format!(
            "git ls-files failed: {}",
            String::from_utf8_lossy(&ls_files_output.stderr).trim()
        ));
    }
    let mut markdown_paths: Vec<String> = String::from_utf8_lossy(&ls_files_output.stdout)
        .lines()
        .map(str::to_owned)
        .collect();
    markdown_paths.sort();
    Ok(markdown_paths)
}
