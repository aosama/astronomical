//! The Astronomical skill (`skill/astronomical`) is shipped to people and read
//! by their agents, which run the commands it names. These tests tie the skill
//! to this CLI: a command, subcommand or flag the skill names must exist in the
//! help text, so a renamed or removed flag fails here instead of sending an
//! agent after something the installed release does not have (issue #2397).
//! A test cannot notice a new ability the skill never mentions; the release
//! skills ask for that review.

use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::PathBuf,
};

const SKILL_NAME: &str = "astronomical";
const INDEX_FILE_NAME: &str = "SKILL.md";
const REFERENCES_FOLDER_NAME: &str = "references";
/// The Agent Skills format caps a description at 1,024 characters, and Zed
/// warns past it.
const MAXIMUM_DESCRIPTION_CHARACTERS: usize = 1024;
/// The index is read whenever the skill is used, so it stays short; detail
/// lives in the reference files, read only when a task needs them.
const MAXIMUM_INDEX_LINES: usize = 150;
const COMMAND_PREFIX: &str = "astronomical ";

// Named character tests instead of closures, following the project's rule to
// keep closures to the minimum.
fn is_help_word_separator(character: char) -> bool {
    character.is_whitespace() || character == '|'
}

fn is_help_flag_wrapper(character: char) -> bool {
    "[](),".contains(character)
}

/// A character right before `astronomical ` that makes it part of a path, a
/// link or a longer word rather than a command.
fn is_part_of_a_path_or_word(character: char) -> bool {
    character.is_alphanumeric() || "/.~-".contains(character)
}

fn is_trailing_punctuation(character: char) -> bool {
    ".,:;)".contains(character)
}

fn is_subcommand_character(character: char) -> bool {
    character.is_ascii_lowercase() || character == '-'
}

fn skill_directory() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../skill")
        .join(SKILL_NAME)
}

fn index_text() -> String {
    fs::read_to_string(skill_directory().join(INDEX_FILE_NAME)).expect("the skill's index exists")
}

fn reference_file_names() -> BTreeSet<String> {
    let mut reference_file_names = BTreeSet::new();
    let references_directory = skill_directory().join(REFERENCES_FOLDER_NAME);
    for directory_entry in fs::read_dir(references_directory).expect("references folder") {
        let file_name = directory_entry.expect("entry").file_name();
        reference_file_names.insert(file_name.to_string_lossy().into_owned());
    }
    reference_file_names
}

/// Every file of the skill, the index first, as (name, text).
fn skill_files() -> Vec<(String, String)> {
    let mut skill_files = vec![(INDEX_FILE_NAME.to_owned(), index_text())];
    for reference_file_name in reference_file_names() {
        let reference_path = skill_directory()
            .join(REFERENCES_FOLDER_NAME)
            .join(&reference_file_name);
        let reference_text = fs::read_to_string(reference_path).expect("reference file");
        skill_files.push((reference_file_name, reference_text));
    }
    skill_files
}

fn frontmatter_value(index_text: &str, key: &str) -> Option<String> {
    let frontmatter = index_text.strip_prefix("---\n")?.split("\n---\n").next()?;
    for frontmatter_line in frontmatter.lines() {
        if let Some(value) = frontmatter_line.strip_prefix(&format!("{key}: ")) {
            return Some(value.to_owned());
        }
    }
    None
}

/// The help's usage lines, keyed by verb: `models` maps to its whole usage line,
/// so a subcommand the skill names can be looked up in it.
fn help_usage_by_verb(help_text: &str) -> BTreeMap<String, String> {
    let mut usage_by_verb: BTreeMap<String, String> = BTreeMap::new();
    for help_line in help_text.lines() {
        let trimmed_line = help_line.trim_start().trim_start_matches("Usage: ");
        let Some(usage) = trimmed_line.strip_prefix(COMMAND_PREFIX) else {
            continue;
        };
        let Some(verb) = usage.split_whitespace().next() else {
            continue;
        };
        if verb.starts_with('-') {
            continue;
        }
        usage_by_verb
            .entry(verb.to_owned())
            .or_default()
            .push_str(&format!(" {usage} "));
    }
    usage_by_verb
}

fn help_flags(help_text: &str) -> BTreeSet<String> {
    let mut help_flags = BTreeSet::new();
    // The help groups alternatives as `(--string|--int|--double|--boolean)`.
    for help_word in help_text.split(is_help_word_separator) {
        let flag = help_word.trim_matches(is_help_flag_wrapper);
        if flag.starts_with("--") && flag.len() > 2 {
            help_flags.insert(flag.to_owned());
        }
    }
    help_flags
}

/// The command text after each `astronomical ` the skill writes, up to the end
/// of its code span or line. The product's name in links and paths
/// (`aosama/astronomical`, `astronomical://`, `~/.astronomical`) is not a command.
fn command_snippets(skill_text: &str) -> Vec<String> {
    let mut command_snippets = Vec::new();
    for (match_index, _) in skill_text.match_indices(COMMAND_PREFIX) {
        let preceding_character = skill_text[..match_index].chars().last();
        if preceding_character.is_some_and(is_part_of_a_path_or_word) {
            continue;
        }
        let command_text = &skill_text[match_index + COMMAND_PREFIX.len()..];
        let snippet_end = command_text.find(['`', '\n']).unwrap_or(command_text.len());
        command_snippets.push(command_text[..snippet_end].to_owned());
    }
    command_snippets
}

fn bare_flag(word: &str) -> Option<&str> {
    let flag = word.trim_end_matches(is_trailing_punctuation);
    (flag.starts_with("--") && flag.len() > 2).then_some(flag)
}

fn is_subcommand_word(word: &str) -> bool {
    !word.is_empty() && word.chars().all(is_subcommand_character)
}

/// What in one command snippet the CLI's help does not have.
fn snippet_problems(command_snippet: &str, help_text: &str) -> Vec<String> {
    let usage_by_verb = help_usage_by_verb(help_text);
    let known_flags = help_flags(help_text);
    let mut snippet_problems = Vec::new();
    let mut snippet_words = command_snippet.split_whitespace();
    let first_word = snippet_words.next().unwrap_or_default();
    if bare_flag(first_word).is_none() {
        match usage_by_verb.get(first_word) {
            None => snippet_problems.push(format!("unknown command `astronomical {first_word}`")),
            Some(verb_usage) => {
                if let Some(second_word) = command_snippet.split_whitespace().nth(1)
                    && is_subcommand_word(second_word)
                    && !verb_usage.contains(&format!(" {second_word} "))
                    && !verb_usage.contains(&format!("|{second_word} "))
                {
                    snippet_problems.push(format!(
                        "unknown subcommand `astronomical {first_word} {second_word}`"
                    ));
                }
            }
        }
    }
    for snippet_word in command_snippet.split_whitespace() {
        if let Some(flag) = bare_flag(snippet_word)
            && !known_flags.contains(flag)
        {
            snippet_problems.push(format!(
                "unknown flag `{flag}` in `astronomical {command_snippet}`"
            ));
        }
    }
    snippet_problems
}

/// Flags named on their own in a code span, such as in a list of options.
fn standalone_flag_problems(skill_text: &str, help_text: &str) -> Vec<String> {
    let known_flags = help_flags(help_text);
    let mut standalone_flag_problems = Vec::new();
    for (span_index, code_span) in skill_text.split('`').enumerate() {
        let is_inside_code_span = span_index % 2 == 1;
        let Some(first_word) = code_span.split_whitespace().next() else {
            continue;
        };
        if is_inside_code_span
            && let Some(flag) = bare_flag(first_word)
            && !known_flags.contains(flag)
        {
            standalone_flag_problems.push(format!("unknown flag `{flag}`"));
        }
    }
    standalone_flag_problems
}

fn skill_text_problems(skill_text: &str, help_text: &str) -> Vec<String> {
    let mut skill_text_problems = Vec::new();
    for command_snippet in command_snippets(skill_text) {
        skill_text_problems.extend(snippet_problems(&command_snippet, help_text));
    }
    skill_text_problems.extend(standalone_flag_problems(skill_text, help_text));
    skill_text_problems
}

#[test]
fn should_name_the_skill_after_its_folder_with_a_description_agents_accept() {
    let index_text = index_text();
    assert_eq!(
        frontmatter_value(&index_text, "name").as_deref(),
        Some(SKILL_NAME)
    );
    let description = frontmatter_value(&index_text, "description").expect("a description");
    assert!(!description.trim().is_empty());
    assert!(
        description.chars().count() <= MAXIMUM_DESCRIPTION_CHARACTERS,
        "description has {} characters",
        description.chars().count()
    );
}

/// The declaration's evidence test (issue #2397) showed an agent reaches for
/// the skill only when the description names the abilities a person asks for.
#[test]
fn should_name_the_abilities_people_ask_for_in_the_description() {
    let description = frontmatter_value(&index_text(), "description").expect("a description");
    for ability in [
        "draw",
        "read text aloud",
        "describe pictures",
        "decision",
        "coding tool",
    ] {
        assert!(
            description.contains(ability),
            "description must mention {ability}"
        );
    }
}

#[test]
fn should_keep_the_index_short_enough_to_read_on_every_task() {
    let index_line_count = index_text().lines().count();
    assert!(
        index_line_count <= MAXIMUM_INDEX_LINES,
        "SKILL.md has {index_line_count} lines; move detail to a reference file"
    );
}

#[test]
fn should_link_every_reference_file_from_the_index_and_no_missing_one() {
    let index_text = index_text();
    let present_reference_files = reference_file_names();
    assert!(!present_reference_files.is_empty());
    for reference_file_name in &present_reference_files {
        assert!(
            index_text.contains(&format!("`references/{reference_file_name}`")),
            "SKILL.md never points to references/{reference_file_name}"
        );
    }
    for (match_index, _) in index_text.match_indices("`references/") {
        let linked_text = &index_text[match_index + "`references/".len()..];
        let linked_file_name = &linked_text[..linked_text.find('`').expect("closing backtick")];
        assert!(
            present_reference_files.contains(linked_file_name),
            "SKILL.md points to a missing references/{linked_file_name}"
        );
    }
}

#[test]
fn should_name_only_commands_subcommands_and_flags_the_cli_has() {
    let help_text = astronomical_cli::help_text();
    let mut problems_by_file = Vec::new();
    for (skill_file_name, skill_text) in skill_files() {
        for skill_text_problem in skill_text_problems(&skill_text, help_text) {
            problems_by_file.push(format!("{skill_file_name}: {skill_text_problem}"));
        }
    }
    assert!(
        problems_by_file.is_empty(),
        "{}",
        problems_by_file.join("\n")
    );
}

#[test]
fn should_refuse_a_command_subcommand_or_flag_the_cli_does_not_have() {
    let invented_text = "Run `astronomical teleport now`, then `astronomical image \"a cat\" --colour red`, \
                         then `astronomical models erase SOME_MODEL`, and pass `--turbo` for speed.";
    let problems = skill_text_problems(invented_text, astronomical_cli::help_text());
    assert_eq!(problems.len(), 4, "{problems:#?}");
    assert!(problems[0].contains("teleport"));
    assert!(problems[1].contains("--colour"));
    assert!(problems[2].contains("models erase"));
    assert!(problems[3].contains("--turbo"));
}

#[test]
fn should_accept_real_commands_and_ignore_the_name_in_links_and_paths() {
    let real_text = "Run `astronomical models download MODEL_ID`, `astronomical launch claude -- -p \"hi\"`, \
                     see https://github.com/aosama/astronomical, open `astronomical://models/huggingface/a/b`, \
                     and read `~/.astronomical/logs`.";
    assert_eq!(
        skill_text_problems(real_text, astronomical_cli::help_text()),
        Vec::<String>::new()
    );
}
