use super::{holds_markdown_table, update_window_notes};

const RELEASE_PAGE: &str = "https://github.com/example-owner/astronomical/releases/tag/v0.2.136";

#[test]
fn should_replace_the_speed_section_and_keep_every_other_section() {
    let release_notes = "## Speed\n\n> caption\n\n| Model | Reply |\n|---|---|\n| A | 50 |\n\nUnder the table.\n\n## What's new\n\n- A change.\n\n## Fixes\n\n- A fix.\n";

    let window_notes = update_window_notes(release_notes, RELEASE_PAGE);

    assert_eq!(
        window_notes,
        "## Speed\n\nHow fast each measured model runs, against the previous release, is in the table on the [release page](https://github.com/example-owner/astronomical/releases/tag/v0.2.136).\n\n## What's new\n\n- A change.\n\n## Fixes\n\n- A fix.\n"
    );
    assert!(!holds_markdown_table(&window_notes));
}

#[test]
fn should_return_notes_without_a_speed_section_unchanged() {
    let release_notes = "## What's new\n\n- A change.\n";

    assert_eq!(
        update_window_notes(release_notes, RELEASE_PAGE),
        release_notes
    );
}

#[test]
fn should_drop_a_speed_section_that_ends_the_notes() {
    let release_notes = "## What's new\n\n- A change.\n\n## Speed\n\n|---|\n";

    let window_notes = update_window_notes(release_notes, RELEASE_PAGE);

    assert!(window_notes.starts_with("## What's new\n\n- A change.\n\n## Speed\n\nHow fast"));
    assert!(!holds_markdown_table(&window_notes));
}

#[test]
fn should_not_take_a_deeper_heading_for_the_end_of_the_speed_section() {
    let release_notes = "## Speed\n\n### Detail\n\n|---|\n\n## Fixes\n\n- A fix.\n";

    let window_notes = update_window_notes(release_notes, RELEASE_PAGE);

    assert!(!window_notes.contains("### Detail"));
    assert!(window_notes.ends_with("## Fixes\n\n- A fix.\n"));
}

#[test]
fn should_find_a_table_by_its_delimiter_row_only() {
    assert!(holds_markdown_table("text\n|---|---|\n"));
    assert!(holds_markdown_table("text\n  |---|\n"));
    assert!(!holds_markdown_table("a | b, and a pipe in a sentence\n"));
    assert!(!holds_markdown_table(""));
}
