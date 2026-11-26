//! The release notes as Sparkle's update window shows them.
//!
//! The window renders the Markdown embedded in the appcast and stops at the
//! first table it meets: 0.2.135's window showed the Speed heading and its
//! caption and nothing after them, neither the table nor any other section.
//! The window's copy of the notes therefore replaces the Speed section with
//! one line that links to the table on the release page; the GitHub release
//! keeps the notes as written. The Discord announcement
//! (`.github/workflows/discord-release.yml`) makes the same replacement for
//! the same reason.

const SPEED_HEADING: &str = "## Speed";

/// `release_notes` with its Speed section, up to the next level-two heading,
/// replaced by a line linking to `release_page_url`. Notes without a Speed
/// section are returned unchanged.
#[must_use]
pub(crate) fn update_window_notes(release_notes: &str, release_page_url: &str) -> String {
    let mut window_notes = String::with_capacity(release_notes.len());
    let mut is_inside_speed_section = false;
    for notes_line in release_notes.lines() {
        if notes_line.trim_end() == SPEED_HEADING {
            window_notes.push_str(SPEED_HEADING);
            window_notes.push_str("\n\nHow fast each measured model runs, against the previous release, is in the table on the [release page](");
            window_notes.push_str(release_page_url);
            window_notes.push_str(").\n\n");
            is_inside_speed_section = true;
            continue;
        }
        if is_inside_speed_section && notes_line.starts_with("## ") {
            is_inside_speed_section = false;
        }
        if !is_inside_speed_section {
            window_notes.push_str(notes_line);
            window_notes.push('\n');
        }
    }
    window_notes
}

/// Whether `notes_text` holds a Markdown table, which the update window
/// cannot show: a delimiter row such as `|---|---|`.
#[must_use]
pub(crate) fn holds_markdown_table(notes_text: &str) -> bool {
    notes_text
        .lines()
        .any(|notes_line| notes_line.trim_start().starts_with("|---"))
}

#[cfg(test)]
#[path = "update_window_notes_tests.rs"]
mod update_window_notes_tests;
