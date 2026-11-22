//! Turns the gathered numbers into plain-text tables, each followed by a line
//! on what its numbers can and cannot tell. Rendering is kept apart from
//! gathering so tests can check the exact text without a network.

use super::model_downloads::ModelDownloads;
use super::release_downloads::{
    ReleaseDownloads, WeeklyReleaseDownloads, median_downloads_per_release,
    recent_releases_for_estimate, total_installer_downloads,
};
use super::repository_traffic::{Referrer, RepositoryCounters, TrafficTotals};

/// A table with a header row and columns padded to their widest cell.
/// Columns listed in `right_aligned_columns` hold numbers and align right.
pub fn render_table(
    column_headers: &[&str],
    table_rows: &[Vec<String>],
    right_aligned_columns: &[usize],
) -> String {
    let mut column_widths: Vec<usize> = Vec::with_capacity(column_headers.len());
    let mut header_cells: Vec<String> = Vec::with_capacity(column_headers.len());
    for column_header in column_headers {
        column_widths.push(column_header.chars().count());
        header_cells.push((*column_header).to_owned());
    }
    for table_row in table_rows {
        for (column_position, cell_text) in table_row.iter().enumerate() {
            if let Some(column_width) = column_widths.get_mut(column_position) {
                *column_width = (*column_width).max(cell_text.chars().count());
            }
        }
    }
    let mut table_text = render_row(&header_cells, &column_widths, right_aligned_columns);
    for table_row in table_rows {
        table_text.push_str(&render_row(
            table_row,
            &column_widths,
            right_aligned_columns,
        ));
    }
    table_text
}

fn render_row(
    row_cells: &[String],
    column_widths: &[usize],
    right_aligned_columns: &[usize],
) -> String {
    let mut padded_cells: Vec<String> = Vec::with_capacity(row_cells.len());
    for (column_position, cell_text) in row_cells.iter().enumerate() {
        let column_width = column_widths.get(column_position).copied().unwrap_or(0);
        if right_aligned_columns.contains(&column_position) {
            padded_cells.push(format!("{cell_text:>column_width$}"));
        } else {
            padded_cells.push(format!("{cell_text:<column_width$}"));
        }
    }
    format!("  {}\n", padded_cells.join("   ").trim_end())
}

/// A count with thousands separators, or "unknown" when it could not be read.
pub fn format_count(known_count: Option<u64>) -> String {
    let Some(count) = known_count else {
        return "unknown".to_owned();
    };
    let digits = count.to_string();
    let mut grouped_digits = String::with_capacity(digits.len() + digits.len() / 3);
    for (digit_position, digit) in digits.chars().enumerate() {
        if digit_position > 0 && (digits.len() - digit_position) % 3 == 0 {
            grouped_digits.push(',');
        }
        grouped_digits.push(digit);
    }
    grouped_digits
}

pub fn render_release_section(
    releases: &[ReleaseDownloads],
    weekly_downloads: &[WeeklyReleaseDownloads],
) -> String {
    let mut section_text = String::from("INSTALLER DOWNLOADS BY WEEK\n");
    let mut week_rows = Vec::with_capacity(weekly_downloads.len());
    for week in weekly_downloads {
        week_rows.push(vec![
            week.week_starting_date.clone(),
            week.release_count.to_string(),
            format_count(Some(week.installer_download_count)),
            week.median_downloads_per_release.to_string(),
        ]);
    }
    section_text.push_str(&render_table(
        &["Week of", "Releases", "Downloads", "Median per release"],
        &week_rows,
        &[1, 2, 3],
    ));
    let recent_releases = recent_releases_for_estimate(releases);
    section_text.push_str(&format!(
        "\n  All releases: {} downloads across {} releases.\n  Macs keeping up to date, estimated as the median of the last {} releases: {}.\n",
        format_count(Some(total_installer_downloads(releases))),
        releases.len(),
        recent_releases.len(),
        format_count(median_downloads_per_release(recent_releases)),
    ));
    section_text.push_str(
        "  What this means: the updater downloads every new version from its GitHub release, so each\n  release's count is roughly the Macs that updated plus new installs. Your own Macs count too,\n  and so does the release procedure, which downloads each installer once to check its digest.\n",
    );
    section_text
}

pub fn render_repository_section(
    repository_counters: Option<&RepositoryCounters>,
    views: Option<&TrafficTotals>,
    clones: Option<&TrafficTotals>,
    referrers: &[Referrer],
) -> String {
    let mut section_text = String::from("REPOSITORY\n");
    if let Some(counters) = repository_counters {
        section_text.push_str(&format!(
            "  Stars {}   Forks {}   Watchers {}\n",
            counters.star_count, counters.fork_count, counters.watcher_count
        ));
    }
    section_text.push_str(&format!(
        "  Views, last 14 days: {}\n",
        render_traffic(views, "visitors")
    ));
    section_text.push_str(&format!(
        "  Clones, last 14 days: {}\n",
        render_traffic(clones, "cloners")
    ));
    if !referrers.is_empty() {
        let mut referrer_rows = Vec::with_capacity(referrers.len());
        for referrer in referrers {
            referrer_rows.push(vec![
                referrer.source_name.clone(),
                referrer.view_count.to_string(),
                referrer.unique_visitor_count.to_string(),
            ]);
        }
        section_text.push_str("\n  Where visitors came from, last 14 days:\n");
        section_text.push_str(&render_table(
            &["Source", "Views", "Visitors"],
            &referrer_rows,
            &[1, 2],
        ));
    }
    section_text.push_str("  What this means: interest in the project, not use of the app. Clones include every CI run\n  and automated checkout, so they say little about people.\n");
    section_text
}

fn render_traffic(traffic_totals: Option<&TrafficTotals>, unique_label: &str) -> String {
    match traffic_totals {
        Some(totals) => format!(
            "{} by {} {unique_label}",
            format_count(Some(totals.total_count)),
            totals.unique_count
        ),
        None => "not available (GitHub shows traffic to people with write access only)".to_owned(),
    }
}

pub fn render_model_section(model_downloads: &[ModelDownloads]) -> String {
    let mut section_text = String::from("CATALOG MODELS ON HUGGING FACE\n");
    let mut model_rows = Vec::with_capacity(model_downloads.len());
    for model in model_downloads {
        model_rows.push(vec![
            model.repository_id.clone(),
            format_count(model.downloads_last_30_days),
            format_count(model.downloads_all_time),
            format_count(model.like_count),
        ]);
    }
    section_text.push_str(&render_table(
        &["Model", "Last 30 days", "All time", "Likes"],
        &model_rows,
        &[1, 2, 3],
    ));
    section_text.push_str(
        "  What this means: Hugging Face counts requests for a model's files, not people, and your own\n  downloads count too. Read it as which models draw interest, not how many people use them.\n",
    );
    section_text
}
