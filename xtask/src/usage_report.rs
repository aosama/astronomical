//! `cargo xtask usage-report [--repo owner/name] [--catalog PATH]`: prints
//! what public numbers say about who uses Astronomical (issue #2413, phase 0).
//!
//! The app sends nothing about its users, so the report reads only what
//! GitHub and Hugging Face already count: installer downloads per release,
//! the repository's stars, forks and 14-day traffic, and downloads of every
//! model in the download catalog. Each table ends with a line on what its
//! numbers can and cannot tell. Progress goes to standard error and the
//! report to standard output, so `> report.txt` saves just the report.

mod calendar_week;
mod model_downloads;
mod progress_reporter;
mod release_downloads;
mod remote_sources;
mod report_text;
mod repository_traffic;

use std::fs;
use std::path::{Path, PathBuf};

#[cfg(test)]
pub(crate) use calendar_week::monday_of_week;
#[cfg(test)]
pub(crate) use model_downloads::{
    ModelDownloads, catalog_repository_ids, parse_model_downloads, sort_by_recent_downloads,
};
#[cfg(test)]
pub(crate) use progress_reporter::estimated_seconds_left;
#[cfg(test)]
pub(crate) use release_downloads::{
    ReleaseDownloads, median_downloads_per_release, parse_release_downloads_tsv,
    weekly_release_downloads,
};
#[cfg(test)]
pub(crate) use report_text::{
    format_count, render_model_section, render_release_section, render_repository_section,
    render_table,
};
#[cfg(test)]
pub(crate) use repository_traffic::{
    Referrer, RepositoryCounters, TrafficTotals, parse_referrers_tsv,
    parse_repository_counters_tsv, parse_traffic_totals_tsv,
};

const DOWNLOAD_CATALOG_PATH: &str = "registry/download_catalog.json";

/// Release asset filter: the disk image is what the updater and the website
/// download, so any other asset attached to a release is not counted.
const RELEASE_DOWNLOADS_JQ: &str = r#".[] | select(.draft | not) | [.tag_name, .published_at, ([.assets[] | select(.name | endswith(".dmg")) | .download_count] | add // 0)] | @tsv"#;

pub(crate) struct UsageReportOptions {
    pub(crate) repository: Option<String>,
    pub(crate) catalog_path: PathBuf,
}

pub fn run(repository_root: &Path, command_arguments: &[String]) -> Result<(), String> {
    let options = parse_options(repository_root, command_arguments)?;
    let repository = match options.repository {
        Some(repository) => repository,
        None => remote_sources::gh_output(&[
            "repo",
            "view",
            "--json",
            "nameWithOwner",
            "--jq",
            ".nameWithOwner",
        ])?
        .trim()
        .to_owned(),
    };
    let release_section = gather_release_section(&repository)?;
    let repository_section = gather_repository_section(&repository);
    let model_section = gather_model_section(&options.catalog_path)?;
    println!("Astronomical usage report for {repository}\n");
    println!("{release_section}\n{repository_section}\n{model_section}");
    Ok(())
}

pub(crate) fn parse_options(
    repository_root: &Path,
    command_arguments: &[String],
) -> Result<UsageReportOptions, String> {
    let mut options = UsageReportOptions {
        repository: None,
        catalog_path: repository_root.join(DOWNLOAD_CATALOG_PATH),
    };
    let mut remaining_arguments = command_arguments.iter();
    while let Some(argument) = remaining_arguments.next() {
        match argument.as_str() {
            "--repo" => {
                options.repository = Some(
                    remaining_arguments
                        .next()
                        .ok_or("--repo requires owner/name")?
                        .clone(),
                )
            }
            "--catalog" => {
                options.catalog_path = PathBuf::from(
                    remaining_arguments
                        .next()
                        .ok_or("--catalog requires a path")?,
                )
            }
            other => return Err(format!("unexpected usage-report argument: {other}")),
        }
    }
    Ok(options)
}

fn gather_release_section(repository: &str) -> Result<String, String> {
    eprintln!("[usage-report] reading releases of {repository}");
    let release_tsv_text = remote_sources::gh_output(&[
        "api",
        "--paginate",
        &format!("repos/{repository}/releases?per_page=100"),
        "--jq",
        RELEASE_DOWNLOADS_JQ,
    ])?;
    let releases = release_downloads::parse_release_downloads_tsv(&release_tsv_text);
    let weekly_downloads = release_downloads::weekly_release_downloads(&releases);
    Ok(report_text::render_release_section(
        &releases,
        &weekly_downloads,
    ))
}

/// Traffic needs write access and may be refused; the section then says so
/// instead of failing the whole report.
fn gather_repository_section(repository: &str) -> String {
    eprintln!("[usage-report] reading repository counters and traffic");
    let counters_text = remote_sources::gh_output(&[
        "api",
        &format!("repos/{repository}"),
        "--jq",
        "[.stargazers_count, .forks_count, .subscribers_count] | @tsv",
    ]);
    let views_text = remote_sources::gh_output(&[
        "api",
        &format!("repos/{repository}/traffic/views"),
        "--jq",
        "[.count, .uniques] | @tsv",
    ]);
    let clones_text = remote_sources::gh_output(&[
        "api",
        &format!("repos/{repository}/traffic/clones"),
        "--jq",
        "[.count, .uniques] | @tsv",
    ]);
    let referrers_text = remote_sources::gh_output(&[
        "api",
        &format!("repos/{repository}/traffic/popular/referrers"),
        "--jq",
        ".[] | [.referrer, .count, .uniques] | @tsv",
    ]);
    report_text::render_repository_section(
        parse_counters_answer(counters_text).as_ref(),
        parse_traffic_answer(views_text).as_ref(),
        parse_traffic_answer(clones_text).as_ref(),
        &repository_traffic::parse_referrers_tsv(&referrers_text.unwrap_or_default()),
    )
}

fn parse_counters_answer(
    counters_answer: Result<String, String>,
) -> Option<repository_traffic::RepositoryCounters> {
    repository_traffic::parse_repository_counters_tsv(&counters_answer.ok()?)
}

fn parse_traffic_answer(
    traffic_answer: Result<String, String>,
) -> Option<repository_traffic::TrafficTotals> {
    repository_traffic::parse_traffic_totals_tsv(&traffic_answer.ok()?)
}

fn gather_model_section(catalog_path: &Path) -> Result<String, String> {
    let catalog_json_text = fs::read_to_string(catalog_path)
        .map_err(|read_error| format!("cannot read {}: {read_error}", catalog_path.display()))?;
    let repository_ids = model_downloads::catalog_repository_ids(&catalog_json_text)?;
    let mut progress =
        progress_reporter::ProgressReporter::start("Hugging Face models", repository_ids.len());
    let mut catalog_model_downloads = Vec::with_capacity(repository_ids.len());
    for (model_position, repository_id) in repository_ids.iter().enumerate() {
        let model_response_text =
            remote_sources::hugging_face_model_response(repository_id).unwrap_or_default();
        catalog_model_downloads.push(model_downloads::parse_model_downloads(
            repository_id,
            &model_response_text,
        ));
        progress.record_completed(model_position + 1);
    }
    model_downloads::sort_by_recent_downloads(&mut catalog_model_downloads);
    Ok(report_text::render_model_section(&catalog_model_downloads))
}
