//! Installer downloads per GitHub release. The in-app updater downloads each
//! new version's disk image from its GitHub release, so one release's count is
//! roughly the Macs that took that update plus fresh installs, the
//! maintainer's own Macs and the release procedure's one digest check
//! included.

use super::calendar_week::monday_of_week;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReleaseDownloads {
    pub tag_name: String,
    pub published_date: String,
    pub installer_download_count: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WeeklyReleaseDownloads {
    pub week_starting_date: String,
    pub release_count: usize,
    pub installer_download_count: u64,
    pub median_downloads_per_release: u64,
}

/// Parses `tag<TAB>published_at<TAB>installer downloads` lines, newest first
/// as the API lists them. Lines that do not have all three fields, or whose
/// count is not a number, are skipped rather than failing the report.
pub fn parse_release_downloads_tsv(release_tsv_text: &str) -> Vec<ReleaseDownloads> {
    let mut releases = Vec::new();
    for release_line in release_tsv_text.lines() {
        let release_fields: Vec<&str> = release_line.split('\t').collect();
        let [tag_name, published_at, download_count_text] = release_fields[..] else {
            continue;
        };
        let (Some(published_date), Ok(installer_download_count)) = (
            published_at.get(..10),
            download_count_text.trim().parse::<u64>(),
        ) else {
            continue;
        };
        releases.push(ReleaseDownloads {
            tag_name: tag_name.to_owned(),
            published_date: published_date.to_owned(),
            installer_download_count,
        });
    }
    releases
}

/// How many recent releases the estimate of up-to-date Macs is taken over:
/// enough to smooth out one odd release, few enough to follow change.
pub const RECENT_RELEASE_COUNT_FOR_ESTIMATE: usize = 10;

/// The newest releases the up-to-date estimate is taken over; `releases`
/// arrive newest first.
pub fn recent_releases_for_estimate(releases: &[ReleaseDownloads]) -> &[ReleaseDownloads] {
    &releases[..releases.len().min(RECENT_RELEASE_COUNT_FOR_ESTIMATE)]
}

/// Every installer download across `releases`.
pub fn total_installer_downloads(releases: &[ReleaseDownloads]) -> u64 {
    let mut total_download_count = 0;
    for release in releases {
        total_download_count += release.installer_download_count;
    }
    total_download_count
}

/// The middle download count of `releases`; the mean of the two middle
/// counts, rounded down, when the number of releases is even.
pub fn median_downloads_per_release(releases: &[ReleaseDownloads]) -> Option<u64> {
    let mut download_counts: Vec<u64> = Vec::with_capacity(releases.len());
    for release in releases {
        download_counts.push(release.installer_download_count);
    }
    download_counts.sort_unstable();
    let middle_position = download_counts.len() / 2;
    match download_counts.len() {
        0 => None,
        release_count if release_count % 2 == 1 => Some(download_counts[middle_position]),
        _ => Some((download_counts[middle_position - 1] + download_counts[middle_position]) / 2),
    }
}

/// Releases grouped by the Monday of their publishing week, newest week
/// first. A median per week, not a total, is the estimate of Macs keeping up
/// to date, because a week with five releases is downloaded by the same Macs
/// five times.
pub fn weekly_release_downloads(releases: &[ReleaseDownloads]) -> Vec<WeeklyReleaseDownloads> {
    let mut week_groups: Vec<(String, Vec<ReleaseDownloads>)> = Vec::new();
    for release in releases {
        let Some(week_starting_date) = monday_of_week(&release.published_date) else {
            continue;
        };
        match week_groups
            .iter_mut()
            .find(|(group_week, _)| *group_week == week_starting_date)
        {
            Some((_, week_releases)) => week_releases.push(release.clone()),
            None => week_groups.push((week_starting_date, vec![release.clone()])),
        }
    }
    week_groups.sort_by(|(left_week, _), (right_week, _)| right_week.cmp(left_week));
    let mut weekly_downloads = Vec::with_capacity(week_groups.len());
    for (week_starting_date, week_releases) in week_groups {
        let mut installer_download_count = 0;
        for release in &week_releases {
            installer_download_count += release.installer_download_count;
        }
        weekly_downloads.push(WeeklyReleaseDownloads {
            week_starting_date,
            release_count: week_releases.len(),
            installer_download_count,
            median_downloads_per_release: median_downloads_per_release(&week_releases).unwrap_or(0),
        });
    }
    weekly_downloads
}
