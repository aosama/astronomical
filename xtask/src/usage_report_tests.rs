//! Unit tests for `cargo xtask usage-report`: parsing recorded GitHub and
//! Hugging Face answers, grouping releases by week, and the report text. No
//! test touches the network.

use std::path::Path;

use crate::usage_report::{
    ModelDownloads, Referrer, ReleaseDownloads, RepositoryCounters, TrafficTotals,
    catalog_repository_ids, estimated_seconds_left, format_count, median_downloads_per_release,
    monday_of_week, parse_model_downloads, parse_options, parse_referrers_tsv,
    parse_release_downloads_tsv, parse_repository_counters_tsv, parse_traffic_totals_tsv,
    render_model_section, render_release_section, render_repository_section, render_table,
    sort_by_recent_downloads, weekly_release_downloads,
};

/// Recorded from `gh api repos/OWNER/NAME/releases --jq ...`, newest first.
const RECORDED_RELEASES_TSV: &str = "v0.2.125\t2026-11-21T07:10:00Z\t3\n\
v0.2.124\t2026-11-21T02:00:00Z\t3\n\
v0.2.120\t2026-11-20T10:00:00Z\t4\n\
v0.2.119\t2026-11-19T10:00:00Z\t4\n\
v0.2.105\t2026-11-14T10:00:00Z\t6\n\
v0.2.102\t2026-11-11T10:00:00Z\t8\n";

fn release(
    tag_name: &str,
    published_date: &str,
    installer_download_count: u64,
) -> ReleaseDownloads {
    ReleaseDownloads {
        tag_name: tag_name.to_owned(),
        published_date: published_date.to_owned(),
        installer_download_count,
    }
}

#[test]
fn should_parse_every_recorded_release_with_its_date_and_downloads() {
    let releases = parse_release_downloads_tsv(RECORDED_RELEASES_TSV);
    assert_eq!(releases.len(), 6);
    assert_eq!(releases[0], release("v0.2.125", "2026-11-21", 3));
    assert_eq!(releases[5], release("v0.2.102", "2026-11-11", 8));
}

#[test]
fn should_skip_release_lines_that_are_malformed() {
    let release_tsv_text = "v1\t2026-11-21T00:00:00Z\tmany\nv2\tshort\nv3\t2026-11-21T00:00:00Z\nv4\t2026-11-20T00:00:00Z\t5\n";
    let releases = parse_release_downloads_tsv(release_tsv_text);
    assert_eq!(releases, vec![release("v4", "2026-11-20", 5)]);
}

#[test]
fn should_return_no_releases_for_empty_text() {
    assert!(parse_release_downloads_tsv("").is_empty());
}

#[test]
fn should_take_the_middle_count_as_median_for_odd_and_even_release_counts() {
    let odd_releases = [
        release("a", "2026-11-21", 9),
        release("b", "2026-11-21", 1),
        release("c", "2026-11-21", 4),
    ];
    assert_eq!(median_downloads_per_release(&odd_releases), Some(4));
    let even_releases = [release("a", "2026-11-21", 3), release("b", "2026-11-21", 6)];
    assert_eq!(median_downloads_per_release(&even_releases), Some(4));
    assert_eq!(median_downloads_per_release(&[]), None);
}

#[test]
fn should_find_the_monday_of_a_week_across_month_year_and_leap_boundaries() {
    assert_eq!(monday_of_week("2026-11-21").as_deref(), Some("2026-11-16"));
    assert_eq!(monday_of_week("2026-11-16").as_deref(), Some("2026-11-16"));
    assert_eq!(monday_of_week("2026-11-22").as_deref(), Some("2026-11-16"));
    assert_eq!(monday_of_week("2027-01-01").as_deref(), Some("2026-12-28"));
    assert_eq!(monday_of_week("2024-03-01").as_deref(), Some("2024-02-26"));
    assert_eq!(monday_of_week("2024-02-29").as_deref(), Some("2024-02-26"));
}

#[test]
fn should_refuse_text_that_is_not_a_calendar_date() {
    assert_eq!(monday_of_week("2026-02-29"), None);
    assert_eq!(monday_of_week("2026-13-01"), None);
    assert_eq!(monday_of_week("2026-11-00"), None);
    assert_eq!(monday_of_week("not a date"), None);
    assert_eq!(monday_of_week(""), None);
}

#[test]
fn should_group_releases_by_week_newest_first_with_totals_and_medians() {
    let releases = parse_release_downloads_tsv(RECORDED_RELEASES_TSV);
    let weekly_downloads = weekly_release_downloads(&releases);
    assert_eq!(weekly_downloads.len(), 2);
    assert_eq!(weekly_downloads[0].week_starting_date, "2026-11-16");
    assert_eq!(weekly_downloads[0].release_count, 4);
    assert_eq!(weekly_downloads[0].installer_download_count, 14);
    assert_eq!(weekly_downloads[0].median_downloads_per_release, 3);
    assert_eq!(weekly_downloads[1].week_starting_date, "2026-11-09");
    assert_eq!(weekly_downloads[1].installer_download_count, 14);
    assert_eq!(weekly_downloads[1].median_downloads_per_release, 7);
}

#[test]
fn should_parse_repository_counters_and_refuse_a_wrong_field_count() {
    assert_eq!(
        parse_repository_counters_tsv("28\t8\t0\n"),
        Some(RepositoryCounters {
            star_count: 28,
            fork_count: 8,
            watcher_count: 0
        })
    );
    assert_eq!(parse_repository_counters_tsv("28\t8"), None);
    assert_eq!(parse_repository_counters_tsv("28\t8\t0\t1"), None);
    assert_eq!(parse_repository_counters_tsv(""), None);
}

#[test]
fn should_parse_traffic_totals_and_referrers() {
    assert_eq!(
        parse_traffic_totals_tsv("631\t31\n"),
        Some(TrafficTotals {
            total_count: 631,
            unique_count: 31
        })
    );
    assert_eq!(parse_traffic_totals_tsv("lots\t31"), None);
    let referrers = parse_referrers_tsv("github.com\t63\t5\nbroken line\nGoogle\t1\t1\n");
    assert_eq!(
        referrers,
        vec![
            Referrer {
                source_name: "github.com".to_owned(),
                view_count: 63,
                unique_visitor_count: 5
            },
            Referrer {
                source_name: "Google".to_owned(),
                view_count: 1,
                unique_visitor_count: 1
            },
        ]
    );
}

#[test]
fn should_list_each_catalog_repository_once_in_catalog_order() {
    let catalog_json_text = r#"{"schema_version": 1, "entries": [
        {"huggingface_id": "org/model-b"},
        {"huggingface_id": "org/model-a"},
        {"huggingface_id": "org/model-b"},
        {"family": "no id here"}
    ]}"#;
    assert_eq!(
        catalog_repository_ids(catalog_json_text),
        Ok(vec!["org/model-b".to_owned(), "org/model-a".to_owned()])
    );
}

#[test]
fn should_refuse_a_catalog_that_is_not_json_or_has_no_entries() {
    assert!(catalog_repository_ids("{not json").is_err());
    assert!(catalog_repository_ids(r#"{"schema_version": 1}"#).is_err());
    assert_eq!(catalog_repository_ids(r#"{"entries": []}"#), Ok(Vec::new()));
}

#[test]
fn should_read_model_counts_and_leave_missing_ones_unknown() {
    let recorded_model_response = r#"{"_id": "x", "id": "org/model", "downloads": 192, "downloadsAllTime": 1250, "likes": 3}"#;
    assert_eq!(
        parse_model_downloads("org/model", recorded_model_response),
        ModelDownloads {
            repository_id: "org/model".to_owned(),
            downloads_last_30_days: Some(192),
            downloads_all_time: Some(1250),
            like_count: Some(3),
        }
    );
    let failed_lookup = parse_model_downloads("org/missing", "");
    assert_eq!(failed_lookup.downloads_last_30_days, None);
    assert_eq!(failed_lookup.like_count, None);
}

#[test]
fn should_sort_models_by_recent_downloads_with_unknown_counts_last() {
    let mut catalog_models = vec![
        parse_model_downloads("org/unknown", ""),
        parse_model_downloads("org/small", r#"{"downloads": 5}"#),
        parse_model_downloads("org/large", r#"{"downloads": 500}"#),
    ];
    sort_by_recent_downloads(&mut catalog_models);
    let mut sorted_ids = Vec::new();
    for catalog_model in &catalog_models {
        sorted_ids.push(catalog_model.repository_id.as_str());
    }
    assert_eq!(sorted_ids, ["org/large", "org/small", "org/unknown"]);
}

#[test]
fn should_group_thousands_and_label_unknown_counts() {
    assert_eq!(format_count(Some(0)), "0");
    assert_eq!(format_count(Some(999)), "999");
    assert_eq!(format_count(Some(1000)), "1,000");
    assert_eq!(format_count(Some(1_234_567)), "1,234,567");
    assert_eq!(format_count(None), "unknown");
}

#[test]
fn should_pad_table_columns_and_align_numbers_right() {
    let table_rows = vec![
        vec!["v0.2.125".to_owned(), "3".to_owned()],
        vec!["v1".to_owned(), "1,250".to_owned()],
    ];
    let table_text = render_table(&["Version", "Downloads"], &table_rows, &[1]);
    assert_eq!(
        table_text,
        "  Version    Downloads\n  v0.2.125           3\n  v1             1,250\n"
    );
}

#[test]
fn should_estimate_time_left_from_the_average_pace() {
    assert_eq!(estimated_seconds_left(10.0, 5, 15), 20.0);
    assert_eq!(estimated_seconds_left(10.0, 0, 15), 0.0);
    assert_eq!(estimated_seconds_left(10.0, 15, 15), 0.0);
}

#[test]
fn should_state_the_up_to_date_estimate_and_its_meaning_in_the_release_section() {
    let releases = parse_release_downloads_tsv(RECORDED_RELEASES_TSV);
    let section_text = render_release_section(&releases, &weekly_release_downloads(&releases));
    assert!(section_text.contains("All releases: 28 downloads across 6 releases."));
    assert!(section_text.contains("median of the last 6 releases: 4."));
    assert!(section_text.contains("Your own Macs count too,"));
    assert!(section_text.contains("release procedure, which downloads each installer once"));
}

#[test]
fn should_say_when_traffic_is_not_available_in_the_repository_section() {
    let counters = RepositoryCounters {
        star_count: 28,
        fork_count: 8,
        watcher_count: 0,
    };
    let views = TrafficTotals {
        total_count: 631,
        unique_count: 31,
    };
    let section_text = render_repository_section(Some(&counters), Some(&views), None, &[]);
    assert!(section_text.contains("Stars 28   Forks 8   Watchers 0"));
    assert!(section_text.contains("Views, last 14 days: 631 by 31 visitors"));
    assert!(section_text.contains("Clones, last 14 days: not available"));
}

#[test]
fn should_show_unknown_for_models_hugging_face_did_not_answer_for() {
    let section_text = render_model_section(&[parse_model_downloads("org/missing", "")]);
    assert!(section_text.contains("org/missing"));
    assert!(section_text.contains("unknown"));
}

fn arguments(argument_texts: &[&str]) -> Vec<String> {
    let mut owned_arguments = Vec::with_capacity(argument_texts.len());
    for argument_text in argument_texts {
        owned_arguments.push((*argument_text).to_owned());
    }
    owned_arguments
}

#[test]
fn should_default_to_the_repository_catalog_and_accept_both_flags() {
    let default_options = parse_options(Path::new("/checkout"), &[]).expect("no flags is valid");
    assert_eq!(default_options.repository, None);
    assert_eq!(
        default_options.catalog_path,
        Path::new("/checkout/registry/download_catalog.json")
    );
    let chosen_options = parse_options(
        Path::new("/checkout"),
        &arguments(&["--repo", "owner/name", "--catalog", "other.json"]),
    )
    .expect("both flags with values are valid");
    assert_eq!(chosen_options.repository.as_deref(), Some("owner/name"));
    assert_eq!(chosen_options.catalog_path, Path::new("other.json"));
}

#[test]
fn should_refuse_unknown_flags_and_flags_without_values() {
    assert!(parse_options(Path::new("/checkout"), &arguments(&["--verbose"])).is_err());
    assert!(parse_options(Path::new("/checkout"), &arguments(&["--repo"])).is_err());
    assert!(parse_options(Path::new("/checkout"), &arguments(&["--catalog"])).is_err());
}

#[test]
fn should_leave_releases_with_unreadable_dates_out_of_the_weekly_table() {
    let releases = [
        release("good", "2026-11-21", 4),
        release("bad", "2026-02-30", 9),
    ];
    let weekly_downloads = weekly_release_downloads(&releases);
    assert_eq!(weekly_downloads.len(), 1);
    assert_eq!(weekly_downloads[0].installer_download_count, 4);
}

#[test]
fn should_take_a_single_release_count_as_its_own_median() {
    assert_eq!(
        median_downloads_per_release(&[release("only", "2026-11-21", 7)]),
        Some(7)
    );
}

#[test]
fn should_render_a_release_section_with_no_releases_without_failing() {
    let section_text = render_release_section(&[], &[]);
    assert!(section_text.contains("All releases: 0 downloads across 0 releases."));
    assert!(section_text.contains("median of the last 0 releases: unknown."));
}
