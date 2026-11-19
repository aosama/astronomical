use std::fs;
use std::path::Path;

use super::site_link_check::{
    decode_percent_escapes, is_skipped_link, link_problem, link_targets, run,
};

fn write_site_file(repository_root: &Path, site_path: &str, file_text: &str) {
    let site_file = repository_root.join("site").join(site_path);
    fs::create_dir_all(site_file.parent().expect("a site file has a folder"))
        .expect("the site folder should be creatable");
    fs::write(site_file, file_text).expect("the site file should be writable");
}

#[test]
fn should_find_every_href_and_src_in_page_order() {
    let page_html = "<link href=\"./a.css\"><img src=\"b.png\" alt=\"\"><a href=\"#top\">x</a>";
    assert_eq!(link_targets(page_html), vec!["./a.css", "b.png", "#top"]);
}

#[test]
fn should_find_an_empty_attribute_and_stop_at_an_unclosed_one() {
    assert_eq!(link_targets("<a href=\"\">x</a><a href=\"open"), vec![""]);
}

#[test]
fn should_skip_external_addresses_fragments_and_empty_targets() {
    for skipped_target in [
        "",
        "#reports",
        "https://x.test/",
        "http://x.test",
        "mailto:a@x.test",
        "data:,",
        "javascript:void(0)",
    ] {
        assert!(
            is_skipped_link(skipped_target),
            "{skipped_target} should be skipped"
        );
    }
    assert!(!is_skipped_link("./index.html#reports"));
}

#[test]
fn should_decode_percent_escapes_including_multibyte_text() {
    assert_eq!(decode_percent_escapes("a%20b").as_deref(), Some("a b"));
    assert_eq!(decode_percent_escapes("caf%C3%A9").as_deref(), Some("café"));
    assert_eq!(decode_percent_escapes("plain").as_deref(), Some("plain"));
}

#[test]
fn should_refuse_a_malformed_or_non_utf8_escape() {
    assert_eq!(decode_percent_escapes("bad%2"), None);
    assert_eq!(decode_percent_escapes("bad%zz"), None);
    assert_eq!(decode_percent_escapes("%FF"), None);
}

#[test]
fn should_resolve_relative_root_and_parent_links_that_exist() {
    let repository_root = tempfile::tempdir().expect("a scratch folder should be creatable");
    write_site_file(repository_root.path(), "index.html", "");
    write_site_file(repository_root.path(), "reports/one/index.html", "");
    let site_directory = repository_root.path().join("site");
    assert_eq!(
        link_problem(
            &site_directory,
            "reports/one/index.html",
            "../../index.html#reports"
        ),
        None
    );
    assert_eq!(
        link_problem(&site_directory, "reports/one/index.html", "/index.html?v=2"),
        None
    );
    assert_eq!(
        link_problem(&site_directory, "index.html", "./reports/one/"),
        None
    );
}

#[test]
fn should_report_a_missing_file_and_a_folder_without_an_index() {
    let repository_root = tempfile::tempdir().expect("a scratch folder should be creatable");
    write_site_file(repository_root.path(), "index.html", "");
    write_site_file(repository_root.path(), "assets/logo.svg", "");
    let site_directory = repository_root.path().join("site");
    assert_eq!(
        link_problem(&site_directory, "index.html", "./gone.html"),
        Some("broken link target \"./gone.html\"".to_owned())
    );
    assert_eq!(
        link_problem(&site_directory, "index.html", "./assets/"),
        Some("directory target \"./assets/\" has no index.html".to_owned())
    );
    assert_eq!(
        link_problem(&site_directory, "index.html", "./bad%2"),
        Some("percent-encoded target does not decode: ./bad%2".to_owned())
    );
}

#[test]
fn should_pass_a_site_whose_links_all_resolve() {
    let repository_root = tempfile::tempdir().expect("a scratch folder should be creatable");
    write_site_file(
        repository_root.path(),
        "index.html",
        "<a href=\"./support.html\">s</a><a href=\"https://x.test\">x</a>",
    );
    write_site_file(
        repository_root.path(),
        "support.html",
        "<a href=\"./index.html\">h</a>",
    );
    run(repository_root.path(), &[]).expect("a site with working links should pass");
}

#[test]
fn should_fail_a_site_with_a_broken_link() {
    let repository_root = tempfile::tempdir().expect("a scratch folder should be creatable");
    write_site_file(
        repository_root.path(),
        "index.html",
        "<a href=\"./gone.html\">g</a>",
    );
    let refusal =
        run(repository_root.path(), &[]).expect_err("a broken link should fail the check");
    assert!(refusal.contains("violations=1"), "{refusal}");
}

#[test]
fn should_refuse_a_missing_site_folder_and_an_unknown_argument() {
    let repository_root = tempfile::tempdir().expect("a scratch folder should be creatable");
    let missing_site = run(repository_root.path(), &[]).expect_err("no site folder should fail");
    assert!(missing_site.contains("does not exist"), "{missing_site}");
    let unknown_argument = run(repository_root.path(), &["--fix".to_owned()])
        .expect_err("an unknown argument should fail");
    assert!(unknown_argument.starts_with("usage:"), "{unknown_argument}");
}
