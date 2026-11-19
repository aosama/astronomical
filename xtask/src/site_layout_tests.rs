use std::fs;
use std::path::Path;

use super::site_layout::{
    SITE_DIRECTORY, SITE_LINKS, SiteLink, apply_shared_layout, current_marker, list_site_pages,
    render_site_footer, render_site_header, run,
};

const HAND_WRITTEN_PAGE: &str = "<body>\n    <header class=\"site-header\">\n      <nav><a href=\"./index.html\">Home</a></nav>\n    </header>\n    <main>Kept as written</main>\n    <footer class=\"site-footer\"><p>old</p></footer>\n</body>\n";

fn site_link_named(label: &str) -> &'static SiteLink {
    SITE_LINKS
        .iter()
        .find(|site_link| site_link.label == label)
        .expect("the menu should hold this link")
}

fn write_site_page(repository_root: &Path, page_path: &str, page_html: &str) {
    let page_file = repository_root.join(SITE_DIRECTORY).join(page_path);
    fs::create_dir_all(page_file.parent().expect("a page file has a folder"))
        .expect("the page folder should be creatable");
    fs::write(page_file, page_html).expect("the page should be writable");
}

#[test]
fn should_mark_the_page_a_link_opens_as_the_current_page() {
    assert_eq!(
        current_marker(site_link_named("Support"), "support.html"),
        Some("page")
    );
    let support_header = render_site_header("support.html", "");
    assert!(
        support_header.contains("<a href=\"./support.html\" aria-current=\"page\">Support</a>")
    );
    assert_eq!(support_header.matches("aria-current").count(), 1);
}

#[test]
fn should_mark_field_notes_as_the_current_section_on_a_report_page() {
    let report_path = "reports/opencode-plugin/index.html";
    assert_eq!(
        current_marker(site_link_named("Field notes"), report_path),
        Some("true")
    );
    assert!(
        render_site_header(report_path, "")
            .contains("<a href=\"../../index.html#reports\" aria-current=\"true\">Field notes</a>")
    );
}

#[test]
fn should_mark_no_link_current_on_the_home_page() {
    assert!(!render_site_header("index.html", "").contains("aria-current"));
}

#[test]
fn should_never_mark_a_footer_link_current() {
    assert!(!render_site_footer("support.html", "").contains("aria-current"));
}

#[test]
fn should_list_the_same_links_in_the_header_and_the_footer() {
    let header_markup = render_site_header("privacy.html", "");
    let footer_markup = render_site_footer("privacy.html", "");
    for site_link in &SITE_LINKS {
        let link_text = format!(">{}</a>", site_link.label);
        assert!(
            header_markup.contains(&link_text),
            "header lacks {}",
            site_link.label
        );
        assert!(
            footer_markup.contains(&link_text),
            "footer lacks {}",
            site_link.label
        );
    }
}

#[test]
fn should_point_site_links_up_two_folders_from_a_page_two_folders_deep() {
    let report_header = render_site_header("reports/memory-ceiling-throughput/index.html", "");
    assert!(report_header.contains("href=\"../../models.html\""));
    assert!(report_header.contains("src=\"../../assets/mark.svg\""));
    assert!(report_header.contains("href=\"https://github.com/aosama/astronomical\""));
    assert!(!report_header.contains("\"./"));
}

#[test]
fn should_replace_only_the_header_and_footer_and_keep_their_indentation() {
    let laid_out_page = apply_shared_layout(HAND_WRITTEN_PAGE, "models.html")
        .expect("a page with one header and one footer should lay out");
    assert!(laid_out_page.contains("    <main>Kept as written</main>\n"));
    assert!(laid_out_page.contains("\n    <header class=\"site-header\">\n      <!-- Shared"));
    assert!(laid_out_page.contains("\n    </header>\n"));
    assert!(!laid_out_page.contains("<p>old</p>"));
    assert!(!laid_out_page.contains(">Home</a>"));
}

#[test]
fn should_change_nothing_when_applied_to_a_page_already_laid_out() {
    let laid_out_once = apply_shared_layout(HAND_WRITTEN_PAGE, "models.html")
        .expect("the first pass should lay out");
    let laid_out_twice =
        apply_shared_layout(&laid_out_once, "models.html").expect("the second pass should lay out");
    assert_eq!(laid_out_once, laid_out_twice);
}

#[test]
fn should_refuse_a_page_without_a_shared_header() {
    let headerless_page = "<body><footer class=\"site-footer\"></footer></body>";
    let refusal = apply_shared_layout(headerless_page, "orphan.html")
        .expect_err("a page without a header should be refused");
    assert!(refusal.contains("site/orphan.html"), "{refusal}");
    assert!(refusal.contains("found 0"), "{refusal}");
}

#[test]
fn should_refuse_a_page_with_two_shared_footers() {
    let doubled_page = "<header class=\"site-header\"></header><footer class=\"site-footer\"></footer><footer class=\"site-footer\"></footer>";
    let refusal = apply_shared_layout(doubled_page, "doubled.html")
        .expect_err("a page with two footers should be refused");
    assert!(refusal.contains("found 2"), "{refusal}");
}

#[test]
fn should_refuse_a_header_that_never_closes() {
    let unclosed_page = "<header class=\"site-header\"><nav></nav>";
    let refusal = apply_shared_layout(unclosed_page, "unclosed.html")
        .expect_err("an unclosed header should be refused");
    assert!(refusal.contains("never closes"), "{refusal}");
}

#[test]
fn should_list_pages_in_nested_folders_with_forward_slashes_in_sorted_order() {
    let repository_root = tempfile::tempdir().expect("a scratch folder should be creatable");
    write_site_page(repository_root.path(), "support.html", HAND_WRITTEN_PAGE);
    write_site_page(
        repository_root.path(),
        "reports/one/index.html",
        HAND_WRITTEN_PAGE,
    );
    write_site_page(repository_root.path(), "assets/notes.txt", "not a page");
    let page_paths = list_site_pages(&repository_root.path().join(SITE_DIRECTORY))
        .expect("the scratch site should list");
    assert_eq!(page_paths, vec!["reports/one/index.html", "support.html"]);
}

#[test]
fn should_fail_the_check_on_a_differing_page_without_changing_it() {
    let repository_root = tempfile::tempdir().expect("a scratch folder should be creatable");
    write_site_page(repository_root.path(), "support.html", HAND_WRITTEN_PAGE);
    let refusal = run(repository_root.path(), &["--check".to_owned()])
        .expect_err("a hand-written header should fail the check");
    assert!(refusal.contains("support.html"), "{refusal}");
    let page_after_check = fs::read_to_string(repository_root.path().join("site/support.html"))
        .expect("the page should still be readable");
    assert_eq!(page_after_check, HAND_WRITTEN_PAGE);
}

#[test]
fn should_pass_the_check_once_the_pages_are_written() {
    let repository_root = tempfile::tempdir().expect("a scratch folder should be creatable");
    write_site_page(repository_root.path(), "support.html", HAND_WRITTEN_PAGE);
    write_site_page(
        repository_root.path(),
        "reports/one/index.html",
        HAND_WRITTEN_PAGE,
    );
    run(repository_root.path(), &[]).expect("writing the layout should succeed");
    run(repository_root.path(), &["--check".to_owned()])
        .expect("the check should pass after the layout is written");
}

#[test]
fn should_refuse_an_unknown_argument() {
    let repository_root = tempfile::tempdir().expect("a scratch folder should be creatable");
    let refusal = run(repository_root.path(), &["--write".to_owned()])
        .expect_err("an unknown flag should be refused");
    assert!(refusal.starts_with("usage:"), "{refusal}");
}

#[test]
fn should_refuse_a_site_without_pages() {
    let repository_root = tempfile::tempdir().expect("a scratch folder should be creatable");
    fs::create_dir_all(repository_root.path().join(SITE_DIRECTORY))
        .expect("the empty site folder should be creatable");
    let refusal = run(repository_root.path(), &["--check".to_owned()])
        .expect_err("a site without pages should be refused");
    assert!(refusal.contains("no pages found"), "{refusal}");
}

#[test]
fn should_point_site_links_up_one_folder_from_a_page_one_folder_deep() {
    let nested_header = render_site_header("reports/index.html", "");
    assert!(nested_header.contains("href=\"../models.html\""));
    assert!(!nested_header.contains("../../"));
}

#[test]
fn should_start_continuation_lines_at_the_margin_when_the_header_shares_its_line() {
    let inline_page = "<body><header class=\"site-header\"></header>\n<footer class=\"site-footer\"></footer></body>";
    let laid_out_page = apply_shared_layout(inline_page, "models.html")
        .expect("a header sharing its line should still lay out");
    assert!(laid_out_page.starts_with("<body><header class=\"site-header\">\n  <!-- Shared"));
    assert!(laid_out_page.contains("\n</header>\n<footer"));
}
