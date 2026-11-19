//! `cargo xtask site-layout [--check]`: writes the shared header and footer
//! of `site_layout/site_header_and_footer.rs` into every page of the public website
//! under `site/`, or with `--check` refuses a page whose copy differs.
//!
//! Each page keeps a plain `<header class="site-header">` and
//! `<footer class="site-footer">` element, and this command replaces exactly
//! those two elements, so the rest of a page stays hand-written. CI runs the
//! check, so a page edited by hand, or a new page copied from an old one,
//! fails until the command has been run.

mod site_header_and_footer;

use std::fs;
use std::path::Path;

use crate::site_pages::{SITE_DIRECTORY, list_site_pages};
use crate::step_log::log_line;

#[cfg(test)]
pub(crate) use site_header_and_footer::{SITE_LINKS, SiteLink, current_marker};
pub(crate) use site_header_and_footer::{render_site_footer, render_site_header};

const USAGE: &str = "usage: cargo xtask site-layout [--check]";

/// One element of the shared layout: how a page opens and closes it, and
/// what renders its replacement.
struct SharedElement {
    opening_tag: &'static str,
    closing_tag: &'static str,
    render: fn(&str, &str) -> String,
}

const SHARED_ELEMENTS: [SharedElement; 2] = [
    SharedElement {
        opening_tag: "<header class=\"site-header\">",
        closing_tag: "</header>",
        render: render_site_header,
    },
    SharedElement {
        opening_tag: "<footer class=\"site-footer\">",
        closing_tag: "</footer>",
        render: render_site_footer,
    },
];

pub fn run(repository_root: &Path, arguments: &[String]) -> Result<(), String> {
    let is_check_only = match arguments {
        [] => false,
        [check_flag] if check_flag == "--check" => true,
        _ => return Err(USAGE.to_owned()),
    };
    let site_directory = repository_root.join(SITE_DIRECTORY);
    let page_paths = list_site_pages(&site_directory)?;
    let mut drifted_page_paths = Vec::new();
    for page_path in &page_paths {
        let page_file = site_directory.join(page_path);
        let page_html = fs::read_to_string(&page_file).map_err(|read_error| {
            format!("[site-layout] cannot read site/{page_path}: {read_error}")
        })?;
        let laid_out_html = apply_shared_layout(&page_html, page_path)?;
        if laid_out_html == page_html {
            continue;
        }
        drifted_page_paths.push(page_path.clone());
        if !is_check_only {
            fs::write(&page_file, laid_out_html).map_err(|write_error| {
                format!("[site-layout] cannot write site/{page_path}: {write_error}")
            })?;
        }
    }
    report_outcome(is_check_only, page_paths.len(), &drifted_page_paths)
}

fn report_outcome(
    is_check_only: bool,
    page_count: usize,
    drifted_page_paths: &[String],
) -> Result<(), String> {
    let mode = if is_check_only { "check" } else { "write" };
    log_line(&format!(
        "[site-layout] status=summary mode={mode} pages={page_count} differing={}",
        drifted_page_paths.len()
    ));
    if !is_check_only {
        for page_path in drifted_page_paths {
            log_line(&format!("[site-layout] updated=site/{page_path}"));
        }
        return Ok(());
    }
    if drifted_page_paths.is_empty() {
        return Ok(());
    }
    Err(format!(
        "[site-layout] these pages differ from the shared header and footer: {}; run `cargo xtask site-layout` and commit the result",
        drifted_page_paths.join(", ")
    ))
}

/// The page with its header and footer replaced by the shared ones.
pub(crate) fn apply_shared_layout(page_html: &str, page_path: &str) -> Result<String, String> {
    let mut laid_out_html = page_html.to_owned();
    for shared_element in &SHARED_ELEMENTS {
        laid_out_html = replace_shared_element(&laid_out_html, page_path, shared_element)?;
    }
    Ok(laid_out_html)
}

fn replace_shared_element(
    page_html: &str,
    page_path: &str,
    shared_element: &SharedElement,
) -> Result<String, String> {
    let occurrence_count = page_html.matches(shared_element.opening_tag).count();
    if occurrence_count != 1 {
        return Err(format!(
            "[site-layout] site/{page_path} must hold exactly one {} element; found {occurrence_count}",
            shared_element.opening_tag
        ));
    }
    let element_start = page_html
        .find(shared_element.opening_tag)
        .unwrap_or_default();
    let Some(closing_offset) = page_html[element_start..].find(shared_element.closing_tag) else {
        return Err(format!(
            "[site-layout] site/{page_path} never closes its {} element",
            shared_element.opening_tag
        ));
    };
    let element_end = element_start + closing_offset + shared_element.closing_tag.len();
    let line_start = page_html[..element_start]
        .rfind('\n')
        .map_or(0, |newline| newline + 1);
    // An element sharing its line with other markup has no indentation of
    // its own to keep; its continuation lines then start at the margin.
    let leading_text = &page_html[line_start..element_start];
    let indentation = if leading_text.chars().all(char::is_whitespace) {
        leading_text
    } else {
        ""
    };
    let replacement = (shared_element.render)(page_path, indentation);
    Ok(format!(
        "{}{replacement}{}",
        &page_html[..element_start],
        &page_html[element_end..]
    ))
}
