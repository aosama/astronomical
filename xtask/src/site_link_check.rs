//! `cargo xtask check-site-links`: every relative `href` or `src` in a page
//! of the public website must name a file that is published with it.
//!
//! Typed port of the former `scripts/check-site-links.js`, keeping its
//! output lines and its exit code (failure on any violation) so the CI step
//! reads the same. A link that leaves the site (`https:`, `mailto:` and the
//! like) or stays on its own page (`#section`) is not checked: the first
//! would need the network, which a CI check must not depend on, and the
//! second has no file to find.

use std::fs;
use std::path::{Path, PathBuf};

use crate::site_pages::{SITE_DIRECTORY, list_site_pages};

const SKIPPED_ADDRESS_PREFIXES: [&str; 5] =
    ["http://", "https://", "mailto:", "data:", "javascript:"];
const LINK_ATTRIBUTE_PREFIXES: [&str; 2] = ["href=\"", "src=\""];

pub fn run(repository_root: &Path, arguments: &[String]) -> Result<(), String> {
    if !arguments.is_empty() {
        return Err("usage: cargo xtask check-site-links".to_owned());
    }
    let site_directory = repository_root.join(SITE_DIRECTORY);
    if !site_directory.is_dir() {
        return Err("static site directory does not exist: site/".to_owned());
    }
    println!("[check-site-links] status=start site_dir=site/");
    let page_paths = list_site_pages(&site_directory)?;
    let mut checked_link_count = 0;
    let mut violations = Vec::new();
    for page_path in &page_paths {
        let page_html = fs::read_to_string(site_directory.join(page_path))
            .map_err(|read_error| format!("cannot read site/{page_path}: {read_error}"))?;
        for link_target in link_targets(&page_html) {
            if is_skipped_link(link_target) {
                continue;
            }
            checked_link_count += 1;
            if let Some(problem) = link_problem(&site_directory, page_path, link_target) {
                violations.push(format!("{page_path}: {problem}"));
            }
        }
    }
    println!(
        "[check-site-links] status=summary html_files={} links_checked={checked_link_count} violations={}",
        page_paths.len(),
        violations.len()
    );
    for violation in &violations {
        eprintln!("[check-site-links] violation: {violation}");
    }
    if violations.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "[check-site-links] status=failed violations={}",
            violations.len()
        ))
    }
}

/// The value of every double-quoted `href` and `src` attribute, in page
/// order. A text scan rather than an HTML parser, as the script did: the
/// pages are hand-written with double-quoted attributes, and a parser
/// would be a dependency for no change in what is found.
pub(crate) fn link_targets(page_html: &str) -> Vec<&str> {
    let mut found_targets = Vec::new();
    let mut search_offset = 0;
    while let Some((attribute_start, prefix_length)) =
        next_link_attribute(&page_html[search_offset..])
    {
        let value_start = search_offset + attribute_start + prefix_length;
        let Some(value_length) = page_html[value_start..].find('"') else {
            break;
        };
        found_targets.push(&page_html[value_start..value_start + value_length]);
        search_offset = value_start + value_length + 1;
    }
    found_targets
}

/// Where the nearest `href="` or `src="` starts, and that prefix's length.
fn next_link_attribute(remaining_html: &str) -> Option<(usize, usize)> {
    let mut nearest_attribute: Option<(usize, usize)> = None;
    for attribute_prefix in LINK_ATTRIBUTE_PREFIXES {
        let Some(attribute_start) = remaining_html.find(attribute_prefix) else {
            continue;
        };
        let is_nearer = match nearest_attribute {
            Some((nearest_start, _)) => attribute_start < nearest_start,
            None => true,
        };
        if is_nearer {
            nearest_attribute = Some((attribute_start, attribute_prefix.len()));
        }
    }
    nearest_attribute
}

pub(crate) fn is_skipped_link(link_target: &str) -> bool {
    link_target.is_empty()
        || link_target.starts_with('#')
        || SKIPPED_ADDRESS_PREFIXES
            .iter()
            .any(|address_prefix| link_target.starts_with(address_prefix))
}

/// Why this link from this page leads nowhere, or `None` when it resolves.
pub(crate) fn link_problem(
    site_directory: &Path,
    page_path: &str,
    link_target: &str,
) -> Option<String> {
    let target_path = link_target.split(['#', '?']).next().unwrap_or_default();
    let Some(decoded_target) = decode_percent_escapes(target_path) else {
        return Some(format!(
            "percent-encoded target does not decode: {link_target}"
        ));
    };
    let resolved_path = resolve_link_target(site_directory, page_path, &decoded_target);
    if !resolved_path.exists() {
        return Some(format!("broken link target \"{link_target}\""));
    }
    if resolved_path.is_dir() && !resolved_path.join("index.html").exists() {
        return Some(format!(
            "directory target \"{link_target}\" has no index.html"
        ));
    }
    None
}

/// A leading `/` starts from the site root, as the published site serves
/// it; anything else starts from the folder of the page holding the link.
fn resolve_link_target(site_directory: &Path, page_path: &str, decoded_target: &str) -> PathBuf {
    if let Some(root_relative_target) = decoded_target.strip_prefix('/') {
        return site_directory.join(root_relative_target);
    }
    let page_folder = Path::new(page_path).parent().unwrap_or(Path::new(""));
    site_directory.join(page_folder).join(decoded_target)
}

/// `%20` and friends turned back into text, or `None` for a malformed
/// escape or bytes that are not UTF-8: the cases in which the browser's
/// `decodeURIComponent`, which the script used, throws.
pub(crate) fn decode_percent_escapes(encoded_text: &str) -> Option<String> {
    let encoded_bytes = encoded_text.as_bytes();
    let mut decoded_bytes = Vec::with_capacity(encoded_bytes.len());
    let mut byte_position = 0;
    while byte_position < encoded_bytes.len() {
        if encoded_bytes[byte_position] != b'%' {
            decoded_bytes.push(encoded_bytes[byte_position]);
            byte_position += 1;
            continue;
        }
        let escape_digits = encoded_text.get(byte_position + 1..byte_position + 3)?;
        decoded_bytes.push(u8::from_str_radix(escape_digits, 16).ok()?);
        byte_position += 3;
    }
    String::from_utf8(decoded_bytes).ok()
}
