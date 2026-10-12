//! The published pages must stay readable without zooming.
//!
//! The marketing pages once shipped their smallest labels at 9px: uppercase mono
//! with wide letter-spacing, which reads even smaller than its nominal size. A
//! visitor reads the site, so a size that only looks good at a bird's-eye view is
//! a defect, not a style choice.
//!
//! These tests state the floor a visitor needs rather than the sizes of the day,
//! so a redesign is free to go larger while any declaration that drops back under
//! the floor fails. The smallest sizes live in stylesheets, not in markup, which
//! is why the check reads declared sizes instead of rendering a page.
//!
//! Every stylesheet and every page under `site/` is in scope, including nested
//! report pages: a stylesheet that only some pages load, and a page that only
//! some visitors reach, still publish text to a visitor. An earlier version of
//! this contract listed three stylesheets by name and missed `report.css`, which
//! left a live report page publishing 9.6px labels one directory away.

use std::{fs, path::PathBuf};

/// 12px at the browser default root size: the smallest text a page may publish.
/// Below this, uppercase mono labels stop being readable for anyone who does not
/// zoom, which is the complaint these pages were rewritten to answer.
const MINIMUM_PUBLISHED_TEXT_REM: f64 = 0.75;

const SITE_RELATIVE_PATH: &str = "../../site";

fn site_directory() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(SITE_RELATIVE_PATH)
}

/// Every published stylesheet or page below one directory, so a nested report
/// directory cannot hide from the floor.
fn published_sources(directory: &PathBuf, extension: &str) -> Vec<PathBuf> {
    let mut sources = Vec::new();
    let mut pending = vec![directory.clone()];

    while let Some(current) = pending.pop() {
        let entries = fs::read_dir(&current)
            .unwrap_or_else(|error| panic!("{} should be readable: {error}", current.display()));
        for entry in entries {
            let path = entry.expect("a site entry should be readable").path();
            if path.is_dir() {
                pending.push(path);
            } else if path.extension().is_some_and(|found| found == extension) {
                sources.push(path);
            }
        }
    }

    sources.sort();
    sources
}

/// Every `font-size` value in the site's stylesheets and inline page styles, as
/// declared. Values inside `clamp()` are included because their low end is the
/// smallest size the declaration can produce.
fn declared_font_size_values() -> Vec<(String, String)> {
    let site = site_directory();
    let mut declarations = Vec::new();

    let mut sources = published_sources(&site, "css");
    sources.extend(published_sources(&site, "html"));

    for source in sources {
        let contents = fs::read_to_string(&source)
            .unwrap_or_else(|error| panic!("{} should be readable: {error}", source.display()));
        let name = source
            .strip_prefix(&site)
            .unwrap_or(&source)
            .display()
            .to_string();
        for declaration in contents.split("font-size:").skip(1) {
            let value = declaration
                .split(';')
                .next()
                .unwrap_or_default()
                .trim()
                .to_owned();
            declarations.push((name.clone(), value));
        }
    }

    declarations
}

/// The `rem` and `px` magnitudes inside one declared value, so a `clamp()` low end
/// is measured as strictly as a plain size.
fn magnitudes_in(value: &str) -> Vec<(String, f64)> {
    let mut magnitudes = Vec::new();
    let mut digits = String::new();
    let characters: Vec<char> = value.chars().collect();

    for (index, character) in characters.iter().enumerate() {
        if character.is_ascii_digit() || *character == '.' {
            digits.push(*character);
            continue;
        }
        let upcoming: String = characters[index..].iter().take(3).collect();
        let unit = if upcoming.starts_with("rem") {
            Some("rem")
        } else if upcoming.starts_with("px") {
            Some("px")
        } else {
            None
        };
        if let (Some(unit), Ok(magnitude)) = (unit, digits.parse::<f64>()) {
            magnitudes.push((unit.to_owned(), magnitude));
        }
        digits.clear();
    }

    magnitudes
}

#[test]
fn should_keep_every_published_text_size_at_or_above_the_readability_floor() {
    let mut violations = Vec::new();
    let mut smallest: Option<(String, String, f64)> = None;

    for (source, value) in declared_font_size_values() {
        for (unit, magnitude) in magnitudes_in(&value) {
            let in_rem = if unit == "px" {
                magnitude / 16.0
            } else {
                magnitude
            };
            if in_rem < MINIMUM_PUBLISHED_TEXT_REM {
                violations.push(format!("{source}: font-size: {value}"));
            }
            let is_smaller = smallest
                .as_ref()
                .is_none_or(|(_, _, current)| in_rem < *current);
            if is_smaller {
                smallest = Some((source.clone(), value.clone(), in_rem));
            }
        }
    }

    assert!(
        violations.is_empty(),
        "these declarations publish text below {MINIMUM_PUBLISHED_TEXT_REM}rem (12px): {violations:#?}"
    );

    let (source, value, in_rem) = smallest.expect("the site should declare text sizes");
    println!("smallest published text size: {source} font-size: {value} ({in_rem}rem)");
}

/// The floor above only holds while one `rem` stays the browser default. A rule
/// that shrank the root would shrink every rem-based size with it, silently
/// undoing the floor while leaving the declarations looking unchanged.
#[test]
fn should_keep_the_root_font_size_at_the_browser_default() {
    for stylesheet in published_sources(&site_directory(), "css") {
        let contents = fs::read_to_string(&stylesheet)
            .unwrap_or_else(|error| panic!("{} should be readable: {error}", stylesheet.display()));
        let name = stylesheet.display();

        for block in contents.split('}') {
            let Some((selector, body)) = block.split_once('{') else {
                continue;
            };
            let sets_the_root_scale = selector
                .split(',')
                .any(|part| matches!(part.trim(), "html" | ":root"));
            assert!(
                !(sets_the_root_scale && body.contains("font-size")),
                "{name} must leave the root font size at the browser default, otherwise \
                 every published rem size renders smaller than the readable floor"
            );
        }
    }
}
