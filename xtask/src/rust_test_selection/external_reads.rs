//! Files and folders outside a package that its Rust sources read: fixtures,
//! the download catalog, the config schema, the menu app's status fixtures.
//! A change to one of them must select that package's tests, although the
//! file is not in the package's folder.
//!
//! Every such read in this repository names its path as a string literal
//! that climbs out of the package with `../`, through `include_str!`,
//! `include_bytes!`, or a join onto `CARGO_MANIFEST_DIR`. The scan resolves
//! each climbing literal from the source file's folder and from the package's
//! folder, and keeps the results the repository really has. A read the scan
//! cannot see is still safe: an unclaimed path outside every package selects
//! every test (see `selection`).

use std::collections::BTreeSet;

/// The repository paths `source_text` reads from outside its package.
/// `known_repository_files` lists every tracked file, so a folder counts as
/// known when a tracked file lies beneath it.
pub(crate) fn external_paths_read_by(
    package_directory: &str,
    source_relative_path: &str,
    source_text: &str,
    known_repository_files: &BTreeSet<String>,
) -> BTreeSet<String> {
    let source_directory = parent_folder(source_relative_path);
    let mut external_paths = BTreeSet::new();
    for climbing_literal in climbing_string_literals(source_text) {
        let relative_literal = climbing_literal.trim_start_matches('/');
        for base_directory in [source_directory, package_directory] {
            let Some(resolved_path) = resolve_repository_path(base_directory, relative_literal)
            else {
                continue;
            };
            let is_outside_package = !is_same_or_beneath(&resolved_path, package_directory);
            let is_meaningful =
                !resolved_path.is_empty() && !is_same_or_beneath(package_directory, &resolved_path);
            if is_outside_package
                && is_meaningful
                && is_known_path(&resolved_path, known_repository_files)
            {
                external_paths.insert(resolved_path);
            }
        }
    }
    external_paths
}

/// String literals in `source_text` that contain `../`. Escapes are skipped
/// so an escaped quote never ends a literal early; raw strings and comments
/// are read like ordinary text, which can only add a candidate the
/// existence check then drops.
pub(crate) fn climbing_string_literals(source_text: &str) -> Vec<String> {
    let mut literals = Vec::new();
    let mut current_literal: Option<String> = None;
    let mut is_escaped = false;
    for source_character in source_text.chars() {
        match (&mut current_literal, source_character) {
            (None, '"') => current_literal = Some(String::new()),
            (None, _) => {}
            (Some(literal), _) if is_escaped => {
                literal.push(source_character);
                is_escaped = false;
            }
            (Some(_), '\\') => is_escaped = true,
            (Some(literal), '"') => {
                if literal.contains("../") {
                    literals.push(std::mem::take(literal));
                }
                current_literal = None;
            }
            (Some(literal), _) => literal.push(source_character),
        }
    }
    literals
}

/// `relative_path` joined to `base_directory` with `.` and `..` resolved, or
/// None when it climbs above the repository root.
pub(crate) fn resolve_repository_path(base_directory: &str, relative_path: &str) -> Option<String> {
    let mut path_components: Vec<&str> = Vec::new();
    for path_component in base_directory.split('/').chain(relative_path.split('/')) {
        match path_component {
            "" | "." => {}
            ".." => {
                path_components.pop()?;
            }
            named_component => path_components.push(named_component),
        }
    }
    Some(path_components.join("/"))
}

fn parent_folder(relative_path: &str) -> &str {
    match relative_path.rfind('/') {
        Some(separator_index) => &relative_path[..separator_index],
        None => "",
    }
}

/// Whether `repository_path` is `folder` itself or lies beneath it. The
/// repository root, an empty folder, holds everything.
pub(crate) fn is_same_or_beneath(repository_path: &str, folder: &str) -> bool {
    folder.is_empty()
        || repository_path == folder
        || repository_path.starts_with(&format!("{folder}/"))
}

fn is_known_path(repository_path: &str, known_repository_files: &BTreeSet<String>) -> bool {
    if known_repository_files.contains(repository_path) {
        return true;
    }
    let folder_prefix = format!("{repository_path}/");
    match known_repository_files.range(folder_prefix.clone()..).next() {
        Some(first_file_after) => first_file_after.starts_with(&folder_prefix),
        None => false,
    }
}
