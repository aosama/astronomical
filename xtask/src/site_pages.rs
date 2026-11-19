//! The pages of the public website, found the same way by every command
//! that reads them (`site-layout`, `check-site-links`), so they agree on
//! which files are pages and print them in the same order.

use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};

/// The published website, relative to the repository root; the Pages
/// workflow uploads this folder as it is.
pub(crate) const SITE_DIRECTORY: &str = "site";

/// Every `.html` file under the site, as `/`-separated paths from the site
/// root, sorted so output and errors read the same on every machine.
pub(crate) fn list_site_pages(site_directory: &Path) -> Result<Vec<String>, String> {
    let mut page_paths = Vec::new();
    collect_site_pages(site_directory, site_directory, &mut page_paths)?;
    page_paths.sort();
    if page_paths.is_empty() {
        return Err(format!("no pages found under {}", site_directory.display()));
    }
    Ok(page_paths)
}

fn collect_site_pages(
    site_directory: &Path,
    folder: &Path,
    page_paths: &mut Vec<String>,
) -> Result<(), String> {
    let folder_entries = fs::read_dir(folder)
        .map_err(|read_error| format!("cannot list {}: {read_error}", folder.display()))?;
    for folder_entry in folder_entries {
        let entry_path: PathBuf = folder_entry
            .map_err(|read_error| format!("cannot list {}: {read_error}", folder.display()))?
            .path();
        if entry_path.is_dir() {
            collect_site_pages(site_directory, &entry_path, page_paths)?;
        } else if entry_path.extension() == Some(OsStr::new("html")) {
            page_paths.push(site_relative_path(site_directory, &entry_path));
        }
    }
    Ok(())
}

fn site_relative_path(site_directory: &Path, page_file: &Path) -> String {
    let relative_path = page_file.strip_prefix(site_directory).unwrap_or(page_file);
    let path_parts: Vec<String> = relative_path
        .components()
        .map(|component| component.as_os_str().to_string_lossy().into_owned())
        .collect();
    path_parts.join("/")
}
