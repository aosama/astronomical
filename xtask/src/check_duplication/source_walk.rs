//! The filesystem edge of `cargo xtask check-duplication`: finds the measured
//! Rust sources and reads them into the two corpora. Every decision about a
//! path or a line is made by the pure modules beside this one.

use std::fs;
use std::path::Path;

use super::progress::ReadProgress;
use super::repeated_blocks::NormalizedSourceFile;
use super::source_classification::{
    MEASURED_ROOT_DIRECTORIES, area_of, is_production_source, is_skipped_directory_name,
};

/// The production and test sources, each normalized, in path order.
pub(crate) struct MeasuredCorpora {
    pub(crate) production_files: Vec<NormalizedSourceFile>,
    pub(crate) test_files: Vec<NormalizedSourceFile>,
}

/// Every Rust source under the measured roots, as repository-relative paths
/// with `/` separators, sorted so a run reads files in the same order on
/// every machine.
pub(crate) fn collect_measured_source_paths(repository_root: &Path) -> Vec<String> {
    let mut relative_paths = Vec::new();
    for root_directory in MEASURED_ROOT_DIRECTORIES {
        collect_rust_sources(
            &repository_root.join(root_directory),
            root_directory,
            &mut relative_paths,
        );
    }
    relative_paths.sort();
    relative_paths
}

/// A directory that cannot be read is skipped rather than failing the run:
/// the measured roots hold only sources this repository tracks, and an
/// unreadable folder there would be noticed by every other check first.
fn collect_rust_sources(
    directory: &Path,
    relative_directory: &str,
    relative_paths: &mut Vec<String>,
) {
    let Ok(directory_entries) = fs::read_dir(directory) else {
        return;
    };
    for directory_entry in directory_entries.flatten() {
        let entry_name = directory_entry.file_name().to_string_lossy().into_owned();
        let relative_entry_path = format!("{relative_directory}/{entry_name}");
        let Ok(entry_type) = directory_entry.file_type() else {
            continue;
        };
        if entry_type.is_dir() {
            if !is_skipped_directory_name(&entry_name) {
                collect_rust_sources(
                    &directory_entry.path(),
                    &relative_entry_path,
                    relative_paths,
                );
            }
        } else if entry_name.ends_with(".rs") {
            relative_paths.push(relative_entry_path);
        }
    }
}

/// Reads and normalizes every source, printing progress every few seconds.
/// Bytes that are not UTF-8 are replaced rather than refused, so one odd
/// fixture cannot stop the measure.
pub(crate) fn read_measured_corpora(
    repository_root: &Path,
    relative_paths: &[String],
) -> Result<MeasuredCorpora, String> {
    let mut measured_corpora = MeasuredCorpora {
        production_files: Vec::new(),
        test_files: Vec::new(),
    };
    let mut read_progress = ReadProgress::start(relative_paths.len());
    for (path_index, relative_path) in relative_paths.iter().enumerate() {
        let source_bytes = fs::read(repository_root.join(relative_path)).map_err(|error| {
            format!("[check-duplication] status=failed reason=source-unreadable path={relative_path} error={error}")
        })?;
        let source_file = NormalizedSourceFile::from_source_text(
            relative_path.clone(),
            area_of(relative_path),
            &String::from_utf8_lossy(&source_bytes),
        );
        if is_production_source(relative_path) {
            measured_corpora.production_files.push(source_file);
        } else {
            measured_corpora.test_files.push(source_file);
        }
        read_progress.observe(path_index + 1);
    }
    Ok(measured_corpora)
}
