//! The filesystem edge of `cargo xtask check-duplication`: finds the measured
//! Rust sources and reads them into the two corpora. Every decision about a
//! path or a line is made by the pure modules beside this one.

use std::fs;
use std::path::Path;

use super::progress::ReadProgress;
use super::repeated_blocks::NormalizedSourceFile;
use super::source_classification::{area_of, is_production_source};
use crate::workspace_source_walk::{collect_authored_file_paths, is_rust_source_file_name};

/// The production and test sources, each normalized, in path order.
pub(crate) struct MeasuredCorpora {
    pub(crate) production_files: Vec<NormalizedSourceFile>,
    pub(crate) test_files: Vec<NormalizedSourceFile>,
}

/// Every Rust source under the authored roots (see `workspace_source_walk`),
/// as repository-relative paths in a stable order.
pub(crate) fn collect_measured_source_paths(repository_root: &Path) -> Vec<String> {
    collect_authored_file_paths(repository_root, is_rust_source_file_name)
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
