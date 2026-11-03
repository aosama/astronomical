//! Per-file SHA-256 manifests of a header tree, byte-identical to what
//! `find . -type f -print0 | LC_ALL=C sort -z | xargs -0 shasum -a 256`
//! printed in the shell pipeline. Matching that format exactly means an
//! extraction published by the shell port still verifies as cached here,
//! so switching implementations never forces a re-extraction.

use std::path::Path;

use astronomical_native_identity::digest_bytes;

/// One `"<sha256>  ./relative/path"` line per regular file, sorted by the
/// path's bytes. Symlinks and directories are skipped, as `find -type f`
/// skipped them.
pub(crate) fn tree_hash_manifest(tree_directory: &Path) -> Result<String, String> {
    let mut relative_file_paths = Vec::new();
    collect_regular_files(tree_directory, ".", &mut relative_file_paths)?;
    relative_file_paths.sort();
    let mut manifest_text = String::new();
    for relative_file_path in relative_file_paths {
        let file_bytes = match std::fs::read(tree_directory.join(&relative_file_path[2..])) {
            Ok(file_bytes) => file_bytes,
            Err(read_error) => {
                return Err(format!("could not hash {relative_file_path}: {read_error}"));
            }
        };
        manifest_text.push_str(&format!(
            "{}  {relative_file_path}\n",
            digest_bytes(&file_bytes)
        ));
    }
    Ok(manifest_text)
}

pub(crate) fn count_regular_files(tree_directory: &Path) -> Result<usize, String> {
    let mut relative_file_paths = Vec::new();
    collect_regular_files(tree_directory, ".", &mut relative_file_paths)?;
    Ok(relative_file_paths.len())
}

fn collect_regular_files(
    directory_path: &Path,
    relative_prefix: &str,
    relative_file_paths: &mut Vec<String>,
) -> Result<(), String> {
    let directory_entries = match std::fs::read_dir(directory_path) {
        Ok(directory_entries) => directory_entries,
        Err(read_error) => {
            return Err(format!(
                "could not list {}: {read_error}",
                directory_path.display()
            ));
        }
    };
    for directory_entry in directory_entries {
        let directory_entry = match directory_entry {
            Ok(directory_entry) => directory_entry,
            Err(entry_error) => {
                return Err(format!(
                    "could not read an entry of {}: {entry_error}",
                    directory_path.display()
                ));
            }
        };
        let entry_relative_path = format!(
            "{relative_prefix}/{}",
            directory_entry.file_name().to_string_lossy()
        );
        let Ok(entry_type) = directory_entry.file_type() else {
            continue;
        };
        if entry_type.is_dir() {
            collect_regular_files(
                &directory_entry.path(),
                &entry_relative_path,
                relative_file_paths,
            )?;
        } else if entry_type.is_file() {
            relative_file_paths.push(entry_relative_path);
        }
    }
    Ok(())
}

/// Paths that differ between two manifests, labelled as the shell reported
/// them: `modified`, `built-only`, or `extracted-only`.
pub(crate) fn manifest_differences(extracted_manifest: &str, built_manifest: &str) -> Vec<String> {
    let extracted_entries = manifest_entries(extracted_manifest);
    let built_entries = manifest_entries(built_manifest);
    let mut differing_paths = Vec::new();
    for (built_path, built_digest) in &built_entries {
        match extracted_entries.get(built_path) {
            Some(extracted_digest) if extracted_digest != built_digest => {
                differing_paths.push(format!("modified {built_path}"));
            }
            Some(_) => {}
            None => differing_paths.push(format!("built-only {built_path}")),
        }
    }
    for extracted_path in extracted_entries.keys() {
        if !built_entries.contains_key(extracted_path) {
            differing_paths.push(format!("extracted-only {extracted_path}"));
        }
    }
    differing_paths
}

fn manifest_entries(manifest_text: &str) -> std::collections::BTreeMap<String, String> {
    let mut path_to_digest = std::collections::BTreeMap::new();
    for manifest_line in manifest_text.lines() {
        if let Some((file_digest, relative_path)) = manifest_line.split_once("  ") {
            path_to_digest.insert(relative_path.to_owned(), file_digest.to_owned());
        }
    }
    path_to_digest
}
