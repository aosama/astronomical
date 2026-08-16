use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(super) enum ConcreteModelFamily {
    Qwen,
    Laguna,
    DeepSeekV4,
}

/// Returns every Rust source beneath a package in deterministic path order.
pub(super) fn rust_source_files_recursively(source_directory: &Path) -> Vec<PathBuf> {
    let mut pending_directories = vec![source_directory.to_path_buf()];
    let mut source_files = Vec::new();
    while let Some(pending_directory) = pending_directories.pop() {
        let mut entries = fs::read_dir(&pending_directory)
            .expect("source directory must be readable")
            .map(|entry| entry.expect("source entry must be readable").path())
            .collect::<Vec<_>>();
        entries.sort();
        for entry_path in entries {
            if entry_path.is_dir() {
                pending_directories.push(entry_path);
            } else if entry_path
                .extension()
                .and_then(|extension| extension.to_str())
                == Some("rs")
            {
                source_files.push(entry_path);
            }
        }
    }
    source_files.sort();
    source_files
}

/// Finds concrete-family identifiers while ignoring comments and quoted fixtures.
pub(super) fn families_mentioned_by_source(source_file: &Path) -> BTreeSet<ConcreteModelFamily> {
    let source = fs::read_to_string(source_file).expect("Rust source must be readable");
    rust_code_identifiers(&source)
        .into_iter()
        .filter_map(|identifier| {
            if identifier.starts_with("qwen3_5") || identifier.starts_with("Qwen3_5") {
                Some(ConcreteModelFamily::Qwen)
            } else if identifier.starts_with("laguna") || identifier.starts_with("Laguna") {
                Some(ConcreteModelFamily::Laguna)
            } else if identifier.starts_with("deepseek_v4") || identifier.starts_with("DeepSeekV4")
            {
                Some(ConcreteModelFamily::DeepSeekV4)
            } else {
                None
            }
        })
        .collect()
}

fn rust_code_identifiers(source: &str) -> Vec<String> {
    let source_bytes = source.as_bytes();
    let mut identifiers = Vec::new();
    let mut source_index = 0;
    let mut block_comment_depth = 0_usize;
    while source_index < source_bytes.len() {
        if block_comment_depth > 0 {
            if source_bytes[source_index..].starts_with(b"/*") {
                block_comment_depth += 1;
                source_index += 2;
            } else if source_bytes[source_index..].starts_with(b"*/") {
                block_comment_depth -= 1;
                source_index += 2;
            } else {
                source_index += 1;
            }
            continue;
        }
        if source_bytes[source_index..].starts_with(b"//") {
            source_index = source_bytes[source_index..]
                .iter()
                .position(|source_byte| *source_byte == b'\n')
                .map_or(source_bytes.len(), |line_length| {
                    source_index + line_length + 1
                });
            continue;
        }
        if source_bytes[source_index..].starts_with(b"/*") {
            block_comment_depth = 1;
            source_index += 2;
            continue;
        }
        if source_bytes[source_index] == b'"' {
            source_index += 1;
            while source_index < source_bytes.len() {
                if source_bytes[source_index] == b'\\' {
                    source_index = source_index.saturating_add(2);
                } else if source_bytes[source_index] == b'"' {
                    source_index += 1;
                    break;
                } else {
                    source_index += 1;
                }
            }
            continue;
        }
        if source_bytes[source_index].is_ascii_alphabetic() || source_bytes[source_index] == b'_' {
            let identifier_start = source_index;
            source_index += 1;
            while source_index < source_bytes.len()
                && (source_bytes[source_index].is_ascii_alphanumeric()
                    || source_bytes[source_index] == b'_')
            {
                source_index += 1;
            }
            identifiers.push(
                String::from_utf8_lossy(&source_bytes[identifier_start..source_index]).into_owned(),
            );
        } else {
            source_index += 1;
        }
    }
    identifiers
}
