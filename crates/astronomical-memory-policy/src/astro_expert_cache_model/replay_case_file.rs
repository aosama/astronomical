//! Reading a recorded reply the engine probe saved: `<workload>.routes`, the
//! routes in the recording format, beside `<workload>.case.json`, what the
//! `astro_expert_cache` held when the reply began and the capacity it had.

use std::path::{Path, PathBuf};

use serde::Deserialize;

use super::{ReplayCase, RoutedReply, RoutedReplyDecodeError};

/// Why a saved case could not be read.
#[derive(Debug, thiserror::Error)]
pub enum ReplayCaseFileError {
    #[error("{path} could not be read: {source}")]
    Unreadable {
        path: String,
        source: std::io::Error,
    },
    #[error("{path} is not a case document: {source}")]
    NotACaseDocument {
        path: String,
        source: serde_json::Error,
    },
    #[error("{path} is not a routes recording: {source}")]
    NotARecording {
        path: String,
        source: RoutedReplyDecodeError,
    },
}

#[derive(Deserialize)]
struct CaseDocument {
    kept_expert_capacity_bytes: u64,
    expert_payload_bytes_by_layer: Vec<u64>,
    held_expert_ids_by_layer_at_reply_start: Vec<Vec<u16>>,
    #[serde(default)]
    held_expert_ids_by_layer_at_reply_end: Vec<Vec<u16>>,
    #[serde(default)]
    warm_slot_count: Option<usize>,
}

/// One saved case: its workload name, its replay case, and what each layer
/// held after the reply (empty for a recording that did not keep it).
#[derive(Clone, Debug)]
pub struct SavedReplayCase {
    pub workload_name: String,
    pub replay_case: ReplayCase,
    pub held_expert_ids_by_layer_at_end: Vec<Vec<u16>>,
    /// The engine's warm slot count when the reply was recorded, when kept.
    pub warm_slot_count: Option<usize>,
}

/// Reads the case whose document is `case_document_path`; its routes are
/// the `.routes` file of the same workload beside it.
///
/// # Errors
///
/// Returns which file could not be read or understood.
pub fn read_saved_replay_case(
    case_document_path: &Path,
) -> Result<SavedReplayCase, ReplayCaseFileError> {
    let case_document_bytes = read_file(case_document_path)?;
    let case_document: CaseDocument =
        serde_json::from_slice(&case_document_bytes).map_err(|source| {
            ReplayCaseFileError::NotACaseDocument {
                path: case_document_path.display().to_string(),
                source,
            }
        })?;
    let workload_name = workload_name_of(case_document_path);
    let routes_path = case_document_path.with_file_name(format!("{workload_name}.routes"));
    let routed_reply =
        RoutedReply::from_recording_bytes(&read_file(&routes_path)?).map_err(|source| {
            ReplayCaseFileError::NotARecording {
                path: routes_path.display().to_string(),
                source,
            }
        })?;
    Ok(SavedReplayCase {
        workload_name,
        held_expert_ids_by_layer_at_end: case_document.held_expert_ids_by_layer_at_reply_end,
        warm_slot_count: case_document.warm_slot_count,
        replay_case: ReplayCase {
            routed_reply,
            expert_payload_bytes_by_layer: case_document.expert_payload_bytes_by_layer,
            capacity_bytes: case_document.kept_expert_capacity_bytes,
            held_expert_ids_by_layer_at_start: case_document
                .held_expert_ids_by_layer_at_reply_start,
            layer_slot_limits: None,
            pinned_layer_indexes: Vec::new(),
        },
    })
}

/// Every `*.case.json` under `evidence_directory`, at any depth, sorted.
///
/// # Errors
///
/// Returns the first directory that could not be listed.
pub fn find_saved_case_documents(
    evidence_directory: &Path,
) -> Result<Vec<PathBuf>, ReplayCaseFileError> {
    let mut case_document_paths = Vec::new();
    let mut directories_to_visit = vec![evidence_directory.to_path_buf()];
    while let Some(directory) = directories_to_visit.pop() {
        let directory_entries =
            std::fs::read_dir(&directory).map_err(|source| ReplayCaseFileError::Unreadable {
                path: directory.display().to_string(),
                source,
            })?;
        for directory_entry in directory_entries.flatten() {
            let entry_path = directory_entry.path();
            if entry_path.is_dir() {
                directories_to_visit.push(entry_path);
            } else if entry_path.to_string_lossy().ends_with(".case.json") {
                case_document_paths.push(entry_path);
            }
        }
    }
    case_document_paths.sort();
    Ok(case_document_paths)
}

fn read_file(path: &Path) -> Result<Vec<u8>, ReplayCaseFileError> {
    std::fs::read(path).map_err(|source| ReplayCaseFileError::Unreadable {
        path: path.display().to_string(),
        source,
    })
}

fn workload_name_of(case_document_path: &Path) -> String {
    let file_name = match case_document_path.file_name() {
        Some(file_name) => file_name.to_string_lossy().into_owned(),
        None => String::new(),
    };
    file_name
        .strip_suffix(".case.json")
        .unwrap_or(&file_name)
        .to_owned()
}
