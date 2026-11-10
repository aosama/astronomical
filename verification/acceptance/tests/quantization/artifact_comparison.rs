//! File-level comparison of two converted artifacts: every tensor's dtype,
//! shape and stored bytes, and the JSON documents key by key.
//!
//! The comparison reads the safetensors files directly instead of loading
//! them through MLX, so "identical" means identical bytes on disk, with no
//! array library in between to round or re-encode anything.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

/// One tensor as a safetensors file stores it.
pub(crate) struct StoredTensor {
    pub(crate) file_name: String,
    pub(crate) dtype: String,
    pub(crate) shape: Vec<u64>,
    pub(crate) bytes: Vec<u8>,
}

/// Every tensor in the top-level `*.safetensors` files of `directory`.
pub(crate) fn read_stored_tensors(directory: &Path) -> BTreeMap<String, StoredTensor> {
    let mut stored_tensors = BTreeMap::new();
    let directory_entries =
        std::fs::read_dir(directory).expect("the artifact directory is readable");
    for directory_entry in directory_entries.flatten() {
        let file_path = directory_entry.path();
        if file_path.extension().and_then(std::ffi::OsStr::to_str) != Some("safetensors") {
            continue;
        }
        let file_name = directory_entry.file_name().to_string_lossy().into_owned();
        let file_bytes = std::fs::read(&file_path).expect("the safetensors file is readable");
        let header_length_bytes: [u8; 8] = file_bytes[..8].try_into().expect("eight length bytes");
        let data_start = 8 + usize::try_from(u64::from_le_bytes(header_length_bytes))
            .expect("the header length fits in memory");
        let header: serde_json::Map<String, serde_json::Value> =
            serde_json::from_slice(&file_bytes[8..data_start]).expect("the header parses");
        for (tensor_name, tensor_entry) in &header {
            if tensor_name == "__metadata__" {
                continue;
            }
            let stored_tensor =
                stored_tensor_from(&file_name, tensor_entry, &file_bytes[data_start..]);
            stored_tensors.insert(tensor_name.clone(), stored_tensor);
        }
    }
    stored_tensors
}

fn stored_tensor_from(
    file_name: &str,
    tensor_entry: &serde_json::Value,
    data_bytes: &[u8],
) -> StoredTensor {
    let mut shape = Vec::new();
    for dimension in tensor_entry["shape"].as_array().expect("a shape array") {
        shape.push(dimension.as_u64().expect("a dimension"));
    }
    StoredTensor {
        file_name: file_name.to_owned(),
        dtype: tensor_entry["dtype"].as_str().expect("a dtype").to_owned(),
        shape,
        bytes: data_bytes[data_offset(tensor_entry, 0)..data_offset(tensor_entry, 1)].to_vec(),
    }
}

/// The start (position 0) or end (position 1) of a tensor's bytes.
fn data_offset(tensor_entry: &serde_json::Value, offset_position: usize) -> usize {
    let offset = tensor_entry["data_offsets"][offset_position]
        .as_u64()
        .expect("a data offset");
    usize::try_from(offset).expect("the offset fits in memory")
}

/// Every way the candidate's tensors differ from the reference's: names
/// present on one side only, and a dtype, shape, file or byte mismatch.
pub(crate) fn tensor_differences(
    reference_directory: &Path,
    candidate_directory: &Path,
) -> Vec<String> {
    let reference_tensors = read_stored_tensors(reference_directory);
    let candidate_tensors = read_stored_tensors(candidate_directory);
    let mut differences = Vec::new();
    let mut tensor_names = BTreeSet::new();
    tensor_names.extend(reference_tensors.keys().cloned());
    tensor_names.extend(candidate_tensors.keys().cloned());
    for tensor_name in tensor_names {
        match (
            reference_tensors.get(&tensor_name),
            candidate_tensors.get(&tensor_name),
        ) {
            (Some(reference), Some(candidate)) => {
                if let Some(difference) = stored_tensor_difference(reference, candidate) {
                    differences.push(format!("{tensor_name}: {difference}"));
                }
            }
            (Some(_), None) => differences.push(format!("{tensor_name}: missing in the candidate")),
            (None, Some(_)) => differences.push(format!("{tensor_name}: only in the candidate")),
            (None, None) => {}
        }
    }
    differences
}

fn stored_tensor_difference(reference: &StoredTensor, candidate: &StoredTensor) -> Option<String> {
    if reference.file_name != candidate.file_name {
        return Some(format!(
            "stored in {} vs {}",
            reference.file_name, candidate.file_name
        ));
    }
    if reference.dtype != candidate.dtype || reference.shape != candidate.shape {
        return Some(format!(
            "{} {:?} vs {} {:?}",
            reference.dtype, reference.shape, candidate.dtype, candidate.shape
        ));
    }
    if reference.bytes != candidate.bytes {
        return Some("stored bytes differ".to_owned());
    }
    None
}

/// Every way two JSON documents differ, as `path: reference != candidate`
/// lines; a key named in `ignored_keys` is skipped wherever it appears.
pub(crate) fn json_differences(
    reference: &serde_json::Value,
    candidate: &serde_json::Value,
    ignored_keys: &[&str],
) -> Vec<String> {
    let mut differences = Vec::new();
    collect_json_differences("", reference, candidate, ignored_keys, &mut differences);
    differences
}

fn collect_json_differences(
    path: &str,
    reference: &serde_json::Value,
    candidate: &serde_json::Value,
    ignored_keys: &[&str],
    differences: &mut Vec<String>,
) {
    let (Some(reference_entries), Some(candidate_entries)) =
        (reference.as_object(), candidate.as_object())
    else {
        if !json_scalars_agree(reference, candidate) {
            differences.push(format!("{path}: {reference} != {candidate}"));
        }
        return;
    };
    let mut keys = BTreeSet::new();
    keys.extend(reference_entries.keys());
    keys.extend(candidate_entries.keys());
    for key in keys {
        if ignored_keys.contains(&key.as_str()) {
            continue;
        }
        let entry_path = format!("{path}.{key}");
        match (reference_entries.get(key), candidate_entries.get(key)) {
            (Some(reference_entry), Some(candidate_entry)) => collect_json_differences(
                &entry_path,
                reference_entry,
                candidate_entry,
                ignored_keys,
                differences,
            ),
            (Some(_), None) => differences.push(format!("{entry_path}: missing in the candidate")),
            _ => differences.push(format!("{entry_path}: only in the candidate")),
        }
    }
}

/// Numbers agree to one part in a billion: the two tools print the same
/// double through different formatters, and `5.0` and `5` are one value.
fn json_scalars_agree(reference: &serde_json::Value, candidate: &serde_json::Value) -> bool {
    if let (Some(reference_number), Some(candidate_number)) =
        (reference.as_f64(), candidate.as_f64())
    {
        let tolerance = 1e-9 * reference_number.abs().max(1.0);
        return (reference_number - candidate_number).abs() <= tolerance;
    }
    reference == candidate
}

/// A JSON document under an artifact directory.
pub(crate) fn read_json_document(
    artifact_directory: &Path,
    relative_path: &str,
) -> serde_json::Value {
    let document_path = artifact_directory.join(relative_path);
    let document_text = match std::fs::read_to_string(&document_path) {
        Ok(document_text) => document_text,
        Err(read_error) => panic!("{} is unreadable: {read_error}", document_path.display()),
    };
    match serde_json::from_str(&document_text) {
        Ok(document) => document,
        Err(parse_error) => panic!("{} does not parse: {parse_error}", document_path.display()),
    }
}
