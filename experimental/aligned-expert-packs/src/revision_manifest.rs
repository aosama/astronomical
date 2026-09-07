//! Revision-level identity, completeness, and standalone integrity for format 3.

use std::{
    fs,
    io::{Read, Write},
    os::unix::fs::OpenOptionsExt,
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::aligned_expert_pack::AlignedExpertPackError;
use crate::per_expert_pack::PER_EXPERT_PACK_FORMAT_VERSION;

pub(crate) const MANIFEST_FILE_NAME: &str = "manifest.json";

/// One always-resident file declared by a streaming-model revision.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct StreamingModelResidentFile {
    pub file_name: String,
    pub expected_file_byte_count: u64,
    pub content_sha256: String,
}

/// One expert file declared by a streaming-model revision.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct StreamingModelExpertFile {
    pub layer_index: usize,
    pub expert_id: usize,
    pub file_name: String,
    pub expected_file_byte_count: u64,
    pub content_sha256: String,
}

/// Complete inventory for one independently loadable streaming-model revision.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct StreamingModelManifest {
    pub format_version: u32,
    pub model_id: String,
    pub source_model_id: String,
    pub model_revision: String,
    pub expert_capacity: usize,
    pub quantization_mode: String,
    pub quantization_bits: i32,
    pub quantization_group_size: i32,
    pub resident_files: Vec<StreamingModelResidentFile>,
    pub expert_files: Vec<StreamingModelExpertFile>,
}

impl StreamingModelManifest {
    /// Builds a manifest for one streaming-model identity.
    pub fn new(
        model_id: impl Into<String>,
        source_model_id: impl Into<String>,
        model_revision: impl Into<String>,
        expert_capacity: usize,
        quantization_mode: impl Into<String>,
        quantization_bits: i32,
        quantization_group_size: i32,
        resident_files: Vec<StreamingModelResidentFile>,
        expert_files: Vec<StreamingModelExpertFile>,
    ) -> Self {
        Self {
            format_version: PER_EXPERT_PACK_FORMAT_VERSION,
            model_id: model_id.into(),
            source_model_id: source_model_id.into(),
            model_revision: model_revision.into(),
            expert_capacity,
            quantization_mode: quantization_mode.into(),
            quantization_bits,
            quantization_group_size,
            resident_files,
            expert_files,
        }
    }

    /// Writes the manifest atomically beside the revision files.
    pub fn write_to_revision_directory(
        &self,
        revision_directory: &Path,
    ) -> Result<(), AlignedExpertPackError> {
        let manifest_path = revision_directory.join(MANIFEST_FILE_NAME);
        let serialized_manifest = serde_json::to_vec_pretty(self)?;
        let mut manifest_file = fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .mode(0o644)
            .open(&manifest_path)?;
        manifest_file.write_all(&serialized_manifest)?;
        manifest_file.sync_all()?;
        Ok(())
    }

    /// Reads and parses one revision manifest.
    pub fn read_from_revision_directory(
        revision_directory: &Path,
    ) -> Result<Self, AlignedExpertPackError> {
        let manifest_bytes = fs::read(revision_directory.join(MANIFEST_FILE_NAME))?;
        let streaming_model_manifest: Self = serde_json::from_slice(&manifest_bytes)?;
        if streaming_model_manifest.format_version != PER_EXPERT_PACK_FORMAT_VERSION {
            return Err(AlignedExpertPackError::UnsupportedFormatVersion {
                actual_format_version: streaming_model_manifest.format_version,
            });
        }
        Ok(streaming_model_manifest)
    }
}

/// SHA-256 of one complete file, used as the standalone integrity gate.
pub fn sha256_hex_digest(file_path: &Path) -> Result<String, AlignedExpertPackError> {
    let mut source_file = fs::File::open(file_path)?;
    let mut digest = Sha256::new();
    let mut hash_scratch_bytes = vec![0_u8; 64 * 1024];
    loop {
        let bytes_read = source_file.read(&mut hash_scratch_bytes)?;
        if bytes_read == 0 {
            break;
        }
        digest.update(&hash_scratch_bytes[..bytes_read]);
    }
    Ok(hex_encode(digest.finalize().into()))
}

/// Records one resident file after it exists on disk.
pub fn resident_file_entry(
    revision_directory: &Path,
    relative_file_name: &str,
) -> Result<StreamingModelResidentFile, AlignedExpertPackError> {
    let file_path = revision_directory.join(relative_file_name);
    let expected_file_byte_count = fs::metadata(&file_path)?.len();
    Ok(StreamingModelResidentFile {
        file_name: relative_file_name.to_owned(),
        expected_file_byte_count,
        content_sha256: sha256_hex_digest(&file_path)?,
    })
}

/// Records one expert file after it exists on disk.
pub fn expert_file_entry(
    revision_directory: &Path,
    layer_index: usize,
    expert_id: usize,
    relative_file_name: PathBuf,
) -> Result<StreamingModelExpertFile, AlignedExpertPackError> {
    let file_path = revision_directory.join(&relative_file_name);
    let expected_file_byte_count = fs::metadata(&file_path)?.len();
    Ok(StreamingModelExpertFile {
        layer_index,
        expert_id,
        file_name: relative_file_name.to_string_lossy().into_owned(),
        expected_file_byte_count,
        content_sha256: sha256_hex_digest(&file_path)?,
    })
}

fn hex_encode(digest_bytes: [u8; 32]) -> String {
    const HEX_DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut hex_digest = String::with_capacity(64);
    for digest_byte in digest_bytes {
        hex_digest.push(HEX_DIGITS[(digest_byte >> 4) as usize] as char);
        hex_digest.push(HEX_DIGITS[(digest_byte & 0x0f) as usize] as char);
    }
    hex_digest
}
