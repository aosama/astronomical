//! A record shipped inside each built model, stating exactly what it was made from
//! and the hash of every file, so anyone can check or repeat the build.
//!
//! It is published, so it holds repository names and commits only, and never a path
//! from the machine that built it. It has no timestamp, so the same inputs produce
//! the same record.

use std::fs::File;
use std::io::Read;
use std::path::Path;

use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::base_matching::{BaseMatchReport, BaseMatchVerdict};
use crate::error::VisionTowerError;

pub const BUILD_RECORD_FILE_NAME: &str = "astronomical-build.json";
const BUILD_RECORD_FORMAT_VERSION: u32 = 1;
const HASH_READ_BYTES: usize = 8 * 1024 * 1024;

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct BuildRecord {
    pub format_version: u32,
    pub built_with: String,
    pub text_model: SourceRecord,
    pub vision_tower: SourceRecord,
    pub base_match: BaseMatchRecord,
    pub files: Vec<FileRecord>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct SourceRecord {
    pub repository: Option<String>,
    pub revision: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct BaseMatchRecord {
    pub verdict: BaseMatchVerdict,
    pub compared_tensor_count: usize,
    pub shared_tensor_count: usize,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct FileRecord {
    pub path: String,
    pub byte_count: u64,
    pub sha256: String,
}

impl BuildRecord {
    #[must_use]
    pub fn new(
        text_model: SourceRecord,
        vision_tower: SourceRecord,
        base_match: &BaseMatchReport,
        files: Vec<FileRecord>,
    ) -> Self {
        let mut shared_tensor_count = 0_usize;
        for compared_tensor in &base_match.compared_tensors {
            if compared_tensor.agreement.is_shared() {
                shared_tensor_count += 1;
            }
        }
        Self {
            format_version: BUILD_RECORD_FORMAT_VERSION,
            built_with: concat!("astronomical vision add ", env!("CARGO_PKG_VERSION")).to_owned(),
            text_model,
            vision_tower,
            base_match: BaseMatchRecord {
                verdict: base_match.verdict,
                compared_tensor_count: base_match.compared_tensors.len(),
                shared_tensor_count,
            },
            files,
        }
    }
}

/// `(sha256 as lowercase hex, byte count)`, streamed so a many-gigabyte weight file
/// needs only one small read at a time in memory.
pub fn hash_file(file_path: &Path) -> Result<(String, u64), VisionTowerError> {
    let mut opened_file = match File::open(file_path) {
        Ok(opened_file) => opened_file,
        Err(read_error) => return Err(unreadable(file_path, &read_error)),
    };
    let mut hasher = Sha256::new();
    let mut read_bytes = vec![0_u8; HASH_READ_BYTES];
    let mut byte_count = 0_u64;
    loop {
        let read_count = match opened_file.read(&mut read_bytes) {
            Ok(read_count) => read_count,
            Err(read_error) => return Err(unreadable(file_path, &read_error)),
        };
        if read_count == 0 {
            break;
        }
        hasher.update(&read_bytes[..read_count]);
        byte_count += read_count as u64;
    }
    Ok((
        astronomical_lowercase_hex::encode_lowercase_hex(&hasher.finalize()),
        byte_count,
    ))
}

fn unreadable(path: &Path, read_error: &std::io::Error) -> VisionTowerError {
    VisionTowerError::FileUnreadable {
        path: path.to_path_buf(),
        cause: read_error.to_string(),
    }
}

#[cfg(test)]
#[path = "build_record_tests.rs"]
mod tests;
