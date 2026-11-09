//! Turns the untrusted entries of a recursive Hub tree into a manifest.
//!
//! Every entry is checked as it arrives: its own shape, its path against
//! every path seen before (case-insensitive collisions, a file that is also
//! a directory), and the running totals against the job's limits. The
//! accumulator is synchronous, so these rules are tested without HTTP.

use std::collections::BTreeSet;

use serde::Deserialize;

use super::{
    HubManifestFile, HuggingFaceHubError, HuggingFaceHubLimits, HuggingFaceManifest,
    MAXIMUM_JAVASCRIPT_SAFE_INTEGER,
};
use crate::library::download::job::download_job::MAXIMUM_DOWNLOAD_JOB_BYTES;
use crate::library::download::transfer::file_digest::is_lowercase_hex;
use crate::library::hub::hugging_face_bounds::{
    estimated_durable_file_metadata_bytes, has_ancestor_path, has_descendant_path,
    is_canonical_ascii_path,
};
use crate::library::{DownloadFileDigest, DownloadPathSelection};

const GIT_SHA1_HEX_CHARACTER_COUNT: usize = 40;
const SHA256_HEX_CHARACTER_COUNT: usize = 64;
/// Durable job metadata reserved before any file is listed.
const DOWNLOAD_JOB_FIXED_METADATA_ALLOWANCE_BYTES: usize = 4_096;

#[derive(Debug, Deserialize)]
pub(super) struct TreeEntryDocument {
    #[serde(rename = "type")]
    entry_type: String,
    size: u64,
    path: String,
    oid: Option<String>,
    lfs: Option<LfsDocument>,
    #[serde(rename = "xetHash")]
    xet_hash: Option<String>,
}

#[derive(Debug, Deserialize)]
struct LfsDocument {
    oid: String,
    size: u64,
    #[serde(rename = "pointerSize")]
    pointer_size: u64,
}

/// The manifest assembled from the tree entries accepted so far.
pub(super) struct TreeManifestAccumulator {
    limits: HuggingFaceHubLimits,
    fetched_tree_entry_count: usize,
    files: Vec<HubManifestFile>,
    /// Every path seen, lowercased: the tree must survive a
    /// case-insensitive file system such as APFS's default.
    normalized_paths: BTreeSet<String>,
    normalized_file_paths: BTreeSet<String>,
    total_bytes: u64,
    estimated_job_metadata_bytes: usize,
}

impl TreeManifestAccumulator {
    pub(super) fn new(limits: HuggingFaceHubLimits) -> Self {
        Self {
            limits,
            fetched_tree_entry_count: 0,
            files: Vec::new(),
            normalized_paths: BTreeSet::new(),
            normalized_file_paths: BTreeSet::new(),
            total_bytes: 0,
            estimated_job_metadata_bytes: DOWNLOAD_JOB_FIXED_METADATA_ALLOWANCE_BYTES,
        }
    }

    /// Checks one entry against the tree so far and keeps it when it is a
    /// selected file.
    pub(super) fn accept_entry(
        &mut self,
        tree_entry: TreeEntryDocument,
        path_selection: &DownloadPathSelection,
    ) -> Result<(), HuggingFaceHubError> {
        self.fetched_tree_entry_count = self.fetched_tree_entry_count.saturating_add(1);
        if self.fetched_tree_entry_count > self.limits.maximum_file_count {
            return Err(HuggingFaceHubError::TooManyTreeEntries);
        }
        validate_tree_entry_common(&tree_entry, self.limits.maximum_path_bytes)?;
        let normalized_path = tree_entry.path.to_ascii_lowercase();
        if has_ancestor_path(&self.normalized_file_paths, &normalized_path) {
            return Err(HuggingFaceHubError::FilePathHierarchyConflict);
        }
        if !self.normalized_paths.insert(normalized_path.clone()) {
            return Err(HuggingFaceHubError::CaseInsensitivePathCollision);
        }
        match tree_entry.entry_type.as_str() {
            "directory" => validate_directory_entry(&tree_entry),
            "file" => {
                if has_descendant_path(&self.normalized_paths, &normalized_path) {
                    return Err(HuggingFaceHubError::FilePathHierarchyConflict);
                }
                if path_selection.includes(&tree_entry.path) {
                    self.keep_selected_file(tree_entry)?;
                }
                self.normalized_file_paths.insert(normalized_path);
                Ok(())
            }
            _ => Err(HuggingFaceHubError::InvalidEntryType),
        }
    }

    /// Adds a selected file while keeping the byte total JavaScript-safe and
    /// the durable job record within its size limit.
    fn keep_selected_file(
        &mut self,
        tree_entry: TreeEntryDocument,
    ) -> Result<(), HuggingFaceHubError> {
        if self.files.len() >= self.limits.maximum_file_count {
            return Err(HuggingFaceHubError::TooManyFiles);
        }
        self.total_bytes = self
            .total_bytes
            .checked_add(tree_entry.size)
            .ok_or(HuggingFaceHubError::TotalBytesTooLarge)?;
        if self.total_bytes > self.limits.maximum_total_bytes
            || self.total_bytes > MAXIMUM_JAVASCRIPT_SAFE_INTEGER
        {
            return Err(HuggingFaceHubError::TotalBytesTooLarge);
        }
        let manifest_file = validated_file(tree_entry)?;
        self.estimated_job_metadata_bytes = self
            .estimated_job_metadata_bytes
            .checked_add(estimated_durable_file_metadata_bytes(&manifest_file))
            .ok_or(HuggingFaceHubError::JobMetadataTooLarge)?;
        let job_metadata_limit_bytes = self
            .limits
            .maximum_job_metadata_bytes
            .min(MAXIMUM_DOWNLOAD_JOB_BYTES);
        if self.estimated_job_metadata_bytes > job_metadata_limit_bytes {
            return Err(HuggingFaceHubError::JobMetadataTooLarge);
        }
        self.files.push(manifest_file);
        Ok(())
    }

    /// The manifest, refused when nothing downloadable was selected.
    pub(super) fn finish(
        self,
        repository_id: &str,
        revision: &str,
    ) -> Result<HuggingFaceManifest, HuggingFaceHubError> {
        if self.files.is_empty() || self.total_bytes == 0 {
            return Err(HuggingFaceHubError::EmptyManifest);
        }
        Ok(HuggingFaceManifest {
            repository_id: repository_id.to_owned(),
            revision: revision.to_owned(),
            files: self.files,
            total_bytes: self.total_bytes,
        })
    }
}

fn validated_file(tree_entry: TreeEntryDocument) -> Result<HubManifestFile, HuggingFaceHubError> {
    let xet_hash = match tree_entry.xet_hash {
        Some(xet_hash) if is_lowercase_hex(&xet_hash, SHA256_HEX_CHARACTER_COUNT) => Some(xet_hash),
        Some(_) => return Err(HuggingFaceHubError::InvalidXetHash),
        None => None,
    };
    let digest = if let Some(lfs) = tree_entry.lfs {
        if lfs.size != tree_entry.size
            || lfs.pointer_size == 0
            || lfs.pointer_size > MAXIMUM_JAVASCRIPT_SAFE_INTEGER
            || !is_lowercase_hex(&lfs.oid, SHA256_HEX_CHARACTER_COUNT)
        {
            return Err(HuggingFaceHubError::InvalidLfsMetadata);
        }
        if let Some(git_oid) = tree_entry.oid.as_deref()
            && !is_lowercase_hex(git_oid, GIT_SHA1_HEX_CHARACTER_COUNT)
        {
            return Err(HuggingFaceHubError::InvalidGitBlobSha1);
        }
        DownloadFileDigest::Sha256(lfs.oid)
    } else {
        let git_oid = tree_entry
            .oid
            .ok_or(HuggingFaceHubError::MissingIndependentDigest)?;
        if !is_lowercase_hex(&git_oid, GIT_SHA1_HEX_CHARACTER_COUNT) {
            return Err(HuggingFaceHubError::InvalidGitBlobSha1);
        }
        DownloadFileDigest::GitBlobSha1(git_oid)
    };
    Ok(HubManifestFile {
        relative_path: tree_entry.path,
        expected_bytes: tree_entry.size,
        digest,
        xet_hash,
    })
}

fn validate_tree_entry_common(
    tree_entry: &TreeEntryDocument,
    maximum_path_bytes: usize,
) -> Result<(), HuggingFaceHubError> {
    if tree_entry.size > MAXIMUM_JAVASCRIPT_SAFE_INTEGER {
        return Err(HuggingFaceHubError::InvalidEntrySize);
    }
    if !is_canonical_ascii_path(&tree_entry.path, maximum_path_bytes) {
        return Err(HuggingFaceHubError::InvalidEntryPath);
    }
    Ok(())
}

fn validate_directory_entry(tree_entry: &TreeEntryDocument) -> Result<(), HuggingFaceHubError> {
    if tree_entry.size != 0 {
        return Err(HuggingFaceHubError::InvalidEntrySize);
    }
    if tree_entry.lfs.is_some() || tree_entry.xet_hash.is_some() {
        return Err(HuggingFaceHubError::InvalidLfsMetadata);
    }
    if let Some(git_oid) = tree_entry.oid.as_deref()
        && !is_lowercase_hex(git_oid, GIT_SHA1_HEX_CHARACTER_COUNT)
    {
        return Err(HuggingFaceHubError::InvalidGitBlobSha1);
    }
    Ok(())
}

#[cfg(test)]
#[path = "tree_manifest_tests.rs"]
mod tree_manifest_tests;
