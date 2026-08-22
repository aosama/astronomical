//! Bounded immutable-revision manifest discovery from the public Hugging Face Hub.

use std::{collections::BTreeSet, sync::Arc};

use serde::Deserialize;
use thiserror::Error;

use super::{
    DownloadFileDigest, DownloadPathSelection,
    download_catalog::{is_valid_huggingface_id, is_valid_immutable_revision},
    download_file_digest::is_lowercase_hex,
    download_job::MAXIMUM_DOWNLOAD_JOB_BYTES,
    hub_transport::{HubHttpRequest, HubHttpResponse, HubTransport, HubTransportError},
    hugging_face_hub_bounds::{
        estimated_durable_file_metadata_bytes, has_ancestor_path, has_descendant_path,
        is_canonical_ascii_path, parse_next_link, validate_tree_page_url,
    },
};

const HUGGING_FACE_ORIGIN: &str = "https://huggingface.co";
const MAXIMUM_JAVASCRIPT_SAFE_INTEGER: u64 = 9_007_199_254_740_991;
const DEFAULT_MAXIMUM_TREE_PAGE_COUNT: usize = 128;
const DEFAULT_MAXIMUM_FILE_COUNT: usize = 65_536;
const DEFAULT_MAXIMUM_PATH_BYTES: usize = 1_024;
const GIT_SHA1_HEX_CHARACTER_COUNT: usize = 40;
const SHA256_HEX_CHARACTER_COUNT: usize = 64;
const DOWNLOAD_JOB_FIXED_METADATA_ALLOWANCE_BYTES: usize = 4_096;
/// Resource limits applied while converting untrusted Hub metadata into a manifest.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HuggingFaceHubLimits {
    maximum_tree_page_count: usize,
    maximum_file_count: usize,
    maximum_path_bytes: usize,
    maximum_total_bytes: u64,
    maximum_job_metadata_bytes: usize,
}
/// Public Hub metadata and recursive tree owner.
pub struct HuggingFaceHub {
    transport: Arc<dyn HubTransport>,
    limits: HuggingFaceHubLimits,
}
/// Validated immutable repository file inventory.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HuggingFaceManifest {
    repository_id: String,
    revision: String,
    files: Vec<HubManifestFile>,
    total_bytes: u64,
}
/// One downloadable regular file with an independently verifiable digest.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HubManifestFile {
    relative_path: String,
    expected_bytes: u64,
    digest: DownloadFileDigest,
    xet_hash: Option<String>,
}
/// Typed manifest retrieval or validation failure.
#[derive(Debug, Error)]
pub enum HuggingFaceHubError {
    #[error("Hugging Face transport failed: {0}")]
    Transport(#[from] HubTransportError),
    #[error("the repository requires authentication or gated access")]
    DownloadGated,
    #[error("Hugging Face returned HTTP status {status}")]
    UnexpectedStatus { status: u16 },
    #[error("the Hugging Face repository identity is invalid")]
    InvalidRepositoryId,
    #[error("the requested immutable revision is invalid")]
    InvalidRevision,
    #[error("Hugging Face metadata is invalid JSON: {0}")]
    InvalidMetadataJson(#[source] serde_json::Error),
    #[error("Hugging Face tree metadata is invalid JSON: {0}")]
    InvalidTreeJson(#[source] serde_json::Error),
    #[error("Hugging Face metadata did not resolve to the requested immutable revision")]
    RevisionMismatch,
    #[error("Hugging Face metadata identifies a different repository")]
    RepositoryMismatch,
    #[error("Hugging Face metadata has an invalid visibility or gating status")]
    InvalidVisibility,
    #[error("the recursive tree exceeds the page limit")]
    TooManyTreePages,
    #[error("the recursive tree exceeds the file limit")]
    TooManyFiles,
    #[error("the recursive tree exceeds the entry limit")]
    TooManyTreeEntries,
    #[error("a tree entry has an invalid type")]
    InvalidEntryType,
    #[error("a tree entry has an invalid size")]
    InvalidEntrySize,
    #[error("a tree entry has a noncanonical relative path")]
    InvalidEntryPath,
    #[error("the recursive tree contains a case-insensitive path collision")]
    CaseInsensitivePathCollision,
    #[error("the recursive tree contains a file and descendant path conflict")]
    FilePathHierarchyConflict,
    #[error("a tree entry has invalid large-file metadata")]
    InvalidLfsMetadata,
    #[error("a tree entry has an invalid Git blob SHA-1")]
    InvalidGitBlobSha1,
    #[error("a tree entry has an invalid Xet hash")]
    InvalidXetHash,
    #[error("a file has no independent downloadable-content digest")]
    MissingIndependentDigest,
    #[error("the manifest exceeds the total JavaScript-safe byte limit")]
    TotalBytesTooLarge,
    #[error("the manifest cannot fit within durable download-job metadata")]
    JobMetadataTooLarge,
    #[error("the repository tree has no nonempty downloadable model payload")]
    EmptyManifest,
    #[error("the Hub pagination Link header is malformed or unsafe")]
    UnsafePaginationLink,
}

#[derive(Debug, Deserialize)]
struct ModelMetadataDocument {
    id: Option<String>,
    sha: String,
    private: Option<bool>,
    gated: Option<GatedStatus>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum GatedStatus {
    Boolean(bool),
    Mode(String),
}

#[derive(Debug, Deserialize)]
struct TreeEntryDocument {
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

impl Default for HuggingFaceHubLimits {
    fn default() -> Self {
        Self {
            maximum_tree_page_count: DEFAULT_MAXIMUM_TREE_PAGE_COUNT,
            maximum_file_count: DEFAULT_MAXIMUM_FILE_COUNT,
            maximum_path_bytes: DEFAULT_MAXIMUM_PATH_BYTES,
            maximum_total_bytes: MAXIMUM_JAVASCRIPT_SAFE_INTEGER,
            maximum_job_metadata_bytes: MAXIMUM_DOWNLOAD_JOB_BYTES,
        }
    }
}

impl HuggingFaceHubLimits {
    #[must_use]
    pub const fn new(
        maximum_tree_page_count: usize,
        maximum_file_count: usize,
        maximum_path_bytes: usize,
        maximum_total_bytes: u64,
    ) -> Self {
        Self {
            maximum_tree_page_count,
            maximum_file_count,
            maximum_path_bytes,
            maximum_total_bytes,
            maximum_job_metadata_bytes: MAXIMUM_DOWNLOAD_JOB_BYTES,
        }
    }

    #[must_use]
    pub const fn with_maximum_job_metadata_bytes(
        mut self,
        maximum_job_metadata_bytes: usize,
    ) -> Self {
        self.maximum_job_metadata_bytes = maximum_job_metadata_bytes;
        self
    }
}

impl HuggingFaceHub {
    #[must_use]
    pub fn new(transport: Arc<dyn HubTransport>) -> Self {
        Self::with_limits(transport, HuggingFaceHubLimits::default())
    }

    #[must_use]
    pub fn with_limits(transport: Arc<dyn HubTransport>, limits: HuggingFaceHubLimits) -> Self {
        Self { transport, limits }
    }

    pub async fn fetch_manifest(
        &self,
        repository_id: &str,
        revision: &str,
    ) -> Result<HuggingFaceManifest, HuggingFaceHubError> {
        let complete_repository = DownloadPathSelection::default();
        self.fetch_selected_manifest(repository_id, revision, &complete_repository)
            .await
    }

    pub async fn fetch_selected_manifest(
        &self,
        repository_id: &str,
        revision: &str,
        path_selection: &DownloadPathSelection,
    ) -> Result<HuggingFaceManifest, HuggingFaceHubError> {
        if !is_valid_huggingface_id(repository_id) {
            return Err(HuggingFaceHubError::InvalidRepositoryId);
        }
        if !is_valid_immutable_revision(revision) {
            return Err(HuggingFaceHubError::InvalidRevision);
        }

        self.fetch_and_validate_metadata(repository_id, revision)
            .await?;
        self.fetch_recursive_tree(repository_id, revision, path_selection)
            .await
    }

    async fn fetch_and_validate_metadata(
        &self,
        repository_id: &str,
        revision: &str,
    ) -> Result<(), HuggingFaceHubError> {
        let metadata_url =
            format!("{HUGGING_FACE_ORIGIN}/api/models/{repository_id}/revision/{revision}");
        let metadata_response = self.execute_metadata_get(metadata_url).await?;
        let metadata_document: ModelMetadataDocument =
            serde_json::from_slice(&metadata_response.body_bytes())
                .map_err(HuggingFaceHubError::InvalidMetadataJson)?;

        if metadata_document.sha != revision {
            return Err(HuggingFaceHubError::RevisionMismatch);
        }
        if metadata_document.id.as_deref() != Some(repository_id) {
            return Err(HuggingFaceHubError::RepositoryMismatch);
        }
        if metadata_document.private == Some(true) {
            return Err(HuggingFaceHubError::DownloadGated);
        }
        if metadata_document.private != Some(false) {
            return Err(HuggingFaceHubError::InvalidVisibility);
        }
        match metadata_document.gated {
            Some(GatedStatus::Boolean(false)) => Ok(()),
            Some(GatedStatus::Boolean(true)) => Err(HuggingFaceHubError::DownloadGated),
            Some(GatedStatus::Mode(gating_mode)) if !gating_mode.is_empty() => {
                Err(HuggingFaceHubError::DownloadGated)
            }
            Some(GatedStatus::Mode(_)) | None => Err(HuggingFaceHubError::InvalidVisibility),
        }
    }

    async fn fetch_recursive_tree(
        &self,
        repository_id: &str,
        revision: &str,
        path_selection: &DownloadPathSelection,
    ) -> Result<HuggingFaceManifest, HuggingFaceHubError> {
        let expected_tree_url =
            format!("{HUGGING_FACE_ORIGIN}/api/models/{repository_id}/tree/{revision}");
        let mut next_page_url = Some(format!("{expected_tree_url}?recursive=true"));
        let mut fetched_page_count = 0_usize;
        let mut fetched_tree_entry_count = 0_usize;
        let mut files = Vec::new();
        let mut normalized_paths = BTreeSet::new();
        let mut normalized_file_paths = BTreeSet::new();
        let mut total_bytes = 0_u64;
        let mut estimated_job_metadata_bytes = DOWNLOAD_JOB_FIXED_METADATA_ALLOWANCE_BYTES;

        while let Some(page_url) = next_page_url {
            fetched_page_count = fetched_page_count.saturating_add(1);
            if fetched_page_count > self.limits.maximum_tree_page_count {
                return Err(HuggingFaceHubError::TooManyTreePages);
            }
            validate_tree_page_url(&page_url, &expected_tree_url)?;
            let tree_response = self.execute_metadata_get(page_url).await?;
            let tree_entries: Vec<TreeEntryDocument> =
                serde_json::from_slice(&tree_response.body_bytes())
                    .map_err(HuggingFaceHubError::InvalidTreeJson)?;

            for tree_entry in tree_entries {
                fetched_tree_entry_count = fetched_tree_entry_count.saturating_add(1);
                if fetched_tree_entry_count > self.limits.maximum_file_count {
                    return Err(HuggingFaceHubError::TooManyTreeEntries);
                }
                validate_tree_entry_common(&tree_entry, self.limits.maximum_path_bytes)?;
                let normalized_path = tree_entry.path.to_ascii_lowercase();
                if has_ancestor_path(&normalized_file_paths, &normalized_path) {
                    return Err(HuggingFaceHubError::FilePathHierarchyConflict);
                }
                if !normalized_paths.insert(normalized_path.clone()) {
                    return Err(HuggingFaceHubError::CaseInsensitivePathCollision);
                }
                match tree_entry.entry_type.as_str() {
                    "directory" => validate_directory_entry(&tree_entry)?,
                    "file" => {
                        if has_descendant_path(&normalized_paths, &normalized_path) {
                            return Err(HuggingFaceHubError::FilePathHierarchyConflict);
                        }
                        if path_selection.includes(&tree_entry.path) {
                            if files.len() >= self.limits.maximum_file_count {
                                return Err(HuggingFaceHubError::TooManyFiles);
                            }
                            total_bytes = total_bytes
                                .checked_add(tree_entry.size)
                                .ok_or(HuggingFaceHubError::TotalBytesTooLarge)?;
                            if total_bytes > self.limits.maximum_total_bytes
                                || total_bytes > MAXIMUM_JAVASCRIPT_SAFE_INTEGER
                            {
                                return Err(HuggingFaceHubError::TotalBytesTooLarge);
                            }
                            let manifest_file = validated_file(tree_entry)?;
                            estimated_job_metadata_bytes = estimated_job_metadata_bytes
                                .checked_add(estimated_durable_file_metadata_bytes(&manifest_file))
                                .ok_or(HuggingFaceHubError::JobMetadataTooLarge)?;
                            if estimated_job_metadata_bytes
                                > self
                                    .limits
                                    .maximum_job_metadata_bytes
                                    .min(MAXIMUM_DOWNLOAD_JOB_BYTES)
                            {
                                return Err(HuggingFaceHubError::JobMetadataTooLarge);
                            }
                            files.push(manifest_file);
                        }
                        normalized_file_paths.insert(normalized_path);
                    }
                    _ => return Err(HuggingFaceHubError::InvalidEntryType),
                }
            }
            next_page_url = parse_next_link(tree_response.selected_header("link"))?;
        }

        if files.is_empty() || total_bytes == 0 {
            return Err(HuggingFaceHubError::EmptyManifest);
        }

        Ok(HuggingFaceManifest {
            repository_id: repository_id.to_owned(),
            revision: revision.to_owned(),
            files,
            total_bytes,
        })
    }

    async fn execute_metadata_get(
        &self,
        url: String,
    ) -> Result<HubHttpResponse, HuggingFaceHubError> {
        let response = self
            .transport
            .execute(HubHttpRequest::metadata_get(url))
            .await?;
        match response.status() {
            200..=299 => Ok(response),
            401 | 403 => Err(HuggingFaceHubError::DownloadGated),
            status => Err(HuggingFaceHubError::UnexpectedStatus { status }),
        }
    }
}

impl HuggingFaceManifest {
    #[must_use]
    pub fn repository_id(&self) -> &str {
        &self.repository_id
    }

    #[must_use]
    pub fn revision(&self) -> &str {
        &self.revision
    }

    #[must_use]
    pub fn files(&self) -> &[HubManifestFile] {
        &self.files
    }

    #[must_use]
    pub const fn total_bytes(&self) -> u64 {
        self.total_bytes
    }
}

impl HubManifestFile {
    #[must_use]
    pub fn relative_path(&self) -> &str {
        &self.relative_path
    }

    #[must_use]
    pub const fn expected_bytes(&self) -> u64 {
        self.expected_bytes
    }

    #[must_use]
    pub fn digest(&self) -> &DownloadFileDigest {
        &self.digest
    }

    #[must_use]
    pub fn xet_hash(&self) -> Option<&str> {
        self.xet_hash.as_deref()
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
