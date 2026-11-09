//! Bounded immutable-revision manifest discovery from the public Hugging Face Hub.

use std::sync::Arc;

use reqwest::Url;
use serde::Deserialize;
use thiserror::Error;

use super::{
    hugging_face_bounds::{parse_next_link, validate_tree_page_url},
    transport::{HubHttpRequest, HubHttpResponse, HubTransport, HubTransportError},
};
use crate::library::download::job::download_job::MAXIMUM_DOWNLOAD_JOB_BYTES;
use tree_manifest::{TreeEntryDocument, TreeManifestAccumulator};

use crate::library::{
    DownloadFileDigest, DownloadPathSelection, is_valid_huggingface_id, is_valid_immutable_revision,
};

const HUGGING_FACE_ORIGIN: &str = "https://huggingface.co";
const MAXIMUM_JAVASCRIPT_SAFE_INTEGER: u64 = 9_007_199_254_740_991;
const DEFAULT_MAXIMUM_TREE_PAGE_COUNT: usize = 128;
const DEFAULT_MAXIMUM_FILE_COUNT: usize = 65_536;
const DEFAULT_MAXIMUM_PATH_BYTES: usize = 1_024;
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
#[derive(Clone)]
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

    /// Walks every page of the recursive tree. Each page's entries are
    /// checked as the page arrives, so a hostile tree fails at its first bad
    /// page instead of after the whole tree has been fetched.
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
        let mut tree_manifest = TreeManifestAccumulator::new(self.limits);
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
                tree_manifest.accept_entry(tree_entry, path_selection)?;
            }
            next_page_url = parse_next_link(tree_response.selected_header("link"))?;
        }
        tree_manifest.finish(repository_id, revision)
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

    /// Fetches one small immutable repository text file through the metadata transport.
    ///
    /// The `/raw/` origin endpoint serves regular Git files directly, so preflight metadata
    /// checks stay on the trusted Hugging Face origin instead of following payload redirects to
    /// content-delivery hosts. The metadata body byte bound applies unchanged.
    pub async fn fetch_bounded_repository_file(
        &self,
        repository_id: &str,
        revision: &str,
        relative_path: &str,
    ) -> Result<Vec<u8>, HuggingFaceHubError> {
        if !is_valid_huggingface_id(repository_id) {
            return Err(HuggingFaceHubError::InvalidRepositoryId);
        }
        if !is_valid_immutable_revision(revision) {
            return Err(HuggingFaceHubError::InvalidRevision);
        }
        let mut file_url = Url::parse(HUGGING_FACE_ORIGIN)
            .map_err(|_| HuggingFaceHubError::InvalidRepositoryId)?;
        {
            let mut path_segments = file_url
                .path_segments_mut()
                .map_err(|_| HuggingFaceHubError::InvalidRepositoryId)?;
            for repository_component in repository_id.split('/') {
                path_segments.push(repository_component);
            }
            path_segments.push("raw").push(revision);
            for file_component in relative_path.split('/') {
                path_segments.push(file_component);
            }
        }
        let file_response = self.execute_metadata_get(file_url.to_string()).await?;
        Ok(file_response.body_bytes())
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

mod tree_manifest;
