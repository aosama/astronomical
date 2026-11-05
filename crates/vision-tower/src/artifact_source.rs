//! One way to read an artifact, wherever it lives. An enum rather than a trait,
//! because there are exactly two places a model can be and the async methods
//! would otherwise need boxing to be chosen at run time.

use crate::artifact_reference::ArtifactReference;
use crate::error::VisionTowerError;
use crate::hugging_face_source::HuggingFaceSource;
use crate::local_directory_source::LocalDirectorySource;

pub const CONFIG_FILE_NAME: &str = "config.json";
pub const WEIGHT_INDEX_FILE_NAME: &str = "model.safetensors.index.json";

/// One file in an artifact, named relative to its root with `/` separators.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ArtifactFile {
    pub relative_path: String,
    pub byte_count: Option<u64>,
}

/// What an artifact holds and where it came from.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ArtifactListing {
    pub display_name: String,
    /// The public repository this came from, when known. Build records use this and
    /// never `display_name`, which for a local artifact is a path on this machine.
    pub origin_repository: Option<String>,
    /// The exact commit every later read is pinned to, for a repository.
    pub pinned_revision: Option<String>,
    pub license: Option<String>,
    pub pipeline_tag: Option<String>,
    pub files: Vec<ArtifactFile>,
}

#[derive(Debug)]
pub enum ArtifactSource {
    LocalDirectory(LocalDirectorySource),
    HuggingFace(HuggingFaceSource),
}

impl ArtifactSource {
    pub async fn open(reference: &ArtifactReference) -> Result<Self, VisionTowerError> {
        match reference {
            ArtifactReference::LocalDirectory(directory) => {
                Ok(Self::LocalDirectory(LocalDirectorySource::open(directory)?))
            }
            ArtifactReference::HuggingFaceRepository {
                repository_id,
                revision,
            } => Ok(Self::HuggingFace(
                HuggingFaceSource::open(repository_id, revision.as_deref()).await?,
            )),
        }
    }

    #[must_use]
    pub const fn listing(&self) -> &ArtifactListing {
        match self {
            Self::LocalDirectory(local_source) => local_source.listing(),
            Self::HuggingFace(hugging_face_source) => hugging_face_source.listing(),
        }
    }

    /// The file's whole contents, or `None` when the artifact does not have it.
    pub async fn read_whole_file(
        &self,
        relative_path: &str,
        maximum_bytes: u64,
    ) -> Result<Option<Vec<u8>>, VisionTowerError> {
        if !self.has_file(relative_path) {
            return Ok(None);
        }
        match self {
            Self::LocalDirectory(local_source) => local_source
                .read_whole_file(relative_path, maximum_bytes)
                .map(Some),
            Self::HuggingFace(hugging_face_source) => hugging_face_source
                .read_whole_file(relative_path, maximum_bytes)
                .await
                .map(Some),
        }
    }

    /// Exactly `byte_count` bytes from `start_byte`, or an error.
    pub async fn read_byte_range(
        &self,
        relative_path: &str,
        start_byte: u64,
        byte_count: u64,
    ) -> Result<Vec<u8>, VisionTowerError> {
        match self {
            Self::LocalDirectory(local_source) => {
                local_source.read_byte_range(relative_path, start_byte, byte_count)
            }
            Self::HuggingFace(hugging_face_source) => {
                hugging_face_source
                    .read_byte_range(relative_path, start_byte, byte_count)
                    .await
            }
        }
    }

    fn has_file(&self, relative_path: &str) -> bool {
        for artifact_file in &self.listing().files {
            if artifact_file.relative_path == relative_path {
                return true;
            }
        }
        false
    }
}
