use std::collections::BTreeMap;
use std::fs::File;

use crate::artifact_validation::{
    ArtifactValidationError, ValidatedRequiredFile, ValidatedWeightsFile,
};
use crate::laguna::{LagunaTargetContract, LagunaTextArtifactDescriptor};

use super::canonical_tensor_contract::LagunaTensorContract;
use super::shard_index::LagunaShardIndex;

/// Producer convention matched by model.safetensors.index.json metadata.total_size.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LagunaIndexTotalSizeSemantics {
    SerializedShardFiles,
    TensorPayload,
}

/// Descriptor-backed result of complete Laguna startup artifact validation.
#[derive(Debug)]
pub struct ValidatedLagunaArtifact {
    pub(super) target_contract: LagunaTargetContract,
    pub(super) shard_index: LagunaShardIndex,
    pub(super) tensor_contract: LagunaTensorContract,
    pub(super) text_artifact: LagunaTextArtifactDescriptor,
    pub(super) config_file: ValidatedRequiredFile,
    pub(super) index_file: ValidatedRequiredFile,
    pub(super) tokenizer_file: ValidatedRequiredFile,
    pub(super) tokenizer_config_file: ValidatedRequiredFile,
    pub(super) generation_config_file: ValidatedRequiredFile,
    pub(super) included_template_files: BTreeMap<String, ValidatedRequiredFile>,
    pub(super) shard_files: BTreeMap<String, ValidatedWeightsFile>,
    pub(super) total_shard_file_bytes: u64,
    pub(super) total_tensor_payload_bytes: u64,
    pub(super) index_total_size_semantics: LagunaIndexTotalSizeSemantics,
    pub(super) storage_fingerprint: [u8; 32],
}

impl ValidatedLagunaArtifact {
    #[must_use]
    pub const fn target_contract(&self) -> &LagunaTargetContract {
        &self.target_contract
    }

    #[must_use]
    pub const fn shard_index(&self) -> &LagunaShardIndex {
        &self.shard_index
    }

    #[must_use]
    pub const fn tensor_contract(&self) -> &LagunaTensorContract {
        &self.tensor_contract
    }

    /// Returns generation-ready text semantics certified with the weight artifact.
    #[must_use]
    pub const fn text_artifact(&self) -> &LagunaTextArtifactDescriptor {
        &self.text_artifact
    }

    /// Returns checked aggregate bytes of complete retained SafeTensors shard files.
    #[must_use]
    pub const fn total_shard_file_bytes(&self) -> u64 {
        self.total_shard_file_bytes
    }

    /// Returns checked aggregate bytes covered by exact tensor source intervals.
    #[must_use]
    pub const fn total_tensor_payload_bytes(&self) -> u64 {
        self.total_tensor_payload_bytes
    }

    #[must_use]
    pub const fn index_total_size_semantics(&self) -> LagunaIndexTotalSizeSemantics {
        self.index_total_size_semantics
    }

    /// Returns the SHA-256 identity of canonical target and physical storage characteristics.
    #[must_use]
    pub const fn storage_fingerprint(&self) -> &[u8; 32] {
        &self.storage_fingerprint
    }

    /// Transfers every descriptor without reopening any model-directory path.
    pub fn into_retained_files(
        self,
    ) -> Result<LagunaRetainedArtifactFiles, ArtifactValidationError> {
        let included_template_files = self
            .included_template_files
            .into_iter()
            .map(|(template_file_name, template_file)| {
                retained_file(template_file)
                    .map(|retained_template_file| (template_file_name, retained_template_file))
            })
            .collect::<Result<BTreeMap<_, _>, _>>()?;
        Ok(LagunaRetainedArtifactFiles {
            text_artifact: self.text_artifact,
            config_file: retained_file(self.config_file)?,
            index_file: retained_file(self.index_file)?,
            tokenizer_file: retained_file(self.tokenizer_file)?,
            tokenizer_config_file: retained_file(self.tokenizer_config_file)?,
            generation_config_file: retained_file(self.generation_config_file)?,
            included_template_files,
            shard_files: self.shard_files,
        })
    }
}

/// Explicit ownership transfer of all files retained by Laguna validation.
#[derive(Debug)]
pub struct LagunaRetainedArtifactFiles {
    text_artifact: LagunaTextArtifactDescriptor,
    config_file: File,
    index_file: File,
    tokenizer_file: File,
    tokenizer_config_file: File,
    generation_config_file: File,
    included_template_files: BTreeMap<String, File>,
    shard_files: BTreeMap<String, ValidatedWeightsFile>,
}

impl LagunaRetainedArtifactFiles {
    #[must_use]
    pub const fn text_artifact(&self) -> &LagunaTextArtifactDescriptor {
        &self.text_artifact
    }

    #[must_use]
    pub const fn config_file(&self) -> &File {
        &self.config_file
    }

    #[must_use]
    pub const fn index_file(&self) -> &File {
        &self.index_file
    }

    #[must_use]
    pub const fn tokenizer_file(&self) -> &File {
        &self.tokenizer_file
    }

    #[must_use]
    pub const fn tokenizer_config_file(&self) -> &File {
        &self.tokenizer_config_file
    }

    #[must_use]
    pub const fn generation_config_file(&self) -> &File {
        &self.generation_config_file
    }

    #[must_use]
    pub const fn included_template_files(&self) -> &BTreeMap<String, File> {
        &self.included_template_files
    }

    #[must_use]
    pub const fn shard_files(&self) -> &BTreeMap<String, ValidatedWeightsFile> {
        &self.shard_files
    }

    /// Consumes the transfer object and returns all shard descriptors by plain filename.
    #[must_use]
    pub fn into_shard_files(self) -> BTreeMap<String, ValidatedWeightsFile> {
        self.shard_files
    }
}

fn retained_file(validated_file: ValidatedRequiredFile) -> Result<File, ArtifactValidationError> {
    // Recheck descriptor identity at ownership transfer without reopening its path.
    Ok(validated_file.into_validated_weights_file()?.into_file())
}
