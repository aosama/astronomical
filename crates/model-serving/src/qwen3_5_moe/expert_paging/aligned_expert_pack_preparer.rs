//! Explicit, atomic preparation of one complete aligned expert-pack revision.

use std::{
    collections::HashMap,
    fs::{self, File, OpenOptions},
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use thiserror::Error;

use super::{
    aligned_expert_pack::{
        AlignedExpertPackBuildRequest, AlignedExpertPackError, build_aligned_expert_pack,
        read_aligned_expert_pack_header, validate_aligned_expert_pack_header,
    },
    quantized_expert_layer_plan::build_quantized_expert_layer_plan,
    quantized_expert_manifest::QuantizedExpertLayerPlan,
    quantized_expert_manifest::{ExpertManifestError, QuantizationMode},
};
use crate::qwen3_5::{
    ModelWeightStorage, Qwen3_5ArtifactValidationError, Qwen3_5ArtifactValidator,
};

const ALIGNED_EXPERT_PACK_ROOT_DIRECTORY_NAME: &str = ".astronomical-aligned-expert-packs";

/// One completed layer reported by the explicit preparation command.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AlignedExpertPackPreparationProgress {
    pub completed_layer_count: usize,
    pub total_layer_count: usize,
    pub layer_index: usize,
    pub layer_byte_count: u64,
    pub total_completed_byte_count: u64,
    pub elapsed: Duration,
}

/// Final outcome of preparing one complete model revision.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AlignedExpertPackPreparationReport {
    pub model_id: String,
    pub model_revision: String,
    pub completed_layer_count: usize,
    pub total_pack_byte_count: u64,
    pub final_revision_directory: PathBuf,
    pub reused_existing_pack_set: bool,
    pub elapsed: Duration,
}

/// Read-only preflight information shown before user consent.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AlignedExpertPackPreparationInspection {
    pub model_id: String,
    pub model_revision: String,
    pub total_layer_count: usize,
    pub total_pack_byte_count: u64,
    pub remaining_pack_byte_count: u64,
    pub available_byte_count: u64,
    pub final_revision_directory: PathBuf,
    pub has_valid_final_revision: bool,
}

/// Failures while planning or atomically preparing one aligned pack set.
#[derive(Debug, Error)]
pub enum AlignedExpertPackPreparationError {
    #[error("aligned expert-pack preparation requires at least one expert layer")]
    EmptyLayerPlans,
    #[error("model directory is not a directory: {model_directory:?}")]
    ModelDirectoryNotFound { model_directory: PathBuf },
    #[error("another aligned expert-pack preparation already owns {lock_path:?}")]
    PreparationAlreadyRunning { lock_path: PathBuf },
    #[error(
        "aligned expert-pack revision already exists but is invalid; rerun with replace: {revision_directory:?}"
    )]
    InvalidExistingRevision { revision_directory: PathBuf },
    #[error(
        "insufficient destination space: required {required_byte_count} bytes, available {available_byte_count} bytes"
    )]
    InsufficientAvailableSpace {
        required_byte_count: u64,
        available_byte_count: u64,
    },
    #[error("aligned expert-pack byte accounting overflowed")]
    ByteCountOverflow,
    #[error("aligned expert-pack filesystem operation failed: {0}")]
    Io(#[from] std::io::Error),
    #[error("aligned expert-pack construction failed: {0}")]
    Pack(#[from] AlignedExpertPackError),
    #[error("model artifact validation failed: {0}")]
    Artifact(#[from] Qwen3_5ArtifactValidationError),
    #[error("expert layer planning failed: {0}")]
    ExpertManifest(#[from] ExpertManifestError),
}

/// Owns preparation inputs for one already-validated model revision.
#[derive(Debug)]
pub struct AlignedExpertPackPreparer {
    model_directory: PathBuf,
    model_id: String,
    model_revision: String,
    layer_plans: Vec<QuantizedExpertLayerPlan>,
}

impl AlignedExpertPackPreparer {
    pub fn for_model_directory(
        model_directory: impl AsRef<Path>,
    ) -> Result<Self, AlignedExpertPackPreparationError> {
        let model_directory = fs::canonicalize(model_directory)?;
        let validated_artifact = Qwen3_5ArtifactValidator::new().validate(&model_directory, 1)?;
        let model_id = validated_artifact.model_id().to_owned();
        let model_revision = validated_artifact.revision().to_owned();
        let config = validated_artifact.config();
        let quantization_mode = match config.model_weight_storage() {
            ModelWeightStorage::AffineQuantized => QuantizationMode::Affine,
            ModelWeightStorage::NativeBfloat16 => QuantizationMode::NativeBfloat16,
        };
        let tensor_name_to_shard_file_name = validated_artifact
            .shard_index()
            .language_tensor_name_to_shard_file_name()
            .iter()
            .map(|(tensor_name, shard_file_name)| (tensor_name.clone(), shard_file_name.clone()))
            .collect::<HashMap<_, _>>();
        let layer_plans = (0..config.layer_count() as usize)
            .map(|layer_index| {
                build_quantized_expert_layer_plan(
                    &model_directory,
                    &tensor_name_to_shard_file_name,
                    &format!("language_model.model.layers.{layer_index}.mlp"),
                    config,
                    quantization_mode,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        Self::from_layer_plans(model_directory, model_id, model_revision, layer_plans)
    }

    pub fn from_layer_plans(
        model_directory: impl AsRef<Path>,
        model_id: impl Into<String>,
        model_revision: impl Into<String>,
        layer_plans: Vec<QuantizedExpertLayerPlan>,
    ) -> Result<Self, AlignedExpertPackPreparationError> {
        let model_directory = model_directory.as_ref();
        if !model_directory.is_dir() {
            return Err(AlignedExpertPackPreparationError::ModelDirectoryNotFound {
                model_directory: model_directory.to_path_buf(),
            });
        }
        if layer_plans.is_empty() {
            return Err(AlignedExpertPackPreparationError::EmptyLayerPlans);
        }
        Ok(Self {
            model_directory: model_directory.to_path_buf(),
            model_id: model_id.into(),
            model_revision: model_revision.into(),
            layer_plans,
        })
    }

    pub fn prepare(
        &self,
        should_replace_existing_revision: bool,
        mut report_progress: impl FnMut(AlignedExpertPackPreparationProgress),
    ) -> Result<AlignedExpertPackPreparationReport, AlignedExpertPackPreparationError> {
        let preparation_started_at = Instant::now();
        let pack_root_directory = self
            .model_directory
            .join(ALIGNED_EXPERT_PACK_ROOT_DIRECTORY_NAME);
        fs::create_dir_all(&pack_root_directory)?;
        let lock_path = pack_root_directory.join(format!(".{}.lock", self.model_revision));
        let lock_file = acquire_preparation_lock(&lock_path)?;
        let final_revision_directory = pack_root_directory.join(&self.model_revision);
        let staging_revision_directory =
            pack_root_directory.join(format!(".{}.preparing", self.model_revision));

        if final_revision_directory.exists() {
            if let Ok(total_pack_byte_count) =
                self.validate_complete_revision(&final_revision_directory)
            {
                drop(lock_file);
                return Ok(AlignedExpertPackPreparationReport {
                    model_id: self.model_id.clone(),
                    model_revision: self.model_revision.clone(),
                    completed_layer_count: self.layer_plans.len(),
                    total_pack_byte_count,
                    final_revision_directory,
                    reused_existing_pack_set: true,
                    elapsed: preparation_started_at.elapsed(),
                });
            }
            if !should_replace_existing_revision {
                return Err(AlignedExpertPackPreparationError::InvalidExistingRevision {
                    revision_directory: final_revision_directory,
                });
            }
            fs::remove_dir_all(&final_revision_directory)?;
        }

        fs::create_dir_all(&staging_revision_directory)?;
        let mut total_pack_byte_count = 0_u64;
        let mut remaining_pack_byte_count = 0_u64;
        let mut staged_layer_headers = Vec::with_capacity(self.layer_plans.len());
        for (layer_index, layer_plan) in self.layer_plans.iter().enumerate() {
            let staged_pack_path = layer_pack_path(&staging_revision_directory, layer_index);
            let validated_existing_header = self
                .read_validated_layer_header(&staged_pack_path, layer_plan, layer_index)
                .ok();
            if let Some(existing_header) = validated_existing_header {
                staged_layer_headers.push(Some(existing_header));
                continue;
            }
            if staged_pack_path.exists() {
                fs::remove_file(&staged_pack_path)?;
            }
            let planned_header = super::aligned_expert_pack::plan_aligned_expert_pack_header(
                &AlignedExpertPackBuildRequest {
                    model_id: &self.model_id,
                    model_revision: &self.model_revision,
                    layer_index,
                    layer_plan,
                },
            )?;
            remaining_pack_byte_count = remaining_pack_byte_count
                .checked_add(planned_header.expected_pack_byte_count)
                .ok_or(AlignedExpertPackPreparationError::ByteCountOverflow)?;
            staged_layer_headers.push(None);
        }
        let available_byte_count = fs4::available_space(&pack_root_directory)?;
        if remaining_pack_byte_count > available_byte_count {
            return Err(
                AlignedExpertPackPreparationError::InsufficientAvailableSpace {
                    required_byte_count: remaining_pack_byte_count,
                    available_byte_count,
                },
            );
        }

        for (layer_index, layer_plan) in self.layer_plans.iter().enumerate() {
            let aligned_expert_pack_header = match staged_layer_headers[layer_index].take() {
                Some(existing_header) => existing_header,
                None => build_aligned_expert_pack(
                    &layer_pack_path(&staging_revision_directory, layer_index),
                    &AlignedExpertPackBuildRequest {
                        model_id: &self.model_id,
                        model_revision: &self.model_revision,
                        layer_index,
                        layer_plan,
                    },
                )?,
            };
            total_pack_byte_count = total_pack_byte_count
                .checked_add(aligned_expert_pack_header.expected_pack_byte_count)
                .ok_or(AlignedExpertPackPreparationError::ByteCountOverflow)?;
            report_progress(AlignedExpertPackPreparationProgress {
                completed_layer_count: layer_index + 1,
                total_layer_count: self.layer_plans.len(),
                layer_index,
                layer_byte_count: aligned_expert_pack_header.expected_pack_byte_count,
                total_completed_byte_count: total_pack_byte_count,
                elapsed: preparation_started_at.elapsed(),
            });
        }
        total_pack_byte_count = self.validate_complete_revision(&staging_revision_directory)?;
        fs::rename(&staging_revision_directory, &final_revision_directory)?;
        drop(lock_file);
        Ok(AlignedExpertPackPreparationReport {
            model_id: self.model_id.clone(),
            model_revision: self.model_revision.clone(),
            completed_layer_count: self.layer_plans.len(),
            total_pack_byte_count,
            final_revision_directory,
            reused_existing_pack_set: false,
            elapsed: preparation_started_at.elapsed(),
        })
    }

    pub fn inspect(
        &self,
    ) -> Result<AlignedExpertPackPreparationInspection, AlignedExpertPackPreparationError> {
        let pack_root_directory = self
            .model_directory
            .join(ALIGNED_EXPERT_PACK_ROOT_DIRECTORY_NAME);
        let final_revision_directory = pack_root_directory.join(&self.model_revision);
        let staging_revision_directory =
            pack_root_directory.join(format!(".{}.preparing", self.model_revision));
        let mut total_pack_byte_count = 0_u64;
        let mut remaining_pack_byte_count = 0_u64;
        for (layer_index, layer_plan) in self.layer_plans.iter().enumerate() {
            let planned_header = super::aligned_expert_pack::plan_aligned_expert_pack_header(
                &AlignedExpertPackBuildRequest {
                    model_id: &self.model_id,
                    model_revision: &self.model_revision,
                    layer_index,
                    layer_plan,
                },
            )?;
            total_pack_byte_count = total_pack_byte_count
                .checked_add(planned_header.expected_pack_byte_count)
                .ok_or(AlignedExpertPackPreparationError::ByteCountOverflow)?;
            let staged_pack_path = layer_pack_path(&staging_revision_directory, layer_index);
            let has_valid_staged_pack = self
                .read_validated_layer_header(&staged_pack_path, layer_plan, layer_index)
                .is_ok();
            if !has_valid_staged_pack {
                remaining_pack_byte_count = remaining_pack_byte_count
                    .checked_add(planned_header.expected_pack_byte_count)
                    .ok_or(AlignedExpertPackPreparationError::ByteCountOverflow)?;
            }
        }
        let has_valid_final_revision = self
            .validate_complete_revision(&final_revision_directory)
            .is_ok();
        if has_valid_final_revision {
            remaining_pack_byte_count = 0;
        }
        let available_byte_count = fs4::available_space(&self.model_directory)?;
        Ok(AlignedExpertPackPreparationInspection {
            model_id: self.model_id.clone(),
            model_revision: self.model_revision.clone(),
            total_layer_count: self.layer_plans.len(),
            total_pack_byte_count,
            remaining_pack_byte_count,
            available_byte_count,
            final_revision_directory,
            has_valid_final_revision,
        })
    }

    fn validate_complete_revision(
        &self,
        revision_directory: &Path,
    ) -> Result<u64, AlignedExpertPackPreparationError> {
        let mut total_pack_byte_count = 0_u64;
        for (layer_index, layer_plan) in self.layer_plans.iter().enumerate() {
            let pack_path = layer_pack_path(revision_directory, layer_index);
            let pack_header = read_aligned_expert_pack_header(&pack_path)?;
            validate_aligned_expert_pack_header(
                &pack_path,
                &pack_header,
                layer_plan,
                &self.model_id,
                &self.model_revision,
                layer_index,
            )?;
            super::aligned_expert_pack::validate_aligned_expert_pack_payload(
                &pack_path,
                &pack_header,
                layer_plan,
            )?;
            total_pack_byte_count = total_pack_byte_count
                .checked_add(pack_header.expected_pack_byte_count)
                .ok_or(AlignedExpertPackPreparationError::ByteCountOverflow)?;
        }
        Ok(total_pack_byte_count)
    }

    fn read_validated_layer_header(
        &self,
        pack_path: &Path,
        layer_plan: &QuantizedExpertLayerPlan,
        layer_index: usize,
    ) -> Result<
        super::aligned_expert_pack::AlignedExpertPackHeader,
        AlignedExpertPackPreparationError,
    > {
        let pack_header = read_aligned_expert_pack_header(pack_path)?;
        validate_aligned_expert_pack_header(
            pack_path,
            &pack_header,
            layer_plan,
            &self.model_id,
            &self.model_revision,
            layer_index,
        )?;
        super::aligned_expert_pack::validate_aligned_expert_pack_payload(
            pack_path,
            &pack_header,
            layer_plan,
        )?;
        Ok(pack_header)
    }
}

fn acquire_preparation_lock(lock_path: &Path) -> Result<File, AlignedExpertPackPreparationError> {
    let lock_file = OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(lock_path)?;
    lock_file.try_lock().map_err(|_| {
        AlignedExpertPackPreparationError::PreparationAlreadyRunning {
            lock_path: lock_path.to_path_buf(),
        }
    })?;
    Ok(lock_file)
}

fn layer_pack_path(revision_directory: &Path, layer_index: usize) -> PathBuf {
    revision_directory.join(format!("layer-{layer_index}.aligned-expert-pack"))
}
