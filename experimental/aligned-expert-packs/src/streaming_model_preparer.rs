//! Converts a Qwen3.5-MoE artifact into a self-sufficient per-expert streaming model.

use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use astronomical_model_serving::{
    QuantizedExpertLayerPlan, Qwen3_5ArtifactValidator, build_quantized_expert_layer_plan,
};

use crate::aligned_expert_pack_preparer::AlignedExpertPackPreparationError;
use crate::per_expert_pack::{
    PerExpertPackBuildRequest, build_per_expert_pack, per_expert_pack_relative_path,
};
use crate::revision_manifest::StreamingModelManifest;

/// Suffix appended to the source model identity for the converted streaming model.
pub const STREAMING_MODEL_IDENTITY_SUFFIX: &str = "-expert-streaming";

/// One completed expert file reported during streaming-model preparation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StreamingModelPreparationProgress {
    pub completed_expert_file_count: usize,
    pub total_expert_file_count: usize,
    pub layer_index: usize,
    pub expert_id: usize,
    pub expert_file_byte_count: u64,
    pub elapsed: Duration,
}

/// Final outcome of preparing one independently loadable streaming model.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StreamingModelPreparationReport {
    pub source_model_id: String,
    pub streaming_model_id: String,
    pub model_revision: String,
    pub completed_expert_file_count: usize,
    pub total_pack_byte_count: u64,
    pub final_model_directory: PathBuf,
    pub reused_existing_revision: bool,
    pub elapsed: Duration,
}

/// Owns conversion inputs for one already-validated Qwen3.5-MoE revision.
#[derive(Debug)]
pub struct StreamingModelPreparer {
    model_directory: PathBuf,
    source_model_id: String,
    streaming_model_id: String,
    model_revision: String,
    layer_plans: Vec<QuantizedExpertLayerPlan>,
}

impl StreamingModelPreparer {
    /// Plans conversion from one complete downloaded Qwen3.5-MoE directory.
    pub fn for_model_directory(
        model_directory: impl AsRef<Path>,
    ) -> Result<Self, AlignedExpertPackPreparationError> {
        let model_directory = fs::canonicalize(model_directory)?;
        let validated_artifact = Qwen3_5ArtifactValidator::new().validate(&model_directory, 1)?;
        let source_model_id = validated_artifact.model_id().to_owned();
        let streaming_model_id = streaming_model_id_for(&source_model_id);
        let model_revision = validated_artifact.revision().to_owned();
        let config = validated_artifact.config();
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
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        Self::from_layer_plans(
            model_directory,
            source_model_id,
            streaming_model_id,
            model_revision,
            layer_plans,
        )
    }

    /// Plans conversion from already-built layer plans (hermetic fixtures).
    pub fn from_layer_plans(
        model_directory: impl AsRef<Path>,
        source_model_id: impl Into<String>,
        streaming_model_id: impl Into<String>,
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
            source_model_id: source_model_id.into(),
            streaming_model_id: streaming_model_id.into(),
            model_revision: model_revision.into(),
            layer_plans,
        })
    }

    /// Returns the converted model identity.
    #[must_use]
    pub fn streaming_model_id(&self) -> &str {
        &self.streaming_model_id
    }

    /// Publishes one self-sufficient streaming-model directory.
    pub fn prepare(
        &self,
        output_directory: impl AsRef<Path>,
        should_replace_existing_revision: bool,
        mut report_progress: impl FnMut(StreamingModelPreparationProgress),
    ) -> Result<StreamingModelPreparationReport, AlignedExpertPackPreparationError> {
        let preparation_started_at = Instant::now();
        let final_model_directory = output_directory.as_ref().to_path_buf();
        if final_model_directory.exists() {
            if super::streaming_model_revision::validate_complete_streaming_model(
                &self.source_model_id,
                &self.streaming_model_id,
                &self.model_revision,
                &self.layer_plans,
                &final_model_directory,
            )
            .is_ok()
            {
                return Ok(StreamingModelPreparationReport {
                    source_model_id: self.source_model_id.clone(),
                    streaming_model_id: self.streaming_model_id.clone(),
                    model_revision: self.model_revision.clone(),
                    completed_expert_file_count: self.total_expert_file_count(),
                    total_pack_byte_count: total_declared_pack_bytes(&final_model_directory)?,
                    final_model_directory,
                    reused_existing_revision: true,
                    elapsed: preparation_started_at.elapsed(),
                });
            }
            if !should_replace_existing_revision {
                return Err(AlignedExpertPackPreparationError::InvalidExistingRevision {
                    revision_directory: final_model_directory,
                });
            }
            fs::remove_dir_all(&final_model_directory)?;
        }
        let parent_directory = final_model_directory.parent().ok_or_else(|| {
            AlignedExpertPackPreparationError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "streaming model output must have a parent directory",
            ))
        })?;
        fs::create_dir_all(parent_directory)?;
        let staging_model_directory =
            existing_staging_directory(parent_directory, &self.streaming_model_id)?.unwrap_or_else(
                || {
                    parent_directory.join(format!(
                        ".{}.preparing-{}",
                        self.streaming_model_id,
                        std::process::id()
                    ))
                },
            );
        fs::create_dir_all(&staging_model_directory)?;
        let copy_outcome = (|| {
            self.publish_revision_files(&staging_model_directory)?;
            let total_expert_file_count = self.total_expert_file_count();
            if self.staging_has_all_expert_files(&staging_model_directory) {
                eprintln!(
                    "status=resume completed_experts={total_expert_file_count}/{total_expert_file_count}"
                );
            } else {
                let mut completed_expert_file_count = 0_usize;
                for (layer_index, layer_plan) in self.layer_plans.iter().enumerate() {
                    let layer_directory = staging_model_directory
                        .join("layers")
                        .join(layer_index.to_string());
                    fs::create_dir_all(&layer_directory)?;
                    for expert_id in 0..layer_plan.expert_capacity {
                        let relative_pack_path =
                            per_expert_pack_relative_path(layer_index, expert_id);
                        let staged_pack_path = staging_model_directory.join(&relative_pack_path);
                        let validated_existing_header =
                            super::streaming_model_revision::read_validated_expert_header(
                                &staged_pack_path,
                                layer_plan,
                                &self.streaming_model_id,
                                &self.model_revision,
                                layer_index,
                                expert_id,
                            )
                            .ok();
                        let expert_file_byte_count =
                            if let Some(existing_header) = validated_existing_header {
                                existing_header.expected_file_byte_count
                            } else {
                                if staged_pack_path.exists() {
                                    fs::remove_file(&staged_pack_path)?;
                                }
                                let pack_header = build_per_expert_pack(
                                    &staged_pack_path,
                                    &PerExpertPackBuildRequest {
                                        model_id: &self.streaming_model_id,
                                        model_revision: &self.model_revision,
                                        layer_index,
                                        expert_id,
                                        layer_plan,
                                    },
                                )?;
                                pack_header.expected_file_byte_count
                            };
                        completed_expert_file_count += 1;
                        report_progress(StreamingModelPreparationProgress {
                            completed_expert_file_count,
                            total_expert_file_count,
                            layer_index,
                            expert_id,
                            expert_file_byte_count,
                            elapsed: preparation_started_at.elapsed(),
                        });
                    }
                }
            }
            super::streaming_model_revision::write_manifest(
                &self.source_model_id,
                &self.streaming_model_id,
                &self.model_revision,
                &self.layer_plans,
                &staging_model_directory,
                |hashed_file_count, total_file_count| {
                    eprintln!("status=hashing hashed_files={hashed_file_count}/{total_file_count}");
                },
            )?;
            Ok::<(), AlignedExpertPackPreparationError>(())
        })();
        match copy_outcome {
            Ok(()) => {
                fs::rename(&staging_model_directory, &final_model_directory)?;
                Ok(StreamingModelPreparationReport {
                    source_model_id: self.source_model_id.clone(),
                    streaming_model_id: self.streaming_model_id.clone(),
                    model_revision: self.model_revision.clone(),
                    completed_expert_file_count: self.total_expert_file_count(),
                    total_pack_byte_count: total_declared_pack_bytes(&final_model_directory)?,
                    final_model_directory,
                    reused_existing_revision: false,
                    elapsed: preparation_started_at.elapsed(),
                })
            }
            Err(preparation_error) => Err(preparation_error),
        }
    }

    /// Publishes the revision's non-pack files as a single-copy layout: small
    /// discovery files are copied, every non-expert tensor moves into one
    /// `resident.safetensors`, and source shards are never published.
    fn publish_revision_files(
        &self,
        staging_model_directory: &Path,
    ) -> Result<(), AlignedExpertPackPreparationError> {
        // A resumed staging directory may still carry the superseded
        // transitional layout (source shards beside the packs); those files
        // must not survive into the single-copy revision.
        self.remove_superseded_staged_shards(staging_model_directory)?;
        for directory_entry in fs::read_dir(&self.model_directory)? {
            let directory_entry = directory_entry?;
            let source_path = directory_entry.path();
            let file_name = directory_entry.file_name();
            let file_name_text = file_name.to_string_lossy();
            if file_name_text.starts_with('.') {
                continue;
            }
            // Language shards are re-laid-out into resident.safetensors plus
            // the per-expert packs; the shard index describes a shard layout
            // the revision intentionally does not carry.
            if file_name_text.ends_with(".safetensors")
                || file_name_text == "model.safetensors.index.json"
            {
                continue;
            }
            let destination_path = staging_model_directory.join(&file_name);
            if source_path.is_dir() {
                copy_directory_recursively(&source_path, &destination_path)?;
            } else {
                publish_regular_file(&source_path, &destination_path)?;
            }
        }
        super::streaming_model_resident_bundle::publish_resident_weights(
            &self.model_directory,
            &self.layer_plans,
            staging_model_directory,
        )?;
        Ok(())
    }

    /// Publishes `resident.safetensors`: every non-expert tensor from the
    /// source language shards, copied byte-for-byte into one standard
    /// safetensors file at the revision root.
    fn remove_superseded_staged_shards(
        &self,
        staging_model_directory: &Path,
    ) -> Result<(), AlignedExpertPackPreparationError> {
        for directory_entry in fs::read_dir(staging_model_directory)? {
            let directory_entry = directory_entry?;
            let file_name = directory_entry.file_name();
            let file_name_text = file_name.to_string_lossy();
            if directory_entry.path().is_file()
                && (file_name_text.ends_with(".safetensors")
                    || file_name_text == "model.safetensors.index.json")
            {
                fs::remove_file(directory_entry.path())?;
            }
        }
        Ok(())
    }

    fn total_expert_file_count(&self) -> usize {
        self.layer_plans
            .iter()
            .map(|layer_plan| layer_plan.expert_capacity)
            .sum()
    }

    fn staging_has_all_expert_files(&self, staging_model_directory: &Path) -> bool {
        let Some((last_layer_index, last_layer_plan)) = self.layer_plans.iter().enumerate().last()
        else {
            return false;
        };
        let last_expert_id = last_layer_plan.expert_capacity.saturating_sub(1);
        staging_model_directory
            .join(per_expert_pack_relative_path(0, 0))
            .is_file()
            && staging_model_directory
                .join(per_expert_pack_relative_path(
                    last_layer_index,
                    last_expert_id,
                ))
                .is_file()
    }
}

fn existing_staging_directory(
    parent_directory: &Path,
    streaming_model_id: &str,
) -> Result<Option<PathBuf>, AlignedExpertPackPreparationError> {
    let staging_prefix = format!(".{streaming_model_id}.preparing-");
    let mut matching_staging_directories = Vec::new();
    let read_dir = match fs::read_dir(parent_directory) {
        Ok(read_dir) => read_dir,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error.into()),
    };
    for directory_entry in read_dir {
        let directory_entry = directory_entry?;
        let file_name = directory_entry.file_name();
        let Some(file_name_text) = file_name.to_str() else {
            continue;
        };
        if file_name_text.starts_with(&staging_prefix) && directory_entry.path().is_dir() {
            matching_staging_directories.push(directory_entry.path());
        }
    }
    matching_staging_directories.sort();
    Ok(matching_staging_directories.pop())
}

/// Builds the converted public model identity from a source model identity.
#[must_use]
pub fn streaming_model_id_for(source_model_id: &str) -> String {
    format!("{source_model_id}{STREAMING_MODEL_IDENTITY_SUFFIX}")
}

fn total_declared_pack_bytes(
    model_directory: &Path,
) -> Result<u64, AlignedExpertPackPreparationError> {
    let streaming_model_manifest =
        StreamingModelManifest::read_from_revision_directory(model_directory)?;
    streaming_model_manifest.expert_files.iter().try_fold(
        0_u64,
        |total_pack_byte_count, expert_file| {
            total_pack_byte_count
                .checked_add(expert_file.expected_file_byte_count)
                .ok_or(AlignedExpertPackPreparationError::ByteCountOverflow)
        },
    )
}

fn publish_regular_file(
    source_path: &Path,
    destination_path: &Path,
) -> Result<(), AlignedExpertPackPreparationError> {
    if destination_path.exists() {
        return Ok(());
    }
    if fs::hard_link(source_path, destination_path).is_ok() {
        return Ok(());
    }
    fs::copy(source_path, destination_path)?;
    Ok(())
}

fn copy_directory_recursively(
    source_directory: &Path,
    destination_directory: &Path,
) -> Result<(), AlignedExpertPackPreparationError> {
    fs::create_dir_all(destination_directory)?;
    for directory_entry in fs::read_dir(source_directory)? {
        let directory_entry = directory_entry?;
        let source_path = directory_entry.path();
        let destination_path = destination_directory.join(directory_entry.file_name());
        if source_path.is_dir() {
            copy_directory_recursively(&source_path, &destination_path)?;
        } else {
            publish_regular_file(&source_path, &destination_path)?;
        }
    }
    Ok(())
}

pub(crate) fn quantization_mode_name(layer_plan: &QuantizedExpertLayerPlan) -> &'static str {
    match layer_plan.quantization_mode {
        astronomical_model_serving::QuantizationMode::Affine => "affine",
        astronomical_model_serving::QuantizationMode::NativeBfloat16 => "native_bfloat16",
    }
}
