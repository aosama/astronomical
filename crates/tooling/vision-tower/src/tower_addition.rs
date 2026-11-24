//! Builds a new model from a local text model and a vision tower.
//!
//! Nothing here touches the text weights: they are cloned, so what ships is exactly
//! what their publisher released. The base match runs first and cannot be skipped,
//! because a tower paired with the wrong weights still loads and then quietly
//! produces poor answers about images, which no later step would catch.
//!
//! The model is assembled beside its destination and renamed into place only once
//! it has passed its final check, so an interrupted run never leaves something at
//! the output path that looks finished.

use std::fs;
use std::path::{Path, PathBuf};

use serde::Serialize;
use serde_json::Value;

use crate::artifact_source::{
    ArtifactFile, ArtifactSource, CONFIG_FILE_NAME, WEIGHT_INDEX_FILE_NAME,
};
use crate::assembled_documents::{
    TOWER_FILE_RELATIVE_PATH, TowerDocumentFacts, config_with_tower, index_with_tower,
};
use crate::base_matching::{BaseMatchProgress, BaseMatchReport, match_tower_to_text};
use crate::build_record::{
    BUILD_RECORD_FILE_NAME, BuildRecord, FileRecord, SourceRecord, hash_file,
};
use crate::error::VisionTowerError;
use crate::local_directory_source::LocalDirectorySource;
use crate::tensor_names::{QUANTIZATION_SCALES_SUFFIX, is_vision_tower_tensor};
use crate::tower_file::{
    PlannedTowerTensor, TowerFileProgress, WrittenTowerFile, plan_tower_tensors, write_tower_file,
};
use crate::tower_inspection::{TowerStatus, inspect_artifact};
use crate::weight_inventory::{WeightInventory, read_weight_inventory};

const MAXIMUM_DOCUMENT_BYTES: u64 = 64 * 1024 * 1024;
// A tower is useless without these for loaders that read them, so they come from
// the tower's side whenever the text model dropped them with its tower.
const IMAGE_SETTINGS_FILES: [&str; 3] = [
    "preprocessor_config.json",
    "processor_config.json",
    "video_preprocessor_config.json",
];
// These describe the source repository, not the new model. The upload tool writes
// its own `.gitattributes`, and a new model needs its own card.
const FILES_NOT_CARRIED_OVER: [&str; 2] = ["README.md", ".gitattributes"];
const FOLDERS_NOT_CARRIED_OVER: [&str; 1] = ["assets/"];
const INCOMPLETE_SUFFIX: &str = ".incomplete";

#[derive(Clone, Copy, Debug)]
pub struct TowerAdditionRequest<'request> {
    pub text_source: &'request ArtifactSource,
    pub tower_source: &'request ArtifactSource,
    pub output_directory: &'request Path,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TowerAdditionStage {
    FetchingTower,
    HashingFiles,
}

/// Told about each stage's bytes, on top of the base match's own progress.
pub trait TowerAdditionProgress: BaseMatchProgress {
    fn stage_bytes(&mut self, stage: TowerAdditionStage, completed_bytes: u64, total_bytes: u64);
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct TowerAdditionReport {
    pub output_directory: PathBuf,
    pub base_match: BaseMatchReport,
    pub tower_tensor_count: u64,
    pub tower_byte_count: u64,
    pub file_count: u64,
    pub total_byte_count: u64,
    pub missing_image_settings: Vec<String>,
}

pub async fn add_tower(
    request: TowerAdditionRequest<'_>,
    progress: &mut dyn TowerAdditionProgress,
) -> Result<TowerAdditionReport, VisionTowerError> {
    let ArtifactSource::LocalDirectory(text_directory) = request.text_source else {
        return Err(VisionTowerError::TextModelNotLocal {
            reference: request.text_source.listing().display_name.clone(),
        });
    };
    let incomplete_directory = incomplete_directory_for(request.output_directory)?;
    refuse_existing_output(request.output_directory, &incomplete_directory)?;
    let base_match =
        match_tower_to_text(request.text_source, request.tower_source, &mut *progress).await?;
    refuse_unsuitable_match(&base_match)?;
    let tower_inventory = read_weight_inventory(request.tower_source).await?;
    refuse_quantized_tower(&tower_inventory)?;
    let planned_tensors = plan_tower_tensors(&tower_inventory)?;

    if let Err(write_error) = fs::create_dir_all(&incomplete_directory) {
        return Err(unwritable(&incomplete_directory, &write_error));
    }
    let assembly = Assembly {
        text_source: request.text_source,
        text_directory,
        tower_source: request.tower_source,
        planned_tensors: &planned_tensors,
        base_match: &base_match,
        directory: &incomplete_directory,
    };
    let assembled = match assembly.run(progress).await {
        Ok(assembled) => assembled,
        Err(assembly_error) => {
            // Our own half-built folder, under a name only this tool uses.
            let _ = fs::remove_dir_all(&incomplete_directory);
            return Err(assembly_error);
        }
    };
    if let Err(write_error) = fs::rename(&incomplete_directory, request.output_directory) {
        return Err(unwritable(request.output_directory, &write_error));
    }
    Ok(TowerAdditionReport {
        output_directory: request.output_directory.to_path_buf(),
        base_match,
        tower_tensor_count: planned_tensors.len() as u64,
        tower_byte_count: assembled.tower_file.tensor_byte_count,
        file_count: assembled.file_count,
        total_byte_count: assembled.total_byte_count,
        missing_image_settings: assembled.missing_image_settings,
    })
}

/// A sibling of the output, built from the path's last name so that `out/` and `out`
/// mean the same folder. A path with no last name, such as `..`, is one that exists.
fn incomplete_directory_for(output_directory: &Path) -> Result<PathBuf, VisionTowerError> {
    let Some(output_name) = output_directory.file_name() else {
        return Err(VisionTowerError::OutputAlreadyExists {
            path: output_directory.to_path_buf(),
        });
    };
    let mut incomplete_name = output_name.to_os_string();
    incomplete_name.push(INCOMPLETE_SUFFIX);
    Ok(output_directory.with_file_name(incomplete_name))
}

fn refuse_existing_output(
    output_directory: &Path,
    incomplete_directory: &Path,
) -> Result<(), VisionTowerError> {
    if output_directory.exists() {
        return Err(VisionTowerError::OutputAlreadyExists {
            path: output_directory.to_path_buf(),
        });
    }
    if incomplete_directory.exists() {
        return Err(VisionTowerError::IncompleteOutputPresent {
            path: incomplete_directory.to_path_buf(),
        });
    }
    Ok(())
}

fn refuse_unsuitable_match(base_match: &BaseMatchReport) -> Result<(), VisionTowerError> {
    if base_match.text_already_has_tower {
        return Err(VisionTowerError::AdditionRefused {
            reason: format!("{} already has a vision tower", base_match.text_artifact),
        });
    }
    if base_match.verdict.allows_adding() {
        return Ok(());
    }
    let reason = if base_match.incompatibilities.is_empty() {
        format!(
            "the base match says {:?}, so nothing shows this tower belongs with these weights",
            base_match.verdict
        )
    } else {
        base_match.incompatibilities.join("; ")
    };
    Err(VisionTowerError::AdditionRefused { reason })
}

/// Astronomical refuses a quantized tower, and nothing is gained by fetching one.
fn refuse_quantized_tower(tower_inventory: &WeightInventory) -> Result<(), VisionTowerError> {
    for tensor_name in tower_inventory.tensors.keys() {
        if is_vision_tower_tensor(tensor_name) && tensor_name.ends_with(QUANTIZATION_SCALES_SUFFIX)
        {
            return Err(VisionTowerError::AdditionRefused {
                reason: "this tower is quantized, and only full-precision towers are supported"
                    .to_owned(),
            });
        }
    }
    Ok(())
}

struct Assembly<'assembly> {
    text_source: &'assembly ArtifactSource,
    text_directory: &'assembly LocalDirectorySource,
    tower_source: &'assembly ArtifactSource,
    planned_tensors: &'assembly [PlannedTowerTensor<'assembly>],
    base_match: &'assembly BaseMatchReport,
    directory: &'assembly Path,
}

struct AssembledModel {
    tower_file: WrittenTowerFile,
    file_count: u64,
    total_byte_count: u64,
    missing_image_settings: Vec<String>,
}

impl Assembly<'_> {
    async fn run(
        &self,
        progress: &mut dyn TowerAdditionProgress,
    ) -> Result<AssembledModel, VisionTowerError> {
        self.clone_text_files()?;
        let tower_file = write_tower_file(
            self.tower_source,
            self.planned_tensors,
            &self.directory.join(TOWER_FILE_RELATIVE_PATH),
            &mut TowerFileProgressAdapter {
                progress: &mut *progress,
            },
        )
        .await?;
        self.write_documents(&tower_file).await?;
        let missing_image_settings = self.copy_image_settings().await?;
        self.check_assembled_model(&tower_file).await?;
        let file_records = self.hash_all_files(progress)?;
        self.write_build_record(file_records.clone())?;
        let mut total_byte_count = 0_u64;
        for file_record in &file_records {
            total_byte_count += file_record.byte_count;
        }
        Ok(AssembledModel {
            tower_file,
            file_count: file_records.len() as u64 + 1,
            total_byte_count,
            missing_image_settings,
        })
    }

    /// `fs::copy` clones on APFS (`fclonefileat`), so a 5 GB weight file costs almost
    /// no disk and no time, and it falls back to a real copy on other filesystems. It
    /// also follows symlinks, so a Hugging Face snapshot becomes ordinary files.
    fn clone_text_files(&self) -> Result<(), VisionTowerError> {
        let listing = self.text_directory.listing();
        for artifact_file in &listing.files {
            if !should_carry_over(artifact_file) {
                continue;
            }
            let relative_path = &artifact_file.relative_path;
            let destination = self.directory.join(relative_path);
            if let Some(parent_directory) = destination.parent()
                && let Err(write_error) = fs::create_dir_all(parent_directory)
            {
                return Err(unwritable(parent_directory, &write_error));
            }
            let source_path = self.text_directory.root_directory().join(relative_path);
            if let Err(write_error) = fs::copy(&source_path, &destination) {
                return Err(unwritable(&destination, &write_error));
            }
        }
        Ok(())
    }

    async fn write_documents(&self, tower_file: &WrittenTowerFile) -> Result<(), VisionTowerError> {
        let text_config = read_json_document(self.text_source, CONFIG_FILE_NAME).await?;
        let tower_config = read_json_document(self.tower_source, CONFIG_FILE_NAME).await?;
        let text_index = read_json_document(self.text_source, WEIGHT_INDEX_FILE_NAME).await?;
        let text_inventory = read_weight_inventory(self.text_source).await?;
        let empty_config = Value::Object(serde_json::Map::new());
        let tower_config = tower_config.unwrap_or_default();
        let facts = TowerDocumentFacts {
            tower_config: &tower_config,
            tensor_names: &tower_file.tensor_names,
            tensor_byte_count: tower_file.tensor_byte_count,
            parameter_count: tower_file.parameter_count,
        };
        let merged_config =
            config_with_tower(text_config.as_ref().unwrap_or(&empty_config), &facts);
        self.write_json(CONFIG_FILE_NAME, &merged_config)?;
        let mut text_tensor_files = Vec::with_capacity(text_inventory.tensors.len());
        for (tensor_name, located_tensor) in &text_inventory.tensors {
            text_tensor_files.push((tensor_name.clone(), located_tensor.relative_path.clone()));
        }
        let merged_index = index_with_tower(text_index.as_ref(), &text_tensor_files, &facts);
        self.write_json(WEIGHT_INDEX_FILE_NAME, &merged_index)
    }

    /// Returns the settings neither side had, which the report passes on, since a
    /// model can still run in Astronomical without them while other loaders cannot.
    async fn copy_image_settings(&self) -> Result<Vec<String>, VisionTowerError> {
        let mut missing_image_settings = Vec::new();
        for settings_file in IMAGE_SETTINGS_FILES {
            let destination = self.directory.join(settings_file);
            if destination.exists() {
                continue;
            }
            let settings_bytes = self
                .tower_source
                .read_whole_file(settings_file, MAXIMUM_DOCUMENT_BYTES)
                .await?;
            match settings_bytes {
                Some(settings_bytes) => {
                    if let Err(write_error) = fs::write(&destination, settings_bytes) {
                        return Err(unwritable(&destination, &write_error));
                    }
                }
                None => missing_image_settings.push(settings_file.to_owned()),
            }
        }
        Ok(missing_image_settings)
    }

    /// Reads the result back the way a loader would, so a mistake in this code fails
    /// here rather than after someone has downloaded the model.
    async fn check_assembled_model(
        &self,
        tower_file: &WrittenTowerFile,
    ) -> Result<(), VisionTowerError> {
        let assembled_source =
            ArtifactSource::LocalDirectory(LocalDirectorySource::open(self.directory)?);
        let inspection = inspect_artifact(&assembled_source).await?;
        if inspection.status != TowerStatus::Present {
            return Err(assembled_invalid(format!(
                "it reads back as {:?} rather than having a whole tower",
                inspection.status
            )));
        }
        let Some(tower) = inspection.tower else {
            return Err(assembled_invalid("it has no tower".to_owned()));
        };
        if tower.is_in_weight_index != Some(true) {
            return Err(assembled_invalid(
                "the weight index does not list the tower".to_owned(),
            ));
        }
        if tower.tensor_count != tower_file.tensor_names.len() as u64
            || tower.byte_count != tower_file.tensor_byte_count
        {
            return Err(assembled_invalid(format!(
                "the tower reads back as {} tensors and {} bytes, but {} and {} were written",
                tower.tensor_count,
                tower.byte_count,
                tower_file.tensor_names.len(),
                tower_file.tensor_byte_count
            )));
        }
        Ok(())
    }

    fn hash_all_files(
        &self,
        progress: &mut dyn TowerAdditionProgress,
    ) -> Result<Vec<FileRecord>, VisionTowerError> {
        let assembled_listing = LocalDirectorySource::open(self.directory)?;
        let mut total_bytes = 0_u64;
        for artifact_file in &assembled_listing.listing().files {
            total_bytes += artifact_file.byte_count.unwrap_or(0);
        }
        let mut completed_bytes = 0_u64;
        let mut file_records = Vec::with_capacity(assembled_listing.listing().files.len());
        for artifact_file in &assembled_listing.listing().files {
            let (sha256, byte_count) =
                hash_file(&self.directory.join(&artifact_file.relative_path))?;
            completed_bytes += byte_count;
            progress.stage_bytes(
                TowerAdditionStage::HashingFiles,
                completed_bytes,
                total_bytes,
            );
            file_records.push(FileRecord {
                path: artifact_file.relative_path.clone(),
                byte_count,
                sha256,
            });
        }
        Ok(file_records)
    }

    fn write_build_record(&self, file_records: Vec<FileRecord>) -> Result<(), VisionTowerError> {
        let text_listing = self.text_directory.listing();
        let tower_listing = self.tower_source.listing();
        let build_record = BuildRecord::new(
            SourceRecord {
                repository: text_listing.origin_repository.clone(),
                revision: text_listing.pinned_revision.clone(),
            },
            SourceRecord {
                repository: tower_listing.origin_repository.clone(),
                revision: tower_listing.pinned_revision.clone(),
            },
            self.base_match,
            file_records,
        );
        let record_value = match serde_json::to_value(&build_record) {
            Ok(record_value) => record_value,
            Err(serialize_error) => {
                return Err(assembled_invalid(serialize_error.to_string()));
            }
        };
        self.write_json(BUILD_RECORD_FILE_NAME, &record_value)
    }

    fn write_json(&self, file_name: &str, document: &Value) -> Result<(), VisionTowerError> {
        let file_path = self.directory.join(file_name);
        let mut document_text = match serde_json::to_string_pretty(document) {
            Ok(document_text) => document_text,
            Err(serialize_error) => {
                return Err(assembled_invalid(serialize_error.to_string()));
            }
        };
        document_text.push('\n');
        match fs::write(&file_path, document_text) {
            Ok(()) => Ok(()),
            Err(write_error) => Err(unwritable(&file_path, &write_error)),
        }
    }
}

struct TowerFileProgressAdapter<'progress> {
    progress: &'progress mut dyn TowerAdditionProgress,
}

impl TowerFileProgress for TowerFileProgressAdapter<'_> {
    fn tower_bytes_written(&mut self, completed_bytes: u64, total_bytes: u64) {
        self.progress.stage_bytes(
            TowerAdditionStage::FetchingTower,
            completed_bytes,
            total_bytes,
        );
    }
}

fn should_carry_over(artifact_file: &ArtifactFile) -> bool {
    let relative_path = artifact_file.relative_path.as_str();
    if relative_path == CONFIG_FILE_NAME || relative_path == WEIGHT_INDEX_FILE_NAME {
        return false;
    }
    if FILES_NOT_CARRIED_OVER.contains(&relative_path) {
        return false;
    }
    for excluded_folder in FOLDERS_NOT_CARRIED_OVER {
        if relative_path.starts_with(excluded_folder) {
            return false;
        }
    }
    true
}

async fn read_json_document(
    source: &ArtifactSource,
    file_name: &str,
) -> Result<Option<Value>, VisionTowerError> {
    let Some(document_bytes) = source
        .read_whole_file(file_name, MAXIMUM_DOCUMENT_BYTES)
        .await?
    else {
        return Ok(None);
    };
    match serde_json::from_slice(&document_bytes) {
        Ok(document) => Ok(Some(document)),
        Err(parse_error) => Err(VisionTowerError::JsonMalformed {
            file: file_name.to_owned(),
            description: parse_error.to_string(),
        }),
    }
}

fn unwritable(path: &Path, write_error: &std::io::Error) -> VisionTowerError {
    VisionTowerError::FileUnwritable {
        path: path.to_path_buf(),
        cause: write_error.to_string(),
    }
}

const fn assembled_invalid(description: String) -> VisionTowerError {
    VisionTowerError::AssembledModelInvalid { description }
}

#[cfg(test)]
#[path = "tower_addition_tests.rs"]
mod tests;
