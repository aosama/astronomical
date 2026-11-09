//! Converts a stock safetensors mixture-of-experts checkpoint into an Atlas
//! model directory of its own.
//!
//! Everything is written into a staging directory beside the destination and
//! renamed into place only once every file has been written, read back and
//! declared in a manifest that validates. An interrupted conversion therefore
//! never leaves a half-written model where discovery would find it. The
//! source is only ever read.

use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::atlas_manifest::{
    ATLAS_FORMAT_NAME, ATLAS_FORMAT_VERSION, ATLAS_MANIFEST_FILE_NAME, AtlasDeclaredFile,
    AtlasExpertLayer, AtlasManifest, AtlasSource, EXPERT_LAYER_DIRECTORY_NAME,
};
use crate::layer_record_writer::{LayerRecordWriteError, SourceShardFiles, write_layer_records};
use crate::record_layout::plan_layer_records;
use crate::resident_bundle_writer::{ResidentBundleWriteError, write_resident_bundle};
use crate::source_checkpoint::{SourceCheckpoint, SourceCheckpointError};
use crate::uncached_file_io::available_bytes_on_volume;

/// Free space kept beyond the Atlas itself, so a conversion fails up front
/// rather than filling the volume other writers depend on.
const VOLUME_RESERVE_BYTES: u64 = 4 * 1_000_000_000;

pub struct ConversionRequest<'paths> {
    pub source_directory: &'paths Path,
    pub source_model_id: &'paths str,
    pub output_directory: &'paths Path,
}

/// A snapshot of conversion progress, reported after every unit of work.
#[derive(Clone, Copy, Debug)]
pub struct ConversionProgress {
    pub stage: ConversionStage,
    pub written_byte_count: u64,
    pub total_byte_count: u64,
    pub completed_layer_count: usize,
    pub total_layer_count: usize,
    pub started_at: Instant,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConversionStage {
    ResidentBundle,
    ExpertLayers,
}

pub trait ConversionProgressObserver {
    fn observe(&mut self, conversion_progress: &ConversionProgress);
}

#[derive(Clone, Debug)]
pub struct ConversionSummary {
    pub output_directory: PathBuf,
    pub atlas_byte_count: u64,
    pub expert_layer_count: usize,
    pub resident_tensor_count: usize,
}

#[derive(Debug, thiserror::Error)]
pub enum AtlasConversionError {
    #[error("{0} already exists; remove it or choose another name with --output-model-id")]
    OutputExists(String),
    #[error(
        "the Atlas needs {needed_gigabytes:.1} GB plus {reserve_gigabytes:.0} GB reserve, but {available_gigabytes:.1} GB are free"
    )]
    NotEnoughSpace {
        needed_gigabytes: f64,
        reserve_gigabytes: f64,
        available_gigabytes: f64,
    },
    #[error("{path} could not be accessed: {source}")]
    Io {
        path: String,
        source: std::io::Error,
    },
    #[error(transparent)]
    Source(#[from] SourceCheckpointError),
    #[error(transparent)]
    LayerRecords(#[from] LayerRecordWriteError),
    #[error(transparent)]
    ResidentBundle(#[from] ResidentBundleWriteError),
    #[error("the written Atlas did not validate: {0}")]
    WrittenManifest(String),
}

fn io_error(path: &Path) -> impl FnOnce(std::io::Error) -> AtlasConversionError + '_ {
    move |source| AtlasConversionError::Io {
        path: path.display().to_string(),
        source,
    }
}

pub fn convert_to_atlas(
    conversion_request: &ConversionRequest<'_>,
    progress_observer: &mut dyn ConversionProgressObserver,
) -> Result<ConversionSummary, AtlasConversionError> {
    let output_directory = conversion_request.output_directory;
    if output_directory.exists() {
        return Err(AtlasConversionError::OutputExists(
            output_directory.display().to_string(),
        ));
    }
    let conversion_plan = plan_conversion(conversion_request.source_directory)?;
    let staging_directory =
        prepare_staging_directory(output_directory, conversion_plan.atlas_byte_count)?;
    let publication = write_and_publish_atlas(
        conversion_request,
        &conversion_plan,
        &staging_directory,
        progress_observer,
    );
    if publication.is_err() {
        // A failed conversion can leave tens of gigabytes in a hidden folder
        // that discovery never shows, so nobody would know to remove it.
        // Best effort: the original error is the one worth reporting.
        let _ = std::fs::remove_dir_all(&staging_directory);
    }
    let atlas_manifest = publication?;
    Ok(ConversionSummary {
        output_directory: output_directory.to_path_buf(),
        atlas_byte_count: conversion_plan.atlas_byte_count,
        expert_layer_count: atlas_manifest.expert_layers.len(),
        resident_tensor_count: conversion_plan.source_checkpoint.resident_tensors.len(),
    })
}

/// Writes every Atlas file into `staging_directory`, then renames it into
/// place under its manifest.
fn write_and_publish_atlas(
    conversion_request: &ConversionRequest<'_>,
    conversion_plan: &ConversionPlan,
    staging_directory: &Path,
    progress_observer: &mut dyn ConversionProgressObserver,
) -> Result<AtlasManifest, AtlasConversionError> {
    copy_companion_files(&conversion_plan.source_checkpoint, staging_directory)?;
    let (resident_file, expert_files) =
        write_atlas_files(conversion_plan, staging_directory, progress_observer)?;
    let atlas_manifest = AtlasManifest {
        format: ATLAS_FORMAT_NAME.to_owned(),
        format_version: ATLAS_FORMAT_VERSION,
        source: AtlasSource {
            model_id: conversion_request.source_model_id.to_owned(),
            revision: source_revision(conversion_request.source_directory),
        },
        resident_files: vec![resident_file],
        expert_files,
        expert_layers: conversion_plan.planned_layers.clone(),
    };
    publish_staging_directory(
        staging_directory,
        conversion_request.output_directory,
        &atlas_manifest,
    )?;
    Ok(atlas_manifest)
}

/// The source read and every layer's records laid out, before any write.
struct ConversionPlan {
    source_checkpoint: SourceCheckpoint,
    planned_layers: Vec<AtlasExpertLayer>,
    atlas_byte_count: u64,
}

fn plan_conversion(source_directory: &Path) -> Result<ConversionPlan, AtlasConversionError> {
    let source_checkpoint = SourceCheckpoint::read(source_directory)?;
    let planned_layers: Vec<AtlasExpertLayer> = source_checkpoint
        .expert_layers
        .iter()
        .map(|source_expert_layer| {
            plan_layer_records(source_expert_layer, source_checkpoint.expert_count)
        })
        .collect();
    let atlas_byte_count = planned_layers
        .iter()
        .map(AtlasExpertLayer::file_byte_count)
        .sum::<u64>()
        + source_checkpoint.resident_payload_byte_count();
    Ok(ConversionPlan {
        source_checkpoint,
        planned_layers,
        atlas_byte_count,
    })
}

/// Writes the resident bundle, then every layer's records, reporting progress
/// after each copied piece and each layer.
fn write_atlas_files(
    conversion_plan: &ConversionPlan,
    staging_directory: &Path,
    progress_observer: &mut dyn ConversionProgressObserver,
) -> Result<(AtlasDeclaredFile, Vec<AtlasDeclaredFile>), AtlasConversionError> {
    let source_checkpoint = &conversion_plan.source_checkpoint;
    let mut conversion_progress = ConversionProgress {
        stage: ConversionStage::ResidentBundle,
        written_byte_count: 0,
        total_byte_count: conversion_plan.atlas_byte_count,
        completed_layer_count: 0,
        total_layer_count: conversion_plan.planned_layers.len(),
        started_at: Instant::now(),
    };
    let mut source_shard_files = SourceShardFiles::default();
    let resident_file = write_resident_bundle(
        source_checkpoint,
        staging_directory,
        &mut source_shard_files,
        &mut |copied_byte_count| {
            conversion_progress.written_byte_count += copied_byte_count;
            progress_observer.observe(&conversion_progress);
        },
    )?;
    conversion_progress.stage = ConversionStage::ExpertLayers;
    let mut expert_files = Vec::with_capacity(conversion_plan.planned_layers.len());
    for (source_expert_layer, planned_layer) in source_checkpoint
        .expert_layers
        .iter()
        .zip(&conversion_plan.planned_layers)
    {
        expert_files.push(write_layer_records(
            &source_checkpoint.model_directory,
            source_expert_layer,
            planned_layer,
            staging_directory,
            &mut source_shard_files,
        )?);
        conversion_progress.written_byte_count += planned_layer.file_byte_count();
        conversion_progress.completed_layer_count += 1;
        progress_observer.observe(&conversion_progress);
    }
    Ok((resident_file, expert_files))
}

/// The staging directory for `output_directory`: a hidden sibling, so a
/// half-written Atlas is never mistaken for a model. A staging directory left
/// by an interrupted conversion is this tool's own and is replaced.
fn prepare_staging_directory(
    output_directory: &Path,
    atlas_byte_count: u64,
) -> Result<PathBuf, AtlasConversionError> {
    let parent_directory = output_directory.parent().ok_or_else(|| {
        AtlasConversionError::OutputExists(output_directory.display().to_string())
    })?;
    std::fs::create_dir_all(parent_directory).map_err(io_error(parent_directory))?;
    let available_byte_count =
        available_bytes_on_volume(parent_directory).map_err(io_error(parent_directory))?;
    if available_byte_count < atlas_byte_count + VOLUME_RESERVE_BYTES {
        return Err(AtlasConversionError::NotEnoughSpace {
            needed_gigabytes: atlas_byte_count as f64 / 1e9,
            reserve_gigabytes: VOLUME_RESERVE_BYTES as f64 / 1e9,
            available_gigabytes: available_byte_count as f64 / 1e9,
        });
    }
    let output_name = output_directory
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default();
    let staging_directory = parent_directory.join(format!(".{output_name}.atlas-partial"));
    if staging_directory.exists() {
        std::fs::remove_dir_all(&staging_directory).map_err(io_error(&staging_directory))?;
    }
    std::fs::create_dir_all(staging_directory.join(EXPERT_LAYER_DIRECTORY_NAME))
        .map_err(io_error(&staging_directory))?;
    Ok(staging_directory)
}

/// Copies every file the Atlas keeps as-is: configuration, tokenizer, chat
/// template, sidecar directories. Skipped: the shards and their index (the
/// Atlas replaces them) and hidden entries (Library bookkeeping that belongs
/// to the source). APFS clones these copies, so they cost almost no space.
fn copy_companion_files(
    source_checkpoint: &SourceCheckpoint,
    staging_directory: &Path,
) -> Result<(), AtlasConversionError> {
    let source_directory = &source_checkpoint.model_directory;
    for directory_entry in
        std::fs::read_dir(source_directory).map_err(io_error(source_directory))?
    {
        let directory_entry = directory_entry.map_err(io_error(source_directory))?;
        let entry_name = directory_entry.file_name().to_string_lossy().into_owned();
        let is_replaced = entry_name == "model.safetensors.index.json"
            || source_checkpoint.shard_file_names.contains(&entry_name);
        if entry_name.starts_with('.') || is_replaced || entry_name == ATLAS_MANIFEST_FILE_NAME {
            continue;
        }
        copy_entry(
            &directory_entry.path(),
            &staging_directory.join(&entry_name),
        )?;
    }
    Ok(())
}

fn copy_entry(source_path: &Path, destination_path: &Path) -> Result<(), AtlasConversionError> {
    let entry_metadata = std::fs::symlink_metadata(source_path).map_err(io_error(source_path))?;
    if entry_metadata.is_dir() {
        std::fs::create_dir_all(destination_path).map_err(io_error(destination_path))?;
        for directory_entry in std::fs::read_dir(source_path).map_err(io_error(source_path))? {
            let directory_entry = directory_entry.map_err(io_error(source_path))?;
            copy_entry(
                &directory_entry.path(),
                &destination_path.join(directory_entry.file_name()),
            )?;
        }
        return Ok(());
    }
    if entry_metadata.is_file() {
        std::fs::copy(source_path, destination_path).map_err(io_error(source_path))?;
    }
    Ok(())
}

/// The source revision a Library download recorded, when there is one.
fn source_revision(source_directory: &Path) -> Option<String> {
    let provenance_bytes =
        std::fs::read(source_directory.join(".astronomical-library-provenance.json")).ok()?;
    let provenance_document: serde_json::Value = serde_json::from_slice(&provenance_bytes).ok()?;
    provenance_document
        .get("revision")?
        .as_str()
        .map(str::to_owned)
}

/// Writes the manifest, checks it reads back as a valid Atlas whose declared
/// files all exist with their declared sizes, then renames into place.
fn publish_staging_directory(
    staging_directory: &Path,
    output_directory: &Path,
    atlas_manifest: &AtlasManifest,
) -> Result<(), AtlasConversionError> {
    let manifest_path = staging_directory.join(ATLAS_MANIFEST_FILE_NAME);
    let manifest_bytes = serde_json::to_vec_pretty(atlas_manifest)
        .map_err(|source| AtlasConversionError::WrittenManifest(source.to_string()))?;
    std::fs::write(&manifest_path, manifest_bytes).map_err(io_error(&manifest_path))?;
    let written_manifest = AtlasManifest::read_from_directory(staging_directory)
        .map_err(|source| AtlasConversionError::WrittenManifest(source.to_string()))?;
    for declared_file in written_manifest
        .resident_files
        .iter()
        .chain(&written_manifest.expert_files)
    {
        check_declared_file(staging_directory, declared_file)?;
    }
    std::fs::rename(staging_directory, output_directory).map_err(io_error(output_directory))
}

fn check_declared_file(
    staging_directory: &Path,
    declared_file: &AtlasDeclaredFile,
) -> Result<(), AtlasConversionError> {
    let declared_path = staging_directory.join(&declared_file.file_name);
    let actual_byte_count = std::fs::metadata(&declared_path)
        .map_err(io_error(&declared_path))?
        .len();
    if actual_byte_count != declared_file.expected_file_byte_count {
        return Err(AtlasConversionError::WrittenManifest(format!(
            "{} has {actual_byte_count} bytes, the manifest declares {}",
            declared_file.file_name, declared_file.expected_file_byte_count
        )));
    }
    Ok(())
}

#[cfg(test)]
#[path = "atlas_conversion_tests.rs"]
mod atlas_conversion_tests;
