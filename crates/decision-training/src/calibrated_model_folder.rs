//! Writing a calibrated copy of a decision model folder.
//!
//! The source folder is only ever read. Writing into it, or into memory
//! mapped from it, is how a tuning tool damages the very weights it was
//! meant to improve, so the calibrated model is a new folder: every source
//! file copied unchanged, plus `decision_calibration.json`.
//!
//! The copy is then proven: every copied file is read back and compared
//! byte for byte with its source. On APFS `std::fs::copy` clones a file
//! instead of duplicating its blocks, so the copy itself is nearly free and
//! the comparison is the cost, about one read of the model per side.
//!
//! Hidden entries, such as the library's provenance stamp and the download
//! bookkeeping under `.cache`, are left behind, as `vision add` leaves them:
//! they describe the published model, and the copy is no longer it. A stamp
//! carried over would make discovery report the copy as the catalog model at
//! the catalog revision.
//!
//! A folder can also hold other models, such as the multilingual checkpoint
//! a Laya repository bundles in a subfolder. The caller names those, and they
//! are left out: the calibration was fitted to one model, and a copy of the
//! other would be served under a name claiming a calibration it never had.
//!
//! The folder is assembled under a hidden sibling name and renamed into place
//! only once it is complete and proven, so an interrupted run never leaves a
//! folder the daemon would discover and serve.

use std::fs;
use std::io::{self, Read};
use std::path::{Path, PathBuf};

use astronomical_decision_contract::calibration_file::{
    CalibrationOverride, DECISION_CALIBRATION_FILE_NAME, calibration_document_text,
};
use astronomical_decision_contract::decision_adapters_file::{
    DECISION_ADAPTERS_FILE_NAME, DecisionAdapters, decision_adapters_bytes, parse_decision_adapters,
};
use astronomical_decision_contract::head_tuning_file::{
    DECISION_HEAD_TUNING_FILE_NAME, HeadTuning, TunableHeadFamily, head_tuning_bytes,
    parse_head_tuning,
};
use thiserror::Error;

/// Bytes compared per read; large enough that a 10 GB model compares at disk speed.
const COMPARISON_READ_BYTES: usize = 1024 * 1024;

/// Told how far the copy-and-prove pass has got.
pub trait FolderCopyProgress {
    /// `processed_bytes` of `total_bytes`, where the total is the size of the
    /// files copied. Copying and comparing are two passes over those bytes,
    /// so each pass moves the count by half its bytes: a person reads the
    /// model's own size, not twice it.
    fn report(&mut self, processed_bytes: u64, total_bytes: u64);
}

/// Why the calibrated folder could not be written.
#[derive(Debug, Error)]
pub enum CalibratedFolderError {
    #[error("{output_directory} already exists; choose a new folder so nothing is overwritten")]
    OutputExists { output_directory: PathBuf },
    #[error("the output folder must not be inside the model folder, or the model folder inside it")]
    OutputOverlapsSource,
    #[error("{path}: {source}")]
    Io { path: PathBuf, source: io::Error },
    #[error("the copy of {relative_path} differs from the original; nothing was written")]
    CopyDiffers { relative_path: PathBuf },
    #[error("the trained head could not be written: {0}")]
    TuningUnwritable(String),
    #[error("the trained head read back differently from what was written; nothing was kept")]
    TuningReadsBackDifferently,
    #[error("the add-on weights could not be written: {0}")]
    AdaptersUnwritable(String),
    #[error("the add-on weights read back differently from what was written; nothing was kept")]
    AdaptersReadBackDifferently,
}

/// What the new folder holds beside the copied model.
#[derive(Clone, Copy)]
pub struct FolderAdditions<'a> {
    /// Fitted temperatures; `None` leaves the copy's own temperatures alone.
    pub calibration_override: Option<&'a CalibrationOverride>,
    /// A trained head and the family it belongs to; `None` keeps the copy's
    /// own trained head, if it has one, since that head is what the
    /// calibration was fitted to.
    pub head_tuning: Option<(&'a HeadTuning, TunableHeadFamily)>,
    /// Add-on weights from a whole-model training; `None` keeps the copy's
    /// own, if it has any.
    pub decision_adapters: Option<&'a DecisionAdapters>,
}

impl FolderAdditions<'_> {
    /// The files this folder writes fresh, which the copy leaves out:
    /// an earlier version is replaced, never stacked under the new one.
    fn replaced_file_names(&self) -> Vec<&'static str> {
        let mut replaced_file_names = Vec::new();
        if self.calibration_override.is_some() {
            replaced_file_names.push(DECISION_CALIBRATION_FILE_NAME);
        }
        if self.head_tuning.is_some() {
            replaced_file_names.push(DECISION_HEAD_TUNING_FILE_NAME);
        }
        if self.decision_adapters.is_some() {
            replaced_file_names.push(DECISION_ADAPTERS_FILE_NAME);
        }
        replaced_file_names
    }
}

/// The model folder to copy and the subfolders that hold other models.
pub struct SourceModelFolder<'a> {
    pub model_directory: &'a Path,
    /// Relative to `model_directory`; left out of the copy.
    pub other_model_subdirectories: &'a [PathBuf],
}

/// Copies the source model to `output_directory`, proves the copy and adds
/// the new files.
pub fn write_calibrated_model_folder(
    source_model_folder: &SourceModelFolder<'_>,
    output_directory: &Path,
    folder_additions: FolderAdditions<'_>,
    copy_progress: &mut dyn FolderCopyProgress,
) -> Result<(), CalibratedFolderError> {
    let source_directory = source_model_folder.model_directory;
    check_output_location(source_directory, output_directory)?;
    let source_files = source_files(
        source_directory,
        source_model_folder.other_model_subdirectories,
        &folder_additions.replaced_file_names(),
    )?;
    let partial_directory = partial_directory_for(output_directory);
    let assembly_outcome = match assemble_folder(
        &partial_directory,
        &source_files,
        source_directory,
        copy_progress,
    ) {
        Ok(()) => write_additions(&partial_directory, folder_additions),
        Err(assembly_error) => Err(assembly_error),
    };
    if let Err(assembly_error) = assembly_outcome {
        // Only the folder this run created is removed; the source is never touched.
        let _ = fs::remove_dir_all(&partial_directory);
        return Err(assembly_error);
    }
    fs::rename(&partial_directory, output_directory).at_path(output_directory)
}

/// Refuses an output folder that exists or that overlaps the model folder.
///
/// Run by [`write_calibrated_model_folder`] itself, and exposed so a caller
/// can refuse a bad output before spending minutes scoring examples.
pub fn check_output_location(
    source_directory: &Path,
    output_directory: &Path,
) -> Result<(), CalibratedFolderError> {
    if output_directory.exists() {
        return Err(CalibratedFolderError::OutputExists {
            output_directory: output_directory.to_path_buf(),
        });
    }
    let canonical_source = canonical_path(source_directory)?;
    let canonical_output = canonical_output_path(output_directory)?;
    if canonical_output.starts_with(&canonical_source)
        || canonical_source.starts_with(&canonical_output)
    {
        return Err(CalibratedFolderError::OutputOverlapsSource);
    }
    Ok(())
}

/// `output_directory` as an absolute path with links resolved, even when it
/// and some of its parents do not exist yet: the nearest folder that exists
/// is resolved and the missing names are added back. The folder is written
/// with its parents created, so a new `calibrated/` folder is a fine target.
///
/// # Errors
/// Returns the folder error when the nearest existing folder cannot be resolved.
pub fn canonical_output_path(output_directory: &Path) -> Result<PathBuf, CalibratedFolderError> {
    let mut missing_names = Vec::new();
    let mut existing_ancestor = output_directory;
    while !existing_ancestor.as_os_str().is_empty() && !existing_ancestor.exists() {
        let Some(missing_name) = existing_ancestor.file_name() else {
            break;
        };
        missing_names.push(missing_name.to_owned());
        existing_ancestor = existing_ancestor.parent().unwrap_or(Path::new(""));
    }
    let mut resolved_output = canonical_path(existing_ancestor)?;
    for missing_name in missing_names.iter().rev() {
        resolved_output.push(missing_name);
    }
    Ok(resolved_output)
}

fn canonical_path(path: &Path) -> Result<PathBuf, CalibratedFolderError> {
    let path_to_resolve = if path.as_os_str().is_empty() {
        Path::new(".")
    } else {
        path
    };
    fs::canonicalize(path_to_resolve).at_path(path)
}

/// A source file and its size; symbolic links (as in a Hugging Face cache
/// snapshot) are followed, so the copy holds real files.
struct SourceFile {
    relative_path: PathBuf,
    byte_count: u64,
}

fn source_files(
    source_directory: &Path,
    other_model_subdirectories: &[PathBuf],
    replaced_file_names: &[&str],
) -> Result<Vec<SourceFile>, CalibratedFolderError> {
    let mut found_files = Vec::new();
    let mut pending_directories = vec![PathBuf::new()];
    while let Some(relative_directory) = pending_directories.pop() {
        let directory_path = source_directory.join(&relative_directory);
        let directory_entries = fs::read_dir(&directory_path).at_path(&directory_path)?;
        for directory_entry in directory_entries {
            let entry_path = directory_entry.at_path(&directory_path)?.path();
            let entry_name = entry_path.file_name().unwrap_or_default();
            if entry_name.to_string_lossy().starts_with('.') {
                continue;
            }
            let relative_path = relative_directory.join(entry_name);
            let entry_metadata = fs::metadata(&entry_path).at_path(&entry_path)?;
            if entry_metadata.is_dir() {
                if !other_model_subdirectories.contains(&relative_path) {
                    pending_directories.push(relative_path);
                }
            } else if !is_replaced(&relative_path, replaced_file_names) {
                found_files.push(SourceFile {
                    relative_path,
                    byte_count: entry_metadata.len(),
                });
            }
        }
    }
    found_files.sort_unstable_by(compare_relative_paths);
    Ok(found_files)
}

/// Whether `relative_path` is a top-level file the new folder writes fresh.
fn is_replaced(relative_path: &Path, replaced_file_names: &[&str]) -> bool {
    for replaced_file_name in replaced_file_names {
        if relative_path == Path::new(replaced_file_name) {
            return true;
        }
    }
    false
}

fn compare_relative_paths(left_file: &SourceFile, right_file: &SourceFile) -> std::cmp::Ordering {
    left_file.relative_path.cmp(&right_file.relative_path)
}

fn partial_directory_for(output_directory: &Path) -> PathBuf {
    let output_name = output_directory
        .file_name()
        .unwrap_or_default()
        .to_string_lossy();
    output_directory.with_file_name(format!(".{output_name}.partial"))
}

fn assemble_folder(
    partial_directory: &Path,
    source_files: &[SourceFile],
    source_directory: &Path,
    copy_progress: &mut dyn FolderCopyProgress,
) -> Result<(), CalibratedFolderError> {
    let mut total_bytes = 0;
    for source_file in source_files {
        total_bytes += source_file.byte_count;
    }
    // Counted over both passes and halved when reported, so the count ends
    // exactly at the model's size.
    let mut processed_pass_bytes = 0;
    for source_file in source_files {
        let source_path = source_directory.join(&source_file.relative_path);
        let copy_path = partial_directory.join(&source_file.relative_path);
        create_parent_directory(&copy_path)?;
        fs::copy(&source_path, &copy_path).at_path(&copy_path)?;
        processed_pass_bytes += source_file.byte_count;
        copy_progress.report(processed_pass_bytes / 2, total_bytes);
    }
    for source_file in source_files {
        let source_path = source_directory.join(&source_file.relative_path);
        let copy_path = partial_directory.join(&source_file.relative_path);
        if !files_are_identical(&source_path, &copy_path)? {
            return Err(CalibratedFolderError::CopyDiffers {
                relative_path: source_file.relative_path.clone(),
            });
        }
        processed_pass_bytes += source_file.byte_count;
        copy_progress.report(processed_pass_bytes / 2, total_bytes);
    }
    Ok(())
}

/// Writes the calibration, the trained head and the add-ons, then reads the
/// trained files back and requires exactly what was written: the files
/// serving will read are checked, not the values held in memory.
fn write_additions(
    partial_directory: &Path,
    folder_additions: FolderAdditions<'_>,
) -> Result<(), CalibratedFolderError> {
    if let Some(calibration_override) = folder_additions.calibration_override {
        let calibration_path = partial_directory.join(DECISION_CALIBRATION_FILE_NAME);
        fs::write(
            &calibration_path,
            calibration_document_text(calibration_override),
        )
        .at_path(&calibration_path)?;
    }
    if let Some((head_tuning, family)) = folder_additions.head_tuning {
        write_head_tuning(partial_directory, head_tuning, family)?;
    }
    if let Some(decision_adapters) = folder_additions.decision_adapters {
        write_decision_adapters(partial_directory, decision_adapters)?;
    }
    Ok(())
}

fn write_head_tuning(
    partial_directory: &Path,
    head_tuning: &HeadTuning,
    family: TunableHeadFamily,
) -> Result<(), CalibratedFolderError> {
    let tuning_path = partial_directory.join(DECISION_HEAD_TUNING_FILE_NAME);
    let tuning_bytes = match head_tuning_bytes(head_tuning) {
        Ok(tuning_bytes) => tuning_bytes,
        Err(write_error) => return Err(CalibratedFolderError::TuningUnwritable(write_error)),
    };
    fs::write(&tuning_path, &tuning_bytes).at_path(&tuning_path)?;
    let written_bytes = fs::read(&tuning_path).at_path(&tuning_path)?;
    match parse_head_tuning(&written_bytes, family) {
        Ok(read_back) if read_back == sorted_tuning(head_tuning) => Ok(()),
        _ => Err(CalibratedFolderError::TuningReadsBackDifferently),
    }
}

fn write_decision_adapters(
    partial_directory: &Path,
    decision_adapters: &DecisionAdapters,
) -> Result<(), CalibratedFolderError> {
    let adapters_path = partial_directory.join(DECISION_ADAPTERS_FILE_NAME);
    let adapters_bytes = match decision_adapters_bytes(decision_adapters) {
        Ok(adapters_bytes) => adapters_bytes,
        Err(write_error) => return Err(CalibratedFolderError::AdaptersUnwritable(write_error)),
    };
    fs::write(&adapters_path, &adapters_bytes).at_path(&adapters_path)?;
    let written_bytes = fs::read(&adapters_path).at_path(&adapters_path)?;
    let mut sorted_adapters = decision_adapters.clone();
    sorted_adapters.adapters.sort_by(compare_adapter_targets);
    match parse_decision_adapters(&written_bytes) {
        Ok(read_back) if read_back == sorted_adapters => Ok(()),
        _ => Err(CalibratedFolderError::AdaptersReadBackDifferently),
    }
}

/// The tuning with its tensors in name order, as a reader returns them.
fn sorted_tuning(head_tuning: &HeadTuning) -> HeadTuning {
    let mut sorted = head_tuning.clone();
    sorted.tensors.sort_by(compare_tensor_names);
    sorted
}

fn compare_tensor_names(
    left_tensor: &astronomical_decision_contract::head_tuning_file::TunedTensor,
    right_tensor: &astronomical_decision_contract::head_tuning_file::TunedTensor,
) -> std::cmp::Ordering {
    left_tensor.name.cmp(&right_tensor.name)
}

fn compare_adapter_targets(
    left_adapter: &astronomical_decision_contract::decision_adapters_file::LinearAdapter,
    right_adapter: &astronomical_decision_contract::decision_adapters_file::LinearAdapter,
) -> std::cmp::Ordering {
    left_adapter.target.cmp(&right_adapter.target)
}

fn create_parent_directory(copy_path: &Path) -> Result<(), CalibratedFolderError> {
    let Some(parent_directory) = copy_path.parent() else {
        return Ok(());
    };
    fs::create_dir_all(parent_directory).at_path(parent_directory)
}

/// Whether two files hold exactly the same bytes, read side by side.
fn files_are_identical(
    source_path: &Path,
    copy_path: &Path,
) -> Result<bool, CalibratedFolderError> {
    let mut source_reader = open_for_reading(source_path)?;
    let mut copy_reader = open_for_reading(copy_path)?;
    let mut source_bytes = vec![0_u8; COMPARISON_READ_BYTES];
    let mut copy_bytes = vec![0_u8; COMPARISON_READ_BYTES];
    loop {
        let source_read = read_full(&mut source_reader, &mut source_bytes, source_path)?;
        let copy_read = read_full(&mut copy_reader, &mut copy_bytes, copy_path)?;
        if source_read != copy_read || source_bytes[..source_read] != copy_bytes[..copy_read] {
            return Ok(false);
        }
        if source_read == 0 {
            return Ok(true);
        }
    }
}

fn open_for_reading(file_path: &Path) -> Result<fs::File, CalibratedFolderError> {
    fs::File::open(file_path).at_path(file_path)
}

/// Fills `read_bytes` as far as the file allows, so two readers that return
/// short reads at different points still compare the same byte ranges.
fn read_full(
    file_reader: &mut fs::File,
    read_bytes: &mut [u8],
    file_path: &Path,
) -> Result<usize, CalibratedFolderError> {
    let mut filled_count = 0;
    while filled_count < read_bytes.len() {
        let read_count = file_reader
            .read(&mut read_bytes[filled_count..])
            .at_path(file_path)?;
        if read_count == 0 {
            break;
        }
        filled_count += read_count;
    }
    Ok(filled_count)
}

/// Names the path an I/O failure happened at.
trait AtPath<T> {
    fn at_path(self, failed_path: &Path) -> Result<T, CalibratedFolderError>;
}

impl<T> AtPath<T> for io::Result<T> {
    fn at_path(self, failed_path: &Path) -> Result<T, CalibratedFolderError> {
        match self {
            Ok(io_value) => Ok(io_value),
            Err(source) => Err(CalibratedFolderError::Io {
                path: failed_path.to_path_buf(),
                source,
            }),
        }
    }
}

#[cfg(test)]
#[path = "calibrated_model_folder_tests.rs"]
mod tests;
