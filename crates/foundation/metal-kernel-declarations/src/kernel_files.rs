//! Finds a crate's Metal files and reads its registered kernels, with the
//! header and body files each one names resolved to paths.

use std::fs;
use std::path::{Path, PathBuf};

use crate::kernel_naming::{KERNEL_DIRECTORY_NAME, KernelFileKind, kernel_file_kind};
use crate::signature_block::{SignatureBlock, parse_signature_block};

/// One registered kernel read from its file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegisteredKernel {
    /// The file stem, which is the registered name.
    pub name: String,
    pub kernel_file: PathBuf,
    pub signature: SignatureBlock,
    /// The header files in the order MLX receives them.
    pub header_files: Vec<PathBuf>,
    /// The kernel's prologues, its own file, then its bodies, in the order
    /// MLX receives them.
    pub source_files: Vec<PathBuf>,
}

/// Every file in a folder named `metal`, or below one, anywhere under
/// `search_root`, sorted, so generated declarations and reports are stable.
pub fn kernel_folder_files(search_root: &Path) -> Vec<PathBuf> {
    let mut kernel_folder_files = Vec::new();
    collect_kernel_folder_files(search_root, false, &mut kernel_folder_files);
    kernel_folder_files.sort();
    kernel_folder_files
}

fn collect_kernel_folder_files(
    directory: &Path,
    is_kernel_folder: bool,
    found_files: &mut Vec<PathBuf>,
) {
    let Ok(directory_entries) = fs::read_dir(directory) else {
        return;
    };
    for directory_entry in directory_entries.flatten() {
        let entry_path = directory_entry.path();
        if entry_path.is_dir() {
            // A folder inside a Metal folder is part of it, so its files meet
            // the naming check instead of hiding from it.
            let is_nested_kernel_folder = is_kernel_folder
                || entry_path
                    .file_name()
                    .and_then(|folder_name| folder_name.to_str())
                    == Some(KERNEL_DIRECTORY_NAME);
            collect_kernel_folder_files(&entry_path, is_nested_kernel_folder, found_files);
        } else if is_kernel_folder {
            found_files.push(entry_path);
        }
    }
}

/// Every file directly in `directory`, sorted; the shared header folder is
/// flat.
pub fn files_in_directory(directory: &Path) -> Vec<PathBuf> {
    let mut directory_files = Vec::new();
    if let Ok(directory_entries) = fs::read_dir(directory) {
        for directory_entry in directory_entries.flatten() {
            if directory_entry.path().is_file() {
                directory_files.push(directory_entry.path());
            }
        }
    }
    directory_files.sort();
    directory_files
}

/// The registered kernels among `kernel_files`, each read and resolved. A
/// named header or body is looked for beside the kernel first, then in
/// `shared_header_directory`.
///
/// # Errors
/// Fails on a file that breaks the naming convention, a block that cannot be
/// read, or a named file that exists in neither place.
pub fn registered_kernels(
    kernel_files: &[PathBuf],
    name_prefix: &str,
    shared_header_directory: &Path,
) -> Result<Vec<RegisteredKernel>, String> {
    let mut kernels = Vec::new();
    for kernel_file in kernel_files {
        let file_name = file_name_of(kernel_file)?;
        if kernel_file_kind(file_name, name_prefix)? != KernelFileKind::RegisteredKernel {
            continue;
        }
        kernels.push(read_registered_kernel(
            kernel_file,
            shared_header_directory,
        )?);
    }
    Ok(kernels)
}

/// Reads one registered kernel file and resolves what it names.
///
/// # Errors
/// Fails when the file cannot be read, its block cannot be parsed, or a
/// named header or body cannot be found.
pub fn read_registered_kernel(
    kernel_file: &Path,
    shared_header_directory: &Path,
) -> Result<RegisteredKernel, String> {
    let kernel_text = fs::read_to_string(kernel_file)
        .map_err(|read_error| format!("{}: {read_error}", kernel_file.display()))?;
    let signature = parse_signature_block(&kernel_text)
        .map_err(|block_error| format!("{}: {block_error}", kernel_file.display()))?;
    let kernel_directory = kernel_file.parent().unwrap_or(Path::new("."));
    let mut header_files = Vec::new();
    for header_name in &signature.header_files {
        header_files.push(resolve_named_file(
            header_name,
            kernel_directory,
            shared_header_directory,
        )?);
    }
    let mut source_files = Vec::new();
    for prologue_name in &signature.prologue_files {
        source_files.push(resolve_named_file(
            prologue_name,
            kernel_directory,
            shared_header_directory,
        )?);
    }
    source_files.push(kernel_file.to_path_buf());
    for body_name in &signature.body_files {
        source_files.push(resolve_named_file(
            body_name,
            kernel_directory,
            shared_header_directory,
        )?);
    }
    let name = file_name_of(kernel_file)?
        .trim_end_matches(".metal")
        .to_owned();
    Ok(RegisteredKernel {
        name,
        kernel_file: kernel_file.to_path_buf(),
        signature,
        header_files,
        source_files,
    })
}

fn resolve_named_file(
    file_name: &str,
    kernel_directory: &Path,
    shared_header_directory: &Path,
) -> Result<PathBuf, String> {
    for candidate_directory in [kernel_directory, shared_header_directory] {
        let candidate_file = candidate_directory.join(file_name);
        if candidate_file.is_file() {
            return Ok(candidate_file);
        }
    }
    Err(format!(
        "{file_name} is neither beside the kernel in {} nor in {}",
        kernel_directory.display(),
        shared_header_directory.display()
    ))
}

fn file_name_of(file_path: &Path) -> Result<&str, String> {
    file_path
        .file_name()
        .and_then(|file_name| file_name.to_str())
        .ok_or_else(|| format!("{} has no readable file name", file_path.display()))
}

/// The repository root above `start_directory`: the first folder whose
/// `Cargo.toml` declares the workspace.
pub fn repository_root_above(start_directory: &Path) -> Option<PathBuf> {
    let mut candidate_directory = Some(start_directory);
    while let Some(candidate) = candidate_directory {
        if let Ok(manifest_text) = fs::read_to_string(candidate.join("Cargo.toml"))
            && manifest_text
                .lines()
                .any(|manifest_line| manifest_line.trim() == "[workspace]")
        {
            return Some(candidate.to_path_buf());
        }
        candidate_directory = candidate.parent();
    }
    None
}
