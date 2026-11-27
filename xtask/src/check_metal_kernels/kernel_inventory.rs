//! Every Metal file in the repository, checked against the naming convention
//! and read into the registered kernels the offline compile builds.

use std::path::{Path, PathBuf};

use metal_kernel_declarations::kernel_files::{
    RegisteredKernel, files_in_directory, kernel_folder_files, read_registered_kernel,
};
use metal_kernel_declarations::kernel_naming::{
    KernelFileKind, SHARED_HEADER_DIRECTORY, crate_kernel_scope, kernel_file_kind,
};

/// The folders whose crates may hold Metal folders.
const KERNEL_SEARCH_ROOTS: [&str; 1] = ["crates"];

/// What the inventory found: the kernels to compile and every rule broken.
#[derive(Debug, Default)]
pub(crate) struct KernelInventory {
    pub(crate) registered_kernels: Vec<RegisteredKernel>,
    /// Helper headers and shared bodies, to check each is used.
    pub(crate) shared_files: Vec<PathBuf>,
    pub(crate) violations: Vec<String>,
}

/// Reads every Metal file under the repository's crates.
pub(crate) fn kernel_inventory(repository_root: &Path) -> KernelInventory {
    let shared_header_directory = repository_root.join(SHARED_HEADER_DIRECTORY);
    let mut inventory = KernelInventory::default();
    for shared_header in files_in_directory(&shared_header_directory) {
        record_kernel_file(
            &mut inventory,
            repository_root,
            &shared_header,
            &shared_header_directory,
        );
    }
    for search_root in KERNEL_SEARCH_ROOTS {
        for kernel_file in kernel_folder_files(&repository_root.join(search_root)) {
            record_kernel_file(
                &mut inventory,
                repository_root,
                &kernel_file,
                &shared_header_directory,
            );
        }
    }
    check_shared_files_are_used(&mut inventory, repository_root);
    inventory
}

fn record_kernel_file(
    inventory: &mut KernelInventory,
    repository_root: &Path,
    kernel_file: &Path,
    shared_header_directory: &Path,
) {
    let shown_path = shown_path(kernel_file, repository_root);
    let Some(name_prefix) = owning_crate_prefix(kernel_file, repository_root) else {
        inventory
            .violations
            .push(format!("{shown_path}: Metal files live only in model-primitives, a family crate or a format crate"));
        return;
    };
    let file_name = kernel_file
        .file_name()
        .and_then(|file_name| file_name.to_str())
        .unwrap_or_default();
    match kernel_file_kind(file_name, &name_prefix) {
        Err(naming_violation) => inventory
            .violations
            .push(format!("{shown_path}: {naming_violation}")),
        Ok(KernelFileKind::RegisteredKernel) => {
            match read_registered_kernel(kernel_file, shared_header_directory) {
                Ok(registered_kernel) => {
                    record_registered_kernel(inventory, &shown_path, registered_kernel)
                }
                Err(read_violation) => inventory.violations.push(read_violation),
            }
        }
        Ok(KernelFileKind::SharedBody | KernelFileKind::Helper) => {
            inventory.shared_files.push(kernel_file.to_path_buf())
        }
    }
}

fn record_registered_kernel(
    inventory: &mut KernelInventory,
    shown_path: &str,
    registered_kernel: RegisteredKernel,
) {
    if registered_kernel.signature.contract.is_none() {
        inventory.violations.push(format!(
            "{shown_path}: the signature block names no @contract (its GPU exactness contract)"
        ));
    }
    if registered_kernel.signature.probe.is_none() {
        inventory.violations.push(format!(
            "{shown_path}: the signature block names no @probe (a capability probe, or `none` and why)"
        ));
    }
    inventory.registered_kernels.push(registered_kernel);
}

/// The name prefix of the crate holding `kernel_file`: the nearest folder
/// above it with a `Cargo.toml`, judged by its place in the repository.
fn owning_crate_prefix(kernel_file: &Path, repository_root: &Path) -> Option<String> {
    let mut candidate_directory = kernel_file.parent();
    while let Some(candidate) = candidate_directory {
        if candidate == repository_root {
            return None;
        }
        if candidate.join("Cargo.toml").is_file() {
            let crate_directory = shown_path(candidate, repository_root);
            return crate_kernel_scope(&crate_directory)
                .map(|crate_scope| crate_scope.name_prefix());
        }
        candidate_directory = candidate.parent();
    }
    None
}

/// A helper or shared body no registered kernel names is dead code that the
/// offline compile would never see.
fn check_shared_files_are_used(inventory: &mut KernelInventory, repository_root: &Path) {
    let mut unused_files = Vec::new();
    for shared_file in &inventory.shared_files {
        let mut is_used = false;
        for registered_kernel in &inventory.registered_kernels {
            if registered_kernel.header_files.contains(shared_file)
                || registered_kernel.source_files.contains(shared_file)
            {
                is_used = true;
            }
        }
        if !is_used {
            unused_files.push(format!(
                "{}: no registered kernel names it in @header or @body",
                shown_path(shared_file, repository_root)
            ));
        }
    }
    inventory.violations.extend(unused_files);
}

/// `file_path` relative to the repository, with `/` separators.
pub(crate) fn shown_path(file_path: &Path, repository_root: &Path) -> String {
    let relative_path = file_path.strip_prefix(repository_root).unwrap_or(file_path);
    let mut path_words = Vec::new();
    for path_component in relative_path.components() {
        path_words.push(path_component.as_os_str().to_string_lossy().into_owned());
    }
    path_words.join("/")
}
