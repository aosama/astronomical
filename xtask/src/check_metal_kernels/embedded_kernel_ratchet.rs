//! No Metal kernel declared from Rust text: every kernel is declared through
//! `declare_metal_kernel` from its generated declaration, so an
//! `MlxMetalKernel::new` anywhere else is a kernel whose Metal is not in a
//! `.metal` file.
//!
//! Counting the constructor, not Metal words, is deliberate: a kernel body
//! never contains `[[kernel]]` (MLX writes the signature), while comments that
//! mention thread attributes are harmless; the constructor is the one thing
//! every embedded kernel needs.

use std::fs;
use std::path::Path;

use super::kernel_inventory::shown_path;

/// The folders whose Rust is searched.
const SEARCHED_ROOTS: [&str; 3] = ["apps", "crates", "verification"];

/// The constructor names that declare a kernel from text.
const EMBEDDED_KERNEL_CONSTRUCTORS: [&str; 3] = [
    "MlxMetalKernel::new(",
    "MlxMetalKernel::new_with_header(",
    "MlxMetalKernel::new_with_options(",
];

/// Paths allowed to call the constructors: the one declarer, and the
/// contracts of MLX's custom-kernel mechanism itself, which test MLX rather
/// than serve anything: tiny kernels (one deliberately invalid) and the
/// wrapper parity contract, which declares every registered kernel from its
/// file to compare the source MLX generates.
const ALLOWED_PATHS: [&str; 2] = [
    "crates/engine/model-primitives/src/metal_kernel_declaration.rs",
    "verification/acceptance/tests/gpu_contracts/runtime_mlx/",
];

/// Every line, as `path:line`, that declares a kernel from Rust text.
pub(crate) fn embedded_kernel_sites(repository_root: &Path) -> Vec<String> {
    let mut embedded_sites = Vec::new();
    for searched_root in SEARCHED_ROOTS {
        collect_embedded_sites(
            repository_root,
            &repository_root.join(searched_root),
            &mut embedded_sites,
        );
    }
    embedded_sites.sort();
    embedded_sites
}

fn collect_embedded_sites(
    repository_root: &Path,
    directory: &Path,
    embedded_sites: &mut Vec<String>,
) {
    let Ok(directory_entries) = fs::read_dir(directory) else {
        return;
    };
    for directory_entry in directory_entries.flatten() {
        let entry_path = directory_entry.path();
        let relative_path = shown_path(&entry_path, repository_root);
        if relative_path.ends_with("/target") || is_allowed_path(&relative_path) {
            continue;
        }
        if entry_path.is_dir() {
            collect_embedded_sites(repository_root, &entry_path, embedded_sites);
        } else if relative_path.ends_with(".rs") {
            record_file_sites(&entry_path, &relative_path, embedded_sites);
        }
    }
}

fn is_allowed_path(relative_path: &str) -> bool {
    for allowed_path in ALLOWED_PATHS {
        if relative_path.starts_with(allowed_path) {
            return true;
        }
    }
    false
}

fn record_file_sites(rust_file: &Path, relative_path: &str, embedded_sites: &mut Vec<String>) {
    let Ok(rust_text) = fs::read_to_string(rust_file) else {
        return;
    };
    for (line_index, rust_line) in rust_text.lines().enumerate() {
        if rust_line.trim_start().starts_with("//") {
            continue;
        }
        for constructor in EMBEDDED_KERNEL_CONSTRUCTORS {
            if rust_line.contains(constructor) {
                embedded_sites.push(format!("{relative_path}:{}", line_index + 1));
            }
        }
    }
}
