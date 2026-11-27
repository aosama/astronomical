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

use super::rust_file_walk::rust_files_under;

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
    for (relative_path, rust_file) in
        rust_files_under(repository_root, &SEARCHED_ROOTS, &ALLOWED_PATHS)
    {
        record_file_sites(&rust_file, &relative_path, &mut embedded_sites);
    }
    embedded_sites.sort();
    embedded_sites
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
