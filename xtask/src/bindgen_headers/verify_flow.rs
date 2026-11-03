//! `--verify-headers`: proves the extracted MLX-C headers are byte-identical
//! to the `include/mlx` tree a completed native build staged. This is the
//! fast surface-diff step of a dependency bump (third-party/README.md).

use std::path::Path;
use std::process::Command;

use super::hash_manifest::{manifest_differences, tree_hash_manifest};
use super::header_locations::{
    COMPLETION_MARKER_FILE_NAME, EXTRACTION_TREE_MLX_C, built_entry_directory, built_identity,
    extraction_directory, native_build_store_schema_version, source_identity,
};
use super::options::ProvisionOptions;
use super::step_log::{log_line, start_step};

/// How many differing paths the mismatch line names, as the shell did.
const REPORTED_DIFFERENCE_LIMIT: usize = 10;

pub(crate) fn verify_headers_against_native_build(
    repository_root: &Path,
    provision_options: &ProvisionOptions,
    cache_directory: &Path,
    store_directory: &Path,
) -> Result<(), String> {
    let identity_step = start_step("resolve-built-identity");
    let target_triple = verification_target_triple(provision_options)?;
    let built_identity = built_identity(
        repository_root,
        provision_options.native_build_profile,
        &target_triple,
    )?;
    let schema_version = native_build_store_schema_version(repository_root)?;
    let built_entry = built_entry_directory(store_directory, schema_version, &built_identity);
    if !built_entry.join("complete").is_file() {
        return Err(format!(
            "no completed native build for identity {built_identity}: {}\nrun cargo xtask prewarm-native-build --profile {} (or pass --profile for a profile that is already built)",
            built_entry.display(),
            provision_options.native_build_profile.as_profile_text()
        ));
    }
    identity_step.finish("success");
    let source_identity = source_identity(repository_root)?;
    let extraction = extraction_directory(cache_directory, &source_identity);
    if !extraction.join(COMPLETION_MARKER_FILE_NAME).is_file() {
        return Err(format!(
            "no provisioned bindgen headers for identity {source_identity}; run cargo xtask provision-bindgen-headers first"
        ));
    }
    compare_header_trees(
        &extraction.join(EXTRACTION_TREE_MLX_C).join("mlx"),
        &built_entry.join("include/mlx"),
    )
}

fn compare_header_trees(extracted_headers: &Path, built_headers: &Path) -> Result<(), String> {
    let compare_step = start_step("compare-headers");
    let extracted_manifest = tree_hash_manifest(extracted_headers)?;
    let built_manifest = tree_hash_manifest(built_headers)?;
    let differing_paths = manifest_differences(&extracted_manifest, &built_manifest);
    let headers_match = differing_paths.is_empty();
    if headers_match {
        log_line(&format!(
            "headers_match=yes files={}",
            extracted_manifest.lines().count()
        ));
    } else {
        log_line("headers_match=no");
    }
    log_line(&format!(
        "extracted_headers={}",
        extracted_headers.display()
    ));
    log_line(&format!("built_headers={}", built_headers.display()));
    if headers_match {
        compare_step.finish("success");
        return Ok(());
    }
    let reported_paths: Vec<String> = differing_paths
        .into_iter()
        .take(REPORTED_DIFFERENCE_LIMIT)
        .collect();
    log_line(&format!("differing_paths={}", reported_paths.join(" ")));
    compare_step.finish("mismatch");
    Err("extracted MLX-C headers differ from the completed native build".to_owned())
}

/// An explicit `--target` wins, then Cargo's `TARGET`, then the rustc host,
/// matching how the native build itself resolves its target.
fn verification_target_triple(provision_options: &ProvisionOptions) -> Result<String, String> {
    if let Some(target_triple) = &provision_options.target_triple {
        return Ok(target_triple.clone());
    }
    if let Ok(cargo_target) = std::env::var("TARGET")
        && !cargo_target.is_empty()
    {
        return Ok(cargo_target);
    }
    match Command::new("rustc")
        .args(["--print", "host-tuple"])
        .output()
    {
        Ok(rustc_output) if rustc_output.status.success() => {
            Ok(String::from_utf8_lossy(&rustc_output.stdout)
                .trim()
                .to_owned())
        }
        _ => Err("could not resolve the rustc host target".to_owned()),
    }
}
