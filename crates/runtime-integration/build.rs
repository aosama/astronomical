//! Cargo orchestration for Astronomical's pinned MLX and MLX-C runtime.
//!
//! Native products are compatibility-keyed outside `OUT_DIR` so release-version
//! changes do not rebuild MLX. Bindings remain in `OUT_DIR`, which preserves
//! Cargo's ownership of generated Rust source. The native compilation itself
//! lives in `build_native_compile.rs`, shared with the standalone native-build
//! tool so pre-warmed builds and in-Cargo builds stay identical.

use std::{env, error::Error, path::Path, time::Instant};

mod build_legacy_native_output;
mod build_native_compile;
mod build_native_linking;
mod build_native_store;
mod build_parallelism;
mod build_progress;

use build_native_compile::{
    NATIVE_ARCHIVE_VARIABLES, NATIVE_BUILD_STATUS_FILE_VARIABLE, NATIVE_BUILD_STORE_VARIABLE,
    NATIVE_DEPENDENCY_CACHE_VARIABLE, NativeRuntimeBuildInputs, RUSTC_WRAPPER_VARIABLE,
    build_pinned_native_runtime, native_build_store_directory, native_parallel_job_count,
    required_path_variable, resolve_native_build_identity, write_native_build_status,
};
use build_native_store::{NativeBuildProfile, NativeBuildStore};
use build_progress::{NATIVE_BUILD_PROGRESS_FILE_VARIABLE, NativeBuildProgress};

const MLX_FEATURE_VARIABLE: &str = "CARGO_FEATURE_MLX";
const MLX_MEMORY_CONTRACT_PROBE_FEATURE_VARIABLE: &str = "CARGO_FEATURE_MLX_MEMORY_CONTRACT_PROBE";

fn main() -> Result<(), Box<dyn Error>> {
    emit_environment_rerun_contracts();
    if env::var_os(MLX_FEATURE_VARIABLE).is_none() {
        return Ok(());
    }

    let manifest_directory = required_path_variable("CARGO_MANIFEST_DIR")?;
    let output_directory = required_path_variable("OUT_DIR")?;
    let native_build_progress = NativeBuildProgress::from_environment()
        .map_err(|message| -> Box<dyn Error> { message.into() })?;
    native_build_progress.record_build_start(&native_parallel_job_count());
    let native_build_started_at = Instant::now();
    let repository_root = manifest_directory.join("../..").canonicalize()?;
    let native_source_directory = manifest_directory.join("native");
    let native_build_profile = selected_native_build_profile();
    let native_build_inputs = NativeRuntimeBuildInputs::from_environment();
    let target_triple = env::var("TARGET")
        .map_err(|_| "required environment variable TARGET is missing".to_owned())?;
    let native_build_identity = resolve_native_build_identity(
        &repository_root,
        native_build_profile.identity_name(),
        &target_triple,
    )?;
    let native_build_store_directory = native_build_store_directory()?;
    let native_build_store = NativeBuildStore::new(
        &native_build_store_directory,
        &native_build_identity,
        native_build_profile,
    )?;
    let native_build_artifacts = native_build_store.resolve_or_build(|native_build_directory| {
        build_pinned_native_runtime(
            &native_source_directory,
            native_build_directory,
            native_build_profile,
            &native_build_inputs,
            &native_build_progress,
        )
    })?;
    build_legacy_native_output::remove_legacy_cargo_native_build_directory(&output_directory)?;
    write_native_build_status(&native_build_artifacts)?;
    native_build_progress.record_build_completion(
        native_build_artifacts.was_built(),
        native_build_started_at.elapsed(),
    );

    build_native_linking::configure_rust_linking(&native_build_artifacts)?;
    emit_native_source_rerun_contracts(
        &manifest_directory,
        &native_source_directory,
        &native_build_inputs,
    );
    Ok(())
}

fn emit_environment_rerun_contracts() {
    for environment_variable in [
        MLX_FEATURE_VARIABLE,
        MLX_MEMORY_CONTRACT_PROBE_FEATURE_VARIABLE,
        RUSTC_WRAPPER_VARIABLE,
        NATIVE_DEPENDENCY_CACHE_VARIABLE,
        NATIVE_BUILD_STORE_VARIABLE,
        NATIVE_BUILD_STATUS_FILE_VARIABLE,
        NATIVE_BUILD_PROGRESS_FILE_VARIABLE,
    ] {
        println!("cargo:rerun-if-env-changed={environment_variable}");
    }
    for archive_variable in NATIVE_ARCHIVE_VARIABLES {
        println!("cargo:rerun-if-env-changed={archive_variable}");
    }
}

fn selected_native_build_profile() -> NativeBuildProfile {
    let should_build_memory_contract_probe =
        env::var_os(MLX_MEMORY_CONTRACT_PROBE_FEATURE_VARIABLE).is_some();
    if should_build_memory_contract_probe {
        NativeBuildProfile::new(true)
    } else {
        NativeBuildProfile::core()
    }
}

fn emit_native_source_rerun_contracts(
    manifest_directory: &Path,
    native_source_directory: &Path,
    native_build_inputs: &NativeRuntimeBuildInputs,
) {
    for source_path in [
        manifest_directory.join("build.rs"),
        manifest_directory.join("build_bindings.rs"),
        manifest_directory.join("build_native_compile.rs"),
        manifest_directory.join("build_native_linking.rs"),
        manifest_directory.join("build_native_store.rs"),
        manifest_directory.join("build_native_store_manifest.rs"),
        manifest_directory.join("native-build-store-schema-version"),
        native_source_directory.join("CMakeLists.txt"),
        native_source_directory.join("apply_patch_if_needed.cmake"),
        native_source_directory.join("tests/mlx_memory_contract_probe.cpp"),
        manifest_directory.join("../../scripts/ci/native-build-cache-fingerprint.sh"),
        manifest_directory.join("../../third-party/native-dependency-manifest.cmake"),
        manifest_directory.join("../../third-party/pins"),
        manifest_directory.join("../../third-party/patches"),
    ] {
        println!("cargo:rerun-if-changed={}", source_path.display());
    }
    // The build script previously tracked these paths only on cold builds;
    // emitting them on every run keeps Cargo's rebuild triggers honest when
    // the store reuses a warm entry.
    if let Some(native_dependency_cache_directory) =
        native_build_inputs.native_dependency_cache_directory()
    {
        println!(
            "cargo:rerun-if-changed={}",
            native_dependency_cache_directory.display()
        );
    }
    for (_, archive_path) in native_build_inputs.native_archive_paths() {
        println!("cargo:rerun-if-changed={}", archive_path.display());
    }
}
