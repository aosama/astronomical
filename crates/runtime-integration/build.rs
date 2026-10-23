//! Cargo orchestration for Astronomical's pinned MLX and MLX-C runtime.
//!
//! # What this script does (and when it does nothing)
//!
//! Without the `mlx` feature this script exits immediately — the crate then
//! builds as pure Rust (types and Metallib path rules only) on any machine.
//! With the feature enabled, the script:
//!
//! 1. resolves the **native build identity** through
//!    `astronomical-native-identity` (sources + toolchain fingerprint);
//! 2. asks the **native build store** (`build-orchestration/native_store.rs`)
//!    to reuse a published entry for that identity or build one under an
//!    exclusive lock;
//! 3. configures **Rust linking** against the published static archives and
//!    publishes the Metallib beside the cargo target (`build-orchestration/
//!    native_linking.rs`);
//! 4. emits **rerun contracts** so Cargo re-runs exactly when an input the
//!    artifact depends on changes — and never for a phantom path (see the
//!    memorialized lesson below).
//!
//! # The module map (and the multi-includer rule)
//!
//! The modules under `build-orchestration/` are shared verbatim with the
//! standalone `astronomical-native-build-tool` and the offline test target
//! through `#[path]` includes, so a pre-warmed out-of-Cargo build and the
//! in-Cargo build execute byte-identical code. Because a `#[path]`-included
//! file must not declare sibling modules itself, **every includer declares
//! the modules it needs with identical names**, and the shared modules reach
//! each other only through `crate::` paths:
//!
//! - `native_compile` — the CMake configure/build invocation (also included
//!   by the standalone tool);
//! - `native_linking` — linker directives and Metallib publication
//!   (Cargo-only; it emits `cargo:` directives);
//! - `native_store` — the identity-keyed, lock-guarded, atomically published
//!   machine-wide store (with `native_store_manifest` as its integrity
//!   manifest submodule);
//! - `native_parallelism` — the native job-count resolution;
//! - `native_progress` — the hosted-CI progress heartbeat file.
//!
//! # Where generated bindings live
//!
//! Native products are identity-keyed outside `OUT_DIR` so release-version
//! changes do not rebuild MLX. The generated Rust bindings live in the
//! `OUT_DIR` of the published `mlx-c-rust` crate's build script (resolved
//! from the cargo registry), which preserves Cargo's ownership of generated
//! source; this script never writes there. The native compilation itself
//! lives in `build-orchestration/native_compile.rs`, shared with the
//! standalone native-build tool so pre-warmed builds and in-Cargo builds
//! stay identical.
//!
//! # A memorialized lesson: never watch a path that may not exist
//!
//! This script once carried a `rerun-if-changed` for `build_bindings.rs`
//! after that file was deleted. Cargo reports a missing watched path at
//! **every** freshness check, which re-ran this script — and rebuilt
//! runtime-integration and its dependents — in every later cargo invocation
//! of a CI job: a measured 25 seconds per step. The offline wiring contract
//! (`tests/offline/native_build_wiring.rs`) now fails if a watched path does
//! not exist, so the class of failure cannot return silently.

use std::{env, error::Error, path::Path, time::Instant};

#[path = "build-orchestration/native_compile.rs"]
mod native_compile;
#[path = "build-orchestration/native_linking.rs"]
mod native_linking;
#[path = "build-orchestration/native_parallelism.rs"]
mod native_parallelism;
#[path = "build-orchestration/native_progress.rs"]
mod native_progress;
#[path = "build-orchestration/native_store.rs"]
mod native_store;

use native_compile::{
    NATIVE_ARCHIVE_VARIABLES, NATIVE_BUILD_STATUS_FILE_VARIABLE, NATIVE_BUILD_STORE_VARIABLE,
    NATIVE_DEPENDENCY_CACHE_VARIABLE, NativeRuntimeBuildInputs, RUSTC_WRAPPER_VARIABLE,
    build_pinned_native_runtime, native_build_store_directory, native_parallel_job_count,
    required_path_variable, resolve_native_build_identity, write_native_build_status,
};
use native_progress::{NATIVE_BUILD_PROGRESS_FILE_VARIABLE, NativeBuildProgress};
use native_store::{NativeBuildProfile, NativeBuildStore};

const MLX_FEATURE_VARIABLE: &str = "CARGO_FEATURE_MLX";
const MLX_MEMORY_CONTRACT_PROBE_FEATURE_VARIABLE: &str = "CARGO_FEATURE_MLX_MEMORY_CONTRACT_PROBE";

fn main() -> Result<(), Box<dyn Error>> {
    emit_environment_rerun_contracts();
    if env::var_os(MLX_FEATURE_VARIABLE).is_none() {
        return Ok(());
    }

    let manifest_directory = required_path_variable("CARGO_MANIFEST_DIR")?;
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
    write_native_build_status(&native_build_artifacts)?;
    native_build_progress.record_build_completion(
        native_build_artifacts.was_built(),
        native_build_started_at.elapsed(),
    );

    native_linking::configure_rust_linking(&native_build_artifacts)?;
    emit_native_source_rerun_contracts(
        &manifest_directory,
        &native_source_directory,
        &native_build_inputs,
    );
    Ok(())
}

/// Declares every environment input whose value change must re-run the
/// build script (and therefore potentially re-resolve the store).
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

/// Maps the requested Cargo features onto a native build profile.
///
/// The memory-contract probe profile exists because the probe binary is an
/// extra CMake target; building it unconditionally would tax every ordinary
/// build for a product used only by the acceptance memory lanes.
fn selected_native_build_profile() -> NativeBuildProfile {
    let should_build_memory_contract_probe =
        env::var_os(MLX_MEMORY_CONTRACT_PROBE_FEATURE_VARIABLE).is_some();
    if should_build_memory_contract_probe {
        NativeBuildProfile::new(true)
    } else {
        NativeBuildProfile::core()
    }
}

/// Declares the file watches for every native build input Cargo cannot infer.
///
/// Two rules govern this list:
///
/// - **Every watched path must exist** (see the phantom-path lesson in the
///   module documentation); the offline wiring contract asserts it.
/// - The identity policy's own sources are watched so an edit to the
///   pathspec policy or manifest assembly re-runs the build script and the
///   store lookup uses the new identity.
fn emit_native_source_rerun_contracts(
    manifest_directory: &Path,
    native_source_directory: &Path,
    native_build_inputs: &NativeRuntimeBuildInputs,
) {
    for source_path in [
        manifest_directory.join("build.rs"),
        manifest_directory.join("build-orchestration/native_compile.rs"),
        manifest_directory.join("build-orchestration/native_linking.rs"),
        manifest_directory.join("build-orchestration/native_store.rs"),
        manifest_directory.join("build-orchestration/native_store_manifest.rs"),
        manifest_directory.join("build-orchestration/native_parallelism.rs"),
        manifest_directory.join("build-orchestration/native_progress.rs"),
        manifest_directory.join("build-orchestration/native-build-store-schema-version"),
        native_source_directory.join("CMakeLists.txt"),
        native_source_directory.join("apply_patch_if_needed.cmake"),
        native_source_directory.join("tests/mlx_memory_contract_probe.cpp"),
        manifest_directory.join("../native-identity/src"),
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
