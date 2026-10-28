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
//! 2. asks the **native build store** (the
//!    `astronomical-native-build-orchestration` crate's `native_store`
//!    module) to reuse a published entry for that identity or build one
//!    under an exclusive lock;
//! 3. configures **Rust linking** against the published static archives and
//!    publishes the Metallib beside the cargo target (the same crate's
//!    `native_linking` module);
//! 4. emits **rerun contracts** so Cargo re-runs exactly when an input the
//!    artifact depends on changes — and never for a phantom path (see the
//!    memorialized lesson below).
//!
//! # The orchestration dependency
//!
//! The CMake invocation, store, linking, parallelism, and progress modules
//! live in the `astronomical-native-build-orchestration` crate — their
//! single compilation home (issue #1628) — consumed here through
//! `[build-dependencies]`. The standalone `astronomical-native-build-tool`
//! pre-warms the same store through the same crate as a normal dependency,
//! so a pre-warmed out-of-Cargo build and the in-Cargo build execute
//! byte-identical code because they compile one crate, not because files
//! are shared across crate roots.
//!
//! # Where generated bindings live
//!
//! Native products are identity-keyed outside `OUT_DIR` so release-version
//! changes do not rebuild MLX. The generated Rust bindings live in the
//! `OUT_DIR` of the published `mlx-c-rust` crate's build script (resolved
//! from the cargo registry), which preserves Cargo's ownership of generated
//! source; this script never writes there. The native compilation itself
//! lives in `astronomical-native-build-orchestration`'s `native_compile`
//! module, shared with the standalone native-build tool so pre-warmed
//! builds and in-Cargo builds stay identical.
//!
//! # A memorialized lesson: never watch a path that may not exist
//!
//! This script once carried a `rerun-if-changed` for `build_bindings.rs`
//! after that file was deleted. Cargo reports a missing watched path at
//! **every** freshness check, which re-ran this script — and rebuilt
//! runtime-integration and its dependents — in every later cargo invocation
//! of a CI job: a measured 25 seconds per step. The integration wiring contract
//! (`tests/integration/native_build_wiring.rs`) now fails if a watched path does
//! not exist, so the class of failure cannot return silently.

use std::{env, error::Error, path::Path, time::Instant};

use astronomical_native_build_orchestration::{
    NATIVE_BUILD_PROGRESS_FILE_VARIABLE, NATIVE_BUILD_STATUS_FILE_VARIABLE,
    NATIVE_BUILD_STORE_VARIABLE, NATIVE_DEPENDENCY_CACHE_VARIABLE, NativeBuildProfile,
    NativeBuildProgress, NativeBuildStore, NativeRuntimeBuildInputs, RUSTC_WRAPPER_VARIABLE,
    build_pinned_native_runtime, native_build_store_directory, native_parallel_job_count,
    required_path_variable, resolve_native_build_identity, write_native_build_status,
};

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

    astronomical_native_build_orchestration::native_linking::configure_rust_linking(
        &native_build_artifacts,
    )?;
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
///   module documentation); the integration wiring contract asserts it.
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
        // One directory watch covers every orchestration module and the
        // schema-version policy file beside them; the modules are
        // identity-exempt executors, so watching their whole source tree is
        // correct-and-conservative (a store hit after re-running this script
        // is cheap).
        manifest_directory.join("../native-build-orchestration/src"),
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
    // The build script previously tracked this path only on cold builds;
    // emitting it on every run keeps Cargo's rebuild triggers honest when
    // the store reuses a warm entry.
    if let Some(native_dependency_cache_directory) =
        native_build_inputs.native_dependency_cache_directory()
    {
        println!(
            "cargo:rerun-if-changed={}",
            native_dependency_cache_directory.display()
        );
    }
}
