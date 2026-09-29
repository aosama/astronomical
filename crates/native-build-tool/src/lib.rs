//! Standalone builder for Astronomical's pinned native runtime.
//!
//! Running the native CMake build inside Cargo's build-script phase makes the
//! native compile compete with Rust compilation for CPU — brutal on the small
//! hosted runners. This tool runs the exact same pinned build ahead of Cargo,
//! so a later Cargo build finds the native build store warm and skips CMake
//! entirely.
//!
//! The build modules are included by `#[path]` from the runtime-integration
//! crate rather than through a dependency, so `cargo run` builds only this
//! tiny crate and never the runtime-integration dependency tree. The included
//! modules reference each other through `crate::` paths, so every module that
//! `build_native_compile` needs must be declared here.

#[path = "../../runtime-integration/build_native_compile.rs"]
pub mod build_native_compile;
#[path = "../../runtime-integration/build_native_store.rs"]
pub mod build_native_store;
#[path = "../../runtime-integration/build_parallelism.rs"]
pub mod build_parallelism;
#[path = "../../runtime-integration/build_progress.rs"]
pub mod build_progress;

use std::{
    error::Error,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use build_native_compile::{
    NativeRuntimeBuildInputs, build_pinned_native_runtime, detect_host_target_triple,
    native_build_store_directory, native_parallel_job_count, resolve_native_build_identity,
    write_native_build_status,
};
use build_native_store::NativeBuildStore;
use build_progress::NativeBuildProgress;

pub const NATIVE_BUILD_TOOL_PREFIX: &str = "[native-build-tool]";

#[derive(Debug)]
pub struct NativeBuildToolArguments {
    native_build_profile: build_native_store::NativeBuildProfile,
    repository_root: PathBuf,
}

impl NativeBuildToolArguments {
    pub fn native_build_profile(&self) -> build_native_store::NativeBuildProfile {
        self.native_build_profile
    }

    pub fn repository_root(&self) -> &Path {
        &self.repository_root
    }
}

#[derive(Debug)]
pub struct NativeBuildOutcome {
    was_built: bool,
    elapsed: Duration,
}

impl NativeBuildOutcome {
    pub fn was_built(&self) -> bool {
        self.was_built
    }

    pub fn elapsed(&self) -> Duration {
        self.elapsed
    }
}

pub fn parse_arguments(arguments: &[String]) -> Result<NativeBuildToolArguments, String> {
    let mut profile_name: Option<String> = None;
    let mut repository_root: Option<String> = None;
    let mut argument_index = 0;
    while argument_index < arguments.len() {
        match arguments[argument_index].as_str() {
            "--profile" => {
                let profile_value = arguments.get(argument_index + 1).ok_or_else(|| {
                    "the --profile argument requires a native build profile name".to_owned()
                })?;
                profile_name = Some(profile_value.clone());
                argument_index += 2;
            }
            "--repository-root" => {
                let repository_root_value = arguments.get(argument_index + 1).ok_or_else(|| {
                    "the --repository-root argument requires a repository directory path".to_owned()
                })?;
                repository_root = Some(repository_root_value.clone());
                argument_index += 2;
            }
            unsupported_argument => {
                return Err(format!("unsupported argument: {unsupported_argument}"));
            }
        }
    }
    let profile_name = profile_name.ok_or("missing required argument --profile <profile-name>")?;
    let repository_root =
        repository_root.ok_or("missing required argument --repository-root <path>")?;
    let repository_root_path = PathBuf::from(repository_root);
    if !repository_root_path.is_absolute() {
        return Err(format!(
            "--repository-root must be an absolute path: {}",
            repository_root_path.display()
        ));
    }
    let native_build_profile = build_native_store::NativeBuildProfile::from_identity_name(
        &profile_name,
    )
    .ok_or_else(|| {
        format!(
            "unsupported native build profile: {profile_name} (supported profiles: core, \
                     core+memory-contract, core+experimental-aligned-expert-packs, \
                     core+memory-contract+experimental-aligned-expert-packs)"
        )
    })?;
    Ok(NativeBuildToolArguments {
        native_build_profile,
        repository_root: repository_root_path,
    })
}

pub fn run_native_build(
    arguments: &NativeBuildToolArguments,
) -> Result<NativeBuildOutcome, Box<dyn Error>> {
    let native_build_progress = NativeBuildProgress::from_environment()
        .map_err(|message| -> Box<dyn Error> { message.into() })?;
    native_build_progress.record_build_start(&native_parallel_job_count());
    let native_build_started_at = Instant::now();
    let native_source_directory = arguments
        .repository_root()
        .join("crates/runtime-integration/native");
    let target_triple = detect_host_target_triple()?;
    let native_build_identity = resolve_native_build_identity(
        arguments.repository_root(),
        arguments.native_build_profile().identity_name(),
        &target_triple,
    )?;
    let native_build_store_directory = native_build_store_directory()?;
    let native_build_store = NativeBuildStore::new(
        &native_build_store_directory,
        &native_build_identity,
        arguments.native_build_profile(),
    )?;
    let native_build_inputs = NativeRuntimeBuildInputs::from_environment();
    let native_build_artifacts = native_build_store.resolve_or_build(|native_build_directory| {
        build_pinned_native_runtime(
            &native_source_directory,
            native_build_directory,
            arguments.native_build_profile(),
            &native_build_inputs,
            &native_build_progress,
        )
    })?;
    write_native_build_status(&native_build_artifacts)?;
    native_build_progress.record_build_completion(
        native_build_artifacts.was_built(),
        native_build_started_at.elapsed(),
    );
    Ok(NativeBuildOutcome {
        was_built: native_build_artifacts.was_built(),
        elapsed: native_build_started_at.elapsed(),
    })
}
