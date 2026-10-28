//! Standalone builder for Astronomical's pinned native runtime.
//!
//! Running the native CMake build inside Cargo's build-script phase makes the
//! native compile compete with Rust compilation for CPU — brutal on the small
//! hosted runners. This tool runs the exact same pinned build ahead of Cargo,
//! so a later Cargo build finds the native build store warm and skips CMake
//! entirely.
//!
//! # The sharing mechanism
//!
//! The build modules are included by `#[path]` from the runtime-integration
//! crate rather than through a dependency, so `cargo run` builds only this
//! tiny crate and never the runtime-integration dependency tree. The included
//! modules reference each other through `crate::` paths, so every module
//! that `native_compile` needs must be declared here with the same module
//! names the other includers use (see `runtime-integration/build.rs` for the
//! canonical module map).

#[path = "../../runtime-integration/build-orchestration/native_compile.rs"]
pub mod native_compile;
#[path = "../../runtime-integration/build-orchestration/native_parallelism.rs"]
pub mod native_parallelism;
#[path = "../../runtime-integration/build-orchestration/native_progress.rs"]
pub mod native_progress;
#[path = "../../runtime-integration/build-orchestration/native_store.rs"]
pub mod native_store;

use std::{
    error::Error,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use native_compile::{
    NativeRuntimeBuildInputs, build_pinned_native_runtime, detect_host_target_triple,
    native_build_store_directory, native_parallel_job_count, resolve_native_build_identity,
    write_native_build_status,
};
use native_progress::NativeBuildProgress;
use native_store::NativeBuildStore;

pub const NATIVE_BUILD_TOOL_PREFIX: &str = "[native-build-tool]";

/// Parsed command-line arguments for one standalone native build.
#[derive(Debug)]
pub struct NativeBuildToolArguments {
    native_build_profile: native_store::NativeBuildProfile,
    repository_root: PathBuf,
}

impl NativeBuildToolArguments {
    pub fn native_build_profile(&self) -> native_store::NativeBuildProfile {
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
    let native_build_profile = native_store::NativeBuildProfile::from_identity_name(&profile_name)
        .ok_or_else(|| {
            format!(
                "unsupported native build profile: {profile_name} (supported profiles: core, \
                     core+memory-contract)"
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
    // No Cargo context here, so the host triple is detected from rustc and
    // stated explicitly to the identity request — producing exactly the
    // identity the Cargo build script will later look up.
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

#[cfg(test)]
mod tests {
    //! Argument-parsing contracts for the standalone native-build tool.
    //!
    //! Pure CPU tests: they exercise the CLI surface that CI and the local
    //! pre-warm scripts rely on, without touching CMake or the native build
    //! store. They live beside `parse_arguments` because each one exercises
    //! exactly that one unit with literal in-memory arguments and crosses no
    //! process, filesystem, or network boundary of any kind.

    use crate::NativeBuildToolArguments;

    const FICTIONAL_REPOSITORY_ROOT: &str = "/tmp/astronomical-native-build-tool-fixture";

    fn parse_with_profile(profile_name: &str) -> Result<NativeBuildToolArguments, String> {
        crate::parse_arguments(&[
            "--profile".to_owned(),
            profile_name.to_owned(),
            "--repository-root".to_owned(),
            FICTIONAL_REPOSITORY_ROOT.to_owned(),
        ])
    }

    #[test]
    fn should_parse_core_profile_arguments() {
        let parsed_arguments =
            parse_with_profile("core").expect("core profile arguments should parse");

        assert_eq!(
            parsed_arguments.native_build_profile().identity_name(),
            "core"
        );
        assert_eq!(
            parsed_arguments.repository_root(),
            std::path::Path::new(FICTIONAL_REPOSITORY_ROOT)
        );
    }

    #[test]
    fn should_parse_every_supported_profile_identity_name() {
        for profile_identity_name in ["core", "core+memory-contract"] {
            let parsed_arguments =
                parse_with_profile(profile_identity_name).unwrap_or_else(|parse_error| {
                    panic!("profile {profile_identity_name} should parse: {parse_error}")
                });

            assert_eq!(
                parsed_arguments.native_build_profile().identity_name(),
                profile_identity_name
            );
        }
    }

    #[test]
    fn should_reject_unknown_profile_name() {
        let parse_error = parse_with_profile("core+unsupported-probe")
            .expect_err("an unknown profile name should be rejected");

        assert!(parse_error.contains("unsupported native build profile"));
        assert!(parse_error.contains("core+memory-contract"));
    }

    #[test]
    fn should_reject_missing_profile_argument() {
        let parse_error = crate::parse_arguments(&[
            "--repository-root".to_owned(),
            FICTIONAL_REPOSITORY_ROOT.to_owned(),
        ])
        .expect_err("a missing --profile argument should be rejected");

        assert!(parse_error.contains("missing required argument --profile"));
    }

    #[test]
    fn should_reject_missing_repository_root_argument() {
        let parse_error = crate::parse_arguments(&["--profile".to_owned(), "core".to_owned()])
            .expect_err("a missing --repository-root argument should be rejected");

        assert!(parse_error.contains("missing required argument --repository-root"));
    }

    #[test]
    fn should_reject_relative_repository_root() {
        let parse_error = crate::parse_arguments(&[
            "--profile".to_owned(),
            "core".to_owned(),
            "--repository-root".to_owned(),
            "relative/path".to_owned(),
        ])
        .expect_err("a relative repository root should be rejected");

        assert!(parse_error.contains("must be an absolute path"));
    }

    #[test]
    fn should_reject_unsupported_argument() {
        let parse_error = crate::parse_arguments(&[
            "--profile".to_owned(),
            "core".to_owned(),
            "--repository-root".to_owned(),
            FICTIONAL_REPOSITORY_ROOT.to_owned(),
            "--jobs".to_owned(),
            "4".to_owned(),
        ])
        .expect_err("an unsupported argument should be rejected");

        assert!(parse_error.contains("unsupported argument: --jobs"));
    }
}
