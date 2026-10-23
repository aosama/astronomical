//! `cargo xtask native-store-entry` — resolves the machine-shared native
//! build store entry for the current tree, building it first when no
//! complete entry exists, and prints its absolute path.
//!
//! Its caller of record is the mlx-c-rust link ladder: since the crate's
//! 0.32.4 line, that ladder emits its own link flags, and only the
//! `MLX_C_RUST_MLX_C_PREFIX` override steers it at the Astronomical patched
//! static pair instead of the crate's own stock shared libraries (see
//! `third-party/README.md`, "Bump-and-release process"). The command
//! resolves and builds through the same shared build-orchestration modules
//! the Cargo build script and the standalone native build tool use, so the
//! printed entry is exactly one the workspace build can link, and the
//! resolve-or-build shape means the exported prefix always names a
//! completed entry — a cold store costs a native build here instead of a
//! broken link line later.

use std::path::Path;

use crate::native_compile::{
    NativeRuntimeBuildInputs, build_pinned_native_runtime, detect_host_target_triple,
    native_build_store_directory, native_parallel_job_count, resolve_native_build_identity,
};
use crate::native_progress::NativeBuildProgress;
use crate::native_store::{NativeBuildProfile, NativeBuildStore};

pub fn run(repository_root: &Path, arguments: &[String]) -> Result<(), String> {
    let profile = parse_arguments(arguments)?;
    let native_source_directory = repository_root.join("crates/runtime-integration/native");
    let target_triple = detect_host_target_triple()
        .map_err(|error| format!("detecting the host target triple failed: {error}"))?;
    // The shared resolver keeps this command's identity bit-identical to the
    // one the workspace build script computes over the same tree; the host
    // triple is stated explicitly because this command runs outside Cargo.
    let native_build_identity =
        resolve_native_build_identity(repository_root, profile.identity_name(), &target_triple)
            .map_err(|error| format!("resolving the native build identity failed: {error}"))?;

    let native_build_store_directory = native_build_store_directory()
        .map_err(|error| format!("resolving the native build store directory failed: {error}"))?;
    let native_build_store = NativeBuildStore::new(
        &native_build_store_directory,
        &native_build_identity,
        profile,
    )
    .map_err(|error| format!("opening the native build store failed: {error}"))?;

    // resolve_or_build is the single reuse path: a warm entry is validated
    // and reused without compiling (the store's own integrity check heals a
    // corrupt cache), and a missing entry is built here rather than treated
    // as an error — the command's contract is to hand out a usable prefix,
    // and refusing would push every cold machine back onto the mlx-c-rust
    // ladder's stock shared pair. The prewarm flow stays the fast path; a
    // build under this command only runs when the store genuinely lacks
    // this tree's entry.
    let native_build_progress = NativeBuildProgress::from_environment()
        .map_err(|error| format!("preparing native build progress failed: {error}"))?;
    native_build_progress.record_build_start(&native_parallel_job_count());
    let native_build_started_at = std::time::Instant::now();
    let native_build_inputs = NativeRuntimeBuildInputs::from_environment();
    let native_build_artifacts = native_build_store
        .resolve_or_build(|native_build_directory| {
            build_pinned_native_runtime(
                &native_source_directory,
                native_build_directory,
                profile,
                &native_build_inputs,
                &native_build_progress,
            )
        })
        .map_err(|error| format!("building the native entry failed: {error}"))?;
    native_build_progress.record_build_completion(
        native_build_artifacts.was_built(),
        native_build_started_at.elapsed(),
    );
    println!("{}", native_build_store.entry_directory().display());
    Ok(())
}

fn parse_arguments(arguments: &[String]) -> Result<NativeBuildProfile, String> {
    let mut profile_identity_name: Option<&str> = None;
    let mut argument_index = 0;
    while argument_index < arguments.len() {
        let argument = arguments[argument_index].as_str();
        match argument {
            "--profile" => {
                let profile_value = arguments.get(argument_index + 1).ok_or_else(|| {
                    "--profile requires a value: expected core or core+memory-contract".to_owned()
                })?;
                profile_identity_name = Some(profile_value);
                argument_index += 2;
                continue;
            }
            "--help" | "-h" => {
                print_usage();
                std::process::exit(0);
            }
            other => {
                return Err(format!("unsupported native store entry option: {other}"));
            }
        }
    }
    let identity_name = profile_identity_name.unwrap_or("core");
    NativeBuildProfile::from_identity_name(identity_name)
        .ok_or_else(|| format!("unsupported native build profile: {identity_name}"))
}

fn print_usage() {
    println!(
        "cargo xtask native-store-entry [--profile core|core+memory-contract]\n\n\
         Prints the absolute path of the native build store entry for the current\n\
         tree, building the entry first when the store holds no completed one, so\n\
         the path always names a usable prefix. Export the printed path as\n\
         MLX_C_RUST_MLX_C_PREFIX so the mlx-c-rust link ladder resolves the\n\
         Astronomical patched static pair; scripts/ci/prewarm-native-build.sh\n\
         remains the preferred way to warm the store beforehand."
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn should_default_to_the_core_profile() {
        let profile = parse_arguments(&[]).expect("no arguments selects core");
        assert_eq!(profile.identity_name(), "core");
    }

    #[test]
    fn should_parse_each_supported_profile_name() {
        for identity_name in ["core", "core+memory-contract"] {
            let profile = parse_arguments(&["--profile".to_owned(), identity_name.to_owned()])
                .expect("a supported profile name parses");
            assert_eq!(profile.identity_name(), identity_name);
        }
    }

    #[test]
    fn should_reject_an_unknown_profile_name() {
        let error = parse_arguments(&["--profile".to_owned(), "experimental".to_owned()])
            .expect_err("an unknown profile name must fail");
        assert!(
            error.contains("experimental"),
            "the error names the unsupported profile: {error}"
        );
    }

    #[test]
    fn should_reject_a_profile_flag_without_a_value() {
        let error = parse_arguments(&["--profile".to_owned()])
            .expect_err("a profile flag without a value must fail");
        assert!(
            error.contains("requires a value"),
            "the error explains the missing value: {error}"
        );
    }

    #[test]
    fn should_reject_unknown_options() {
        let error =
            parse_arguments(&["--unexpected".to_owned()]).expect_err("an unknown option must fail");
        assert!(
            error.contains("--unexpected"),
            "the error names the unknown option: {error}"
        );
    }
}
