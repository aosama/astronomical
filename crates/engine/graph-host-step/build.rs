//! Compiles the native host step against the patched MLX sources.
//!
//! The step is C++ and not Rust because it is an MLX primitive: MLX's
//! evaluator calls it while it walks a graph, and it queues its work on MLX's
//! own CPU stream. Neither is reachable through MLX-C, so the step is written
//! against MLX's C++ headers and linked into the same image as MLX.
//!
//! It is compiled here, with `cc`, and not as a target of the native MLX
//! build, so an edit to the step recompiles one file in a second instead of
//! re-keying the native build identity and rebuilding MLX.

use std::env;
use std::error::Error;
use std::path::Path;

use native_build_orchestration::{
    ASTRO_NATIVE_BUILD_CACHE_DEPENDENCIES_VARIABLE, pinned_mlx_macos_deployment_target,
    provisioned_mlx_sources, workspace_root_above,
};

const DIRECT_MLX_FEATURE_VARIABLE: &str = "CARGO_FEATURE_DIRECT_MLX";

fn main() -> Result<(), Box<dyn Error>> {
    println!("cargo:rerun-if-env-changed={DIRECT_MLX_FEATURE_VARIABLE}");
    if env::var_os(DIRECT_MLX_FEATURE_VARIABLE).is_none() {
        return Ok(());
    }
    let manifest_directory = env::var_os("CARGO_MANIFEST_DIR")
        .map(std::path::PathBuf::from)
        .ok_or("CARGO_MANIFEST_DIR is missing")?;
    let repository_root = workspace_root_above(&manifest_directory)
        .ok_or("no Cargo.lock above the graph-host-step crate")?
        .canonicalize()?;
    let provisioned_sources = provisioned_mlx_sources(&repository_root)?;
    emit_rerun_contracts(&repository_root);

    let mut native_step_build = cc::Build::new();
    native_step_build
        .cpp(true)
        .std("c++20")
        .file("native/graph_host_step.cpp")
        .file("native/evaluation_counts.cpp")
        .include("native")
        .warnings_into_errors(true);
    // System include paths: the warnings this crate turns into errors are for
    // its own file, and MLX's headers do not build clean under them.
    for system_include_directory in [
        &provisioned_sources.mlx_source_directory,
        &provisioned_sources.mlx_c_source_directory,
    ] {
        native_step_build
            .flag("-isystem")
            .flag(system_include_directory);
    }
    if env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        // The target MLX itself is built for: its headers choose code paths
        // by it, so the step and MLX must agree.
        native_step_build.flag(format!(
            "-mmacosx-version-min={}",
            pinned_mlx_macos_deployment_target(&repository_root)?
        ));
    }
    native_step_build.compile("astronomical_graph_host_step");
    Ok(())
}

/// The step is recompiled when its own sources change and when the pins or
/// patches do, because those select a different MLX source tree.
fn emit_rerun_contracts(repository_root: &Path) {
    println!("cargo:rerun-if-env-changed={ASTRO_NATIVE_BUILD_CACHE_DEPENDENCIES_VARIABLE}");
    for native_source in [
        "graph_host_step.cpp",
        "graph_host_step.h",
        "evaluation_counts.cpp",
        "evaluation_counts.h",
    ] {
        println!("cargo:rerun-if-changed=native/{native_source}");
    }
    for identity_input in ["third-party/pins", "third-party/patches"] {
        println!(
            "cargo:rerun-if-changed={}",
            repository_root.join(identity_input).display()
        );
    }
}
