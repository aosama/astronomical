//! Entry point of `cargo xtask provision-bindgen-headers`: resolves the
//! directories and the patch pipeline once, then runs the selected mode.

use std::path::Path;
use std::time::Instant;

use super::header_locations::{native_build_store_directory, native_dependency_cache_directory};
use super::options::{ProvisionMode, ProvisionOptions, parse_provision_options};
use super::patch_pipeline::{
    PatchStep, parse_cmake_patch_pipeline, parse_patch_list, render_patch_list, require_patch_files,
};
use super::provision_flow::provision_headers;
use super::step_log::start_step;
use super::verify_flow::verify_headers_against_native_build;

const NATIVE_CMAKE_LISTS_PATH: &str = "crates/runtime-integration/native/CMakeLists.txt";

pub fn run(repository_root: &Path, arguments: &[String]) -> Result<(), String> {
    let provision_options = parse_provision_options(arguments)?;
    let run_started_at = Instant::now();
    let inputs_step = start_step("validate-inputs");
    let cache_directory =
        native_dependency_cache_directory(provision_options.cache_directory.as_deref())?;
    let store_directory =
        native_build_store_directory(provision_options.native_build_store_directory.as_deref())?;
    inputs_step.finish("success");
    let pipeline_step = start_step("parse-patch-pipeline");
    let patch_steps = effective_patch_steps(repository_root, &provision_options)?;
    pipeline_step.finish("success");
    let mode_result = match provision_options.mode {
        ProvisionMode::PrintPatchList => {
            print!("{}", render_patch_list(&patch_steps));
            Ok(())
        }
        ProvisionMode::VerifyHeaders => verify_headers_against_native_build(
            repository_root,
            &provision_options,
            &cache_directory,
            &store_directory,
        ),
        ProvisionMode::Provision => provision_headers(
            repository_root,
            &provision_options,
            &cache_directory,
            &patch_steps,
        ),
    };
    if mode_result.is_ok() {
        println!(
            "[provision-bindgen-headers] status=success elapsed_seconds={:.1}",
            run_started_at.elapsed().as_secs_f64()
        );
    }
    mode_result
}

fn effective_patch_steps(
    repository_root: &Path,
    provision_options: &ProvisionOptions,
) -> Result<Vec<PatchStep>, String> {
    let patch_steps = match &provision_options.explicit_patch_list_path {
        Some(patch_list_path) => match std::fs::read_to_string(patch_list_path) {
            Ok(patch_list_text) => parse_patch_list(&patch_list_text)?,
            Err(read_error) => {
                return Err(format!("could not read {patch_list_path}: {read_error}"));
            }
        },
        None => match std::fs::read_to_string(repository_root.join(NATIVE_CMAKE_LISTS_PATH)) {
            Ok(cmake_lists_text) => parse_cmake_patch_pipeline(&cmake_lists_text),
            Err(read_error) => {
                return Err(format!(
                    "could not read {NATIVE_CMAKE_LISTS_PATH}: {read_error}"
                ));
            }
        },
    };
    require_patch_files(repository_root, &patch_steps)?;
    Ok(patch_steps)
}
