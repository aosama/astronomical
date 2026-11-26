//! Compiles the pinned MLX performance profiler against the patched MLX
//! sources.
//!
//! The profiler reads MLX's graph and its Metal command buffers through
//! MLX's own C++ headers, so it has to be compiled against the very headers
//! the linked MLX image was built from: the provisioned extraction of the
//! pinned archives with this repository's patches applied. It is compiled
//! here, with `cc`, from its digest-verified release archive, and not built
//! with its own CMake, so it never links a second MLX and an edit to a pin
//! or a patch recompiles it against the new tree.

use std::env;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use native_build_orchestration::{
    ASTRO_NATIVE_BUILD_CACHE_DEPENDENCIES_VARIABLE, NativeRuntimeBuildInputs,
    pinned_mlx_macos_deployment_target, provisioned_mlx_sources, workspace_root_above,
};
use sha2::{Digest, Sha256};

const DIRECT_MLX_FEATURE_VARIABLE: &str = "CARGO_FEATURE_DIRECT_MLX";
const PINS_DIRECTORY: &str = "third-party/pins";
const PROFILER_ARCHIVE_VARIABLES: PinnedArchiveVariables = PinnedArchiveVariables {
    file_name: "ASTRONOMICAL_MLX_PERFORMANCE_PROFILER_SOURCE_ARCHIVE_FILE_NAME",
    sha256: "ASTRONOMICAL_MLX_PERFORMANCE_PROFILER_SOURCE_ARCHIVE_SHA256",
};
const METAL_CPP_ARCHIVE_VARIABLES: PinnedArchiveVariables = PinnedArchiveVariables {
    file_name: "ASTRONOMICAL_METAL_CPP_SOURCE_ARCHIVE_FILE_NAME",
    sha256: "ASTRONOMICAL_METAL_CPP_SOURCE_ARCHIVE_SHA256",
};
const PROFILER_VERSION_VARIABLE: &str = "ASTRONOMICAL_MLX_PERFORMANCE_PROFILER_VERSION";
/// MLX's own consumer definitions, as the profiler's CMake passes them:
/// the Metal backend's surface, Accelerate's LAPACK layout and static
/// linkage. `NDEBUG` because MLX's headers change class layouts without it
/// and the linked MLX is a release build.
const MLX_CONSUMER_DEFINITIONS: [&str; 4] =
    ["MLX_STATIC", "_METAL_", "ACCELERATE_NEW_LAPACK", "NDEBUG"];
/// The library target of the profiler's `CMakeLists.txt` at the pinned
/// release: capture, timelines, reports, procedure comparisons, kernel
/// launch records and the C API, without its demo, server or model code. One more source is generated
/// (`embedded_graph_view_assets`).
const PROFILER_LIBRARY_SOURCES: [&str; 61] = [
    "graph_model.cpp",
    "graph_build_timer.cpp",
    "telemetry.cpp",
    "latency_breakdown.cpp",
    "progress_reporter.cpp",
    "metal_command_buffer_probe.cpp",
    "dispatch_occupancy.cpp",
    "gpu_device_facts.cpp",
    "metal_dispatch_recorder.cpp",
    "metal_object_classes.cpp",
    "metal_host_clock.cpp",
    "metal_submission_recorder.cpp",
    "gpu_busy_time.cpp",
    "procedure_measurement.cpp",
    "report_procedure_page.cpp",
    "procedure_comparison_record.cpp",
    "procedure_report_files.cpp",
    "kernel_launch_summary.cpp",
    "gpu_timeline.cpp",
    "timeline_dependencies.cpp",
    "timed_units.cpp",
    "stage_plan.cpp",
    "stage_timeline.cpp",
    "timeline_console.cpp",
    "text_format.cpp",
    "latency_slices.cpp",
    "dot_writer.cpp",
    "graph_view_section.cpp",
    "report_assets.cpp",
    "report_components.cpp",
    "report_summary_section.cpp",
    "report_task_table_section.cpp",
    "report_stage_table_section.cpp",
    "report_gantt_section.cpp",
    "report_comparison_page.cpp",
    "report_launch_comparison_section.cpp",
    "report_verdict_section.cpp",
    "comparison_record.cpp",
    "report_writer.cpp",
    "report_bundle.cpp",
    "sample_statistics.cpp",
    "output_agreement.cpp",
    "measurement_session.cpp",
    "paired_measurement.cpp",
    "measurement_document.cpp",
    "measurement_file.cpp",
    "json_value.cpp",
    "json_writer.cpp",
    "console_output.cpp",
    "mlx_c_interop.cpp",
    "c_api_support.cpp",
    "mlx_perf_c_api.cpp",
    "mlx_perf_c_api_procedures.cpp",
    "mlx_perf_c_api_wired_memory.cpp",
    "system_wired_memory.cpp",
    "wired_memory_sample.cpp",
    "wired_memory_sampler.cpp",
    "wired_memory_record.cpp",
    "wired_memory_analysis.cpp",
    "wire_release_probe.cpp",
    "report_wired_memory_section.cpp",
];

/// The report's interactive graph view: the function each embedded file is
/// served by and the file, in the pairs the profiler's `CMakeLists.txt`
/// declares. Two are vendored libraries (Cytoscape.js and its dagre layout,
/// MIT), the rest the profiler's own script and stylesheet.
const GRAPH_VIEW_ASSETS: [(&str, &str); 7] = [
    (
        "graph_view_cytoscape_script",
        "third_party/cytoscape/cytoscape.min.js",
    ),
    (
        "graph_view_dagre_layout_script",
        "third_party/cytoscape-dagre/cytoscape-dagre.min.js",
    ),
    (
        "graph_view_model_script",
        "src/graph_view/graph_view_model.js",
    ),
    (
        "graph_view_style_script",
        "src/graph_view/graph_view_style.js",
    ),
    (
        "graph_view_panels_script",
        "src/graph_view/graph_view_panels.js",
    ),
    ("graph_view_viewer_script", "src/graph_view/graph_view.js"),
    ("graph_view_stylesheet", "src/graph_view/graph_view.css"),
];

/// What the profiler is compiled from: its extracted release and the one
/// source generated from it.
struct ProfilerSources {
    profiler_directory: PathBuf,
    embedded_assets_source: PathBuf,
}

/// The two pin variables that name an archive and its digest.
struct PinnedArchiveVariables {
    file_name: &'static str,
    sha256: &'static str,
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("cargo:rerun-if-env-changed={DIRECT_MLX_FEATURE_VARIABLE}");
    if env::var_os(DIRECT_MLX_FEATURE_VARIABLE).is_none() {
        return Ok(());
    }
    let manifest_directory =
        PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").ok_or("CARGO_MANIFEST_DIR is missing")?);
    let repository_root = workspace_root_above(&manifest_directory)
        .ok_or("no Cargo.lock above the mlx-performance-profiler crate")?
        .canonicalize()?;
    println!("cargo:rerun-if-env-changed={ASTRO_NATIVE_BUILD_CACHE_DEPENDENCIES_VARIABLE}");
    for identity_input in ["third-party/pins", "third-party/patches"] {
        println!(
            "cargo:rerun-if-changed={}",
            repository_root.join(identity_input).display()
        );
    }
    let output_directory = PathBuf::from(env::var_os("OUT_DIR").ok_or("OUT_DIR is missing")?);
    let profiler_pin = pin_that_sets(&repository_root, PROFILER_VERSION_VARIABLE)?;
    let mlx_pin = pin_that_sets(&repository_root, METAL_CPP_ARCHIVE_VARIABLES.file_name)?;
    let profiler_directory = extracted_archive(
        &profiler_pin,
        &PROFILER_ARCHIVE_VARIABLES,
        &output_directory.join("profiler"),
    )?;
    let metal_cpp_directory = extracted_archive(
        &mlx_pin,
        &METAL_CPP_ARCHIVE_VARIABLES,
        &output_directory.join("metal_cpp"),
    )?;
    let generated_include_directory = generated_version_header(
        &profiler_directory,
        pin_value(&profiler_pin, PROFILER_VERSION_VARIABLE)?,
        &output_directory,
    )?;
    let embedded_assets_source =
        embedded_graph_view_assets(&profiler_directory, &output_directory)?;
    compile_profiler(
        &repository_root,
        &ProfilerSources {
            profiler_directory,
            embedded_assets_source,
        },
        &[&generated_include_directory, &metal_cpp_directory],
    )
}

/// The text of the pin file that sets `variable`. Pins are found by what
/// they set, not by their file names, so a bump renames a file and edits
/// nothing here.
fn pin_that_sets(repository_root: &Path, variable: &str) -> Result<String, Box<dyn Error>> {
    let pins_directory = repository_root.join(PINS_DIRECTORY);
    let setting = format!("set({variable} \"");
    for pin_entry in fs::read_dir(&pins_directory)? {
        let pin_text = fs::read_to_string(pin_entry?.path())?;
        if pin_text.contains(&setting) {
            return Ok(pin_text);
        }
    }
    Err(format!("no pin under {} sets {variable}", pins_directory.display()).into())
}

/// The value a pin file sets `variable` to: `set(VARIABLE "value")`.
fn pin_value<'pin>(pin_text: &'pin str, variable: &str) -> Result<&'pin str, Box<dyn Error>> {
    let setting = format!("set({variable} \"");
    let value_start = pin_text
        .find(&setting)
        .ok_or_else(|| format!("the pin does not set {variable}"))?
        + setting.len();
    let value_length = pin_text[value_start..]
        .find('"')
        .ok_or_else(|| format!("the pin's {variable} is not closed"))?;
    Ok(&pin_text[value_start..value_start + value_length])
}

/// Verifies a provisioned archive against its pinned SHA-256 and extracts it
/// into `extraction_directory`, returning the one directory it holds.
fn extracted_archive(
    pin_text: &str,
    archive_variables: &PinnedArchiveVariables,
    extraction_directory: &Path,
) -> Result<PathBuf, Box<dyn Error>> {
    let archive_path = verified_archive(pin_text, archive_variables)?;
    if extraction_directory.exists() {
        fs::remove_dir_all(extraction_directory)?;
    }
    fs::create_dir_all(extraction_directory)?;
    let is_zip_archive = archive_path
        .extension()
        .is_some_and(|extension| extension == "zip");
    let mut extraction = if is_zip_archive {
        let mut unzip = Command::new("unzip");
        unzip.arg("-q").arg(&archive_path).arg("-d");
        unzip
    } else {
        let mut untar = Command::new("tar");
        untar.arg("-xzf").arg(&archive_path).arg("-C");
        untar
    };
    if !extraction.arg(extraction_directory).status()?.success() {
        return Err(format!("could not extract {}", archive_path.display()).into());
    }
    for extracted_entry in fs::read_dir(extraction_directory)? {
        let extracted_path = extracted_entry?.path();
        if extracted_path.is_dir() {
            return Ok(extracted_path);
        }
    }
    Err(format!("{} holds no directory", archive_path.display()).into())
}

/// The provisioned archive a pin names, once its bytes hash to the pinned
/// SHA-256: nothing is compiled from an archive that does not.
fn verified_archive(
    pin_text: &str,
    archive_variables: &PinnedArchiveVariables,
) -> Result<PathBuf, Box<dyn Error>> {
    let pinned_sha256 = pin_value(pin_text, archive_variables.sha256)?;
    let native_build_inputs = NativeRuntimeBuildInputs::from_environment();
    let archive_path = native_build_inputs
        .astro_native_build_cache_dependencies_directory()
        .ok_or("no native dependency directory: set HOME")?
        .join(pin_value(pin_text, archive_variables.file_name)?);
    let archive_bytes = fs::read(&archive_path).map_err(|read_error| {
        format!(
            "{} is not provisioned ({read_error}); run cargo xtask bootstrap-native-dependencies",
            archive_path.display()
        )
    })?;
    let mut archive_sha256 = String::with_capacity(64);
    for digest_byte in Sha256::digest(&archive_bytes) {
        archive_sha256.push_str(&format!("{digest_byte:02x}"));
    }
    if archive_sha256 != pinned_sha256 {
        return Err(format!(
            "{} has SHA-256 {archive_sha256}, the pin says {pinned_sha256}",
            archive_path.display()
        )
        .into());
    }
    Ok(archive_path)
}

/// Writes the version header the profiler's CMake would generate from the
/// project version, and returns the include directory that holds it.
fn generated_version_header(
    profiler_directory: &Path,
    profiler_version: &str,
    output_directory: &Path,
) -> Result<PathBuf, Box<dyn Error>> {
    let version_parts: Vec<&str> = profiler_version.split('.').collect();
    let [major, minor, patch] = version_parts.as_slice() else {
        return Err(
            format!("the profiler version {profiler_version} is not MAJOR.MINOR.PATCH").into(),
        );
    };
    let header_template =
        fs::read_to_string(profiler_directory.join("include/mlx_perf/mlx_perf_version.h.in"))?;
    let version_header = header_template
        .replace("@PROJECT_VERSION_MAJOR@", major)
        .replace("@PROJECT_VERSION_MINOR@", minor)
        .replace("@PROJECT_VERSION_PATCH@", patch)
        .replace("@PROJECT_VERSION@", profiler_version);
    let generated_include_directory = output_directory.join("generated/include");
    let header_directory = generated_include_directory.join("mlx_perf");
    fs::create_dir_all(&header_directory)?;
    fs::write(header_directory.join("mlx_perf_version.h"), version_header)?;
    Ok(generated_include_directory)
}

/// Embeds the graph view's scripts and stylesheet in a generated C++ source,
/// with the profiler's own CMake script, so every report stays one HTML file
/// that opens offline. Run as the profiler's build runs it, in script mode.
fn embedded_graph_view_assets(
    profiler_directory: &Path,
    output_directory: &Path,
) -> Result<PathBuf, Box<dyn Error>> {
    let mut asset_functions = Vec::with_capacity(GRAPH_VIEW_ASSETS.len());
    let mut asset_files = Vec::with_capacity(GRAPH_VIEW_ASSETS.len());
    for (asset_function, asset_file) in GRAPH_VIEW_ASSETS {
        asset_functions.push(asset_function.to_owned());
        asset_files.push(profiler_directory.join(asset_file).display().to_string());
    }
    let embedded_assets_source = output_directory.join("generated/graph_view_assets.cpp");
    let embedding_status = Command::new("cmake")
        .arg(format!(
            "-DOUTPUT_FILE={}",
            embedded_assets_source.display()
        ))
        .arg("-DHEADER_NAME=graph_view_assets.h")
        .arg(format!("-DASSET_FUNCTIONS={}", asset_functions.join("|")))
        .arg(format!("-DASSET_FILES={}", asset_files.join("|")))
        .arg("-P")
        .arg(profiler_directory.join("cmake/mlx_perf_embed_text_assets.cmake"))
        .status()?;
    if !embedding_status.success() {
        return Err("the profiler's graph view assets could not be embedded".into());
    }
    Ok(embedded_assets_source)
}

/// Compiles the profiler's library sources into this crate's archive.
fn compile_profiler(
    repository_root: &Path,
    profiler_sources: &ProfilerSources,
    [generated_include_directory, metal_cpp_directory]: &[&PathBuf; 2],
) -> Result<(), Box<dyn Error>> {
    let profiler_directory = &profiler_sources.profiler_directory;
    let provisioned_sources = provisioned_mlx_sources(repository_root)?;
    let mut profiler_build = cc::Build::new();
    profiler_build
        .cpp(true)
        .std("c++20")
        // The profiler times its own host-side work, so it is optimized
        // whatever profile the measurement test is built in.
        .opt_level(2)
        .include(profiler_directory.join("include"))
        .include(generated_include_directory)
        .include(profiler_directory.join("src"));
    for library_source in PROFILER_LIBRARY_SOURCES {
        profiler_build.file(profiler_directory.join("src").join(library_source));
    }
    profiler_build.file(&profiler_sources.embedded_assets_source);
    for consumer_definition in MLX_CONSUMER_DEFINITIONS {
        profiler_build.define(consumer_definition, None);
    }
    // System include paths: MLX's and Metal's headers are not this
    // repository's code and do not build clean under warnings.
    for system_include_directory in [
        &provisioned_sources.mlx_source_directory,
        &provisioned_sources.mlx_c_source_directory,
        *metal_cpp_directory,
    ] {
        profiler_build
            .flag("-isystem")
            .flag(system_include_directory);
    }
    if env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        // The target MLX itself is built for: its headers choose code paths
        // by it, so the profiler and MLX must agree.
        profiler_build.flag(format!(
            "-mmacosx-version-min={}",
            pinned_mlx_macos_deployment_target(repository_root)?
        ));
    }
    profiler_build.compile("astronomical_mlx_performance_profiler");
    Ok(())
}
