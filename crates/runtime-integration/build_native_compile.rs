//! Pinned native runtime compilation shared by the Cargo build script and the
//! standalone native-build tool.
//!
//! Both entry points include this module through `#[path]`, so it must stay
//! free of `cargo:` directives and reach its sibling build modules only
//! through `crate::` paths that every includer declares. Like the other build
//! orchestration sources, this file is excluded from the native identity
//! fingerprint: semantic CMake changes require bumping
//! `native-build-store-schema-version` instead.

use std::{
    env,
    error::Error,
    fs::File,
    io::Write,
    path::{Path, PathBuf},
    process::Command,
};

use crate::build_native_store::{NativeBuildArtifacts, NativeBuildProfile};
use crate::build_parallelism;
use crate::build_progress::NativeBuildProgress;

pub const RUSTC_WRAPPER_VARIABLE: &str = "RUSTC_WRAPPER";
pub const NATIVE_DEPENDENCY_CACHE_VARIABLE: &str = "ASTRONOMICAL_NATIVE_DEPENDENCY_CACHE_DIR";
pub const NATIVE_BUILD_STORE_VARIABLE: &str = "ASTRONOMICAL_NATIVE_BUILD_STORE_DIR";
pub const NATIVE_BUILD_STATUS_FILE_VARIABLE: &str = "ASTRONOMICAL_NATIVE_BUILD_STATUS_FILE";
pub const NATIVE_ARCHIVE_VARIABLES: [&str; 5] = [
    "ASTRONOMICAL_MLX_SOURCE_ARCHIVE",
    "ASTRONOMICAL_MLX_C_SOURCE_ARCHIVE",
    "ASTRONOMICAL_METAL_CPP_SOURCE_ARCHIVE",
    "ASTRONOMICAL_JSON_SOURCE_ARCHIVE",
    "ASTRONOMICAL_FMT_SOURCE_ARCHIVE",
];

const SCCACHE_EXECUTABLE_NAME: &str = "sccache";

/// Native inputs resolved from the environment once per run so the CMake
/// configuration and Cargo's path tracking observe the same values.
pub struct NativeRuntimeBuildInputs {
    native_dependency_cache_directory: Option<PathBuf>,
    native_archive_paths: Vec<(&'static str, PathBuf)>,
}

impl NativeRuntimeBuildInputs {
    pub fn from_environment() -> Self {
        Self {
            native_dependency_cache_directory: native_dependency_cache_directory(),
            native_archive_paths: NATIVE_ARCHIVE_VARIABLES
                .into_iter()
                .filter_map(|archive_variable| {
                    env::var_os(archive_variable)
                        .map(PathBuf::from)
                        .map(|archive_path| (archive_variable, archive_path))
                })
                .collect(),
        }
    }

    pub fn native_dependency_cache_directory(&self) -> Option<&Path> {
        self.native_dependency_cache_directory.as_deref()
    }

    pub fn native_archive_paths(&self) -> &[(&'static str, PathBuf)] {
        &self.native_archive_paths
    }

    fn archive_path(&self, archive_variable: &str) -> Option<&Path> {
        self.native_archive_paths
            .iter()
            .find(|(variable_name, _)| *variable_name == archive_variable)
            .map(|(_, archive_path)| archive_path.as_path())
    }
}

pub fn resolve_native_build_identity(
    repository_root: &Path,
    native_build_profile: &str,
    target_triple: &str,
) -> Result<String, Box<dyn Error>> {
    let fingerprint_script_path = repository_root.join("scripts/native-build-cache-fingerprint.sh");
    let mut identity_command = Command::new(&fingerprint_script_path);
    identity_command
        .arg("--profile")
        .arg(native_build_profile)
        .arg(repository_root)
        // Cargo exports TARGET to build scripts; the standalone native-build
        // tool has no Cargo context, so every caller states the triple it
        // builds for and the identity stays identical in both paths.
        .env("TARGET", target_triple);
    // Fixture overrides make identity contracts hermetic, but a production
    // build must always fingerprint the toolchain that CMake actually uses.
    for override_variable in [
        "ASTRONOMICAL_NATIVE_IDENTITY_XCODE",
        "ASTRONOMICAL_NATIVE_IDENTITY_SDK",
        "ASTRONOMICAL_NATIVE_IDENTITY_CLANG",
        "ASTRONOMICAL_NATIVE_IDENTITY_CMAKE",
        "ASTRONOMICAL_NATIVE_IDENTITY_RUSTC",
        "ASTRONOMICAL_NATIVE_IDENTITY_TARGET",
    ] {
        identity_command.env_remove(override_variable);
    }
    let identity_output = identity_command.output()?;
    if !identity_output.status.success() {
        let diagnostic_text = String::from_utf8_lossy(&identity_output.stderr);
        return Err(format!(
            "native build identity failed with {}: {}",
            identity_output.status,
            diagnostic_text.trim()
        )
        .into());
    }
    let native_build_identity = String::from_utf8(identity_output.stdout)?;
    let native_build_identity = native_build_identity.trim().to_owned();
    if !identity_output.stderr.is_empty() {
        eprint!("{}", String::from_utf8_lossy(&identity_output.stderr));
    }
    Ok(native_build_identity)
}

pub fn native_build_store_directory() -> Result<PathBuf, Box<dyn Error>> {
    let store_directory = env::var_os(NATIVE_BUILD_STORE_VARIABLE)
        .map(PathBuf::from)
        .or_else(|| {
            env::var_os("HOME")
                .map(PathBuf::from)
                .map(|home_directory| {
                    home_directory.join("Library/Caches/Astronomical/native-builds")
                })
        })
        .ok_or_else(|| {
            format!("set {NATIVE_BUILD_STORE_VARIABLE} or HOME to select native build storage")
        })?;
    if !store_directory.is_absolute() {
        return Err(format!(
            "{NATIVE_BUILD_STORE_VARIABLE} must be an absolute path: {}",
            store_directory.display()
        )
        .into());
    }
    Ok(store_directory)
}

pub fn build_pinned_native_runtime(
    native_source_directory: &Path,
    native_build_directory: &Path,
    native_build_profile: NativeBuildProfile,
    native_build_inputs: &NativeRuntimeBuildInputs,
    native_build_progress: &NativeBuildProgress,
) -> Result<(), Box<dyn Error>> {
    let mut configure_command = Command::new("cmake");
    let clang_compiler_path = discover_xcrun_path(&["--find", "clang"])?;
    let clang_cxx_compiler_path = discover_xcrun_path(&["--find", "clang++"])?;
    let macos_sdk_path = discover_xcrun_path(&["--sdk", "macosx", "--show-sdk-path"])?;
    configure_command
        .arg("-G")
        .arg("Unix Makefiles")
        .arg("-S")
        .arg(native_source_directory)
        .arg("-B")
        .arg(native_build_directory)
        .arg("-DCMAKE_BUILD_TYPE=Release")
        .arg(format!(
            "-DCMAKE_C_COMPILER={}",
            clang_compiler_path.display()
        ))
        .arg(format!(
            "-DCMAKE_CXX_COMPILER={}",
            clang_cxx_compiler_path.display()
        ))
        .arg(format!("-DCMAKE_OSX_SYSROOT={}", macos_sdk_path.display()))
        .arg("-DCMAKE_OSX_ARCHITECTURES=arm64");
    remove_uncontrolled_native_environment(&mut configure_command);
    let compiler_launcher_argument = sccache_compiler_launcher_path()
        .map_or_else(String::new, |launcher_path| {
            launcher_path.display().to_string()
        });
    configure_command
        .arg(format!(
            "-DCMAKE_C_COMPILER_LAUNCHER={compiler_launcher_argument}"
        ))
        .arg(format!(
            "-DCMAKE_CXX_COMPILER_LAUNCHER={compiler_launcher_argument}"
        ))
        .arg(format!(
            "-DASTRONOMICAL_BUILD_MEMORY_CONTRACT_PROBE={}",
            cmake_boolean(native_build_profile.should_build_memory_contract_probe())
        ));
    append_native_archive_configuration(&mut configure_command, native_build_inputs);
    run_command(
        &mut configure_command,
        "native-configure",
        native_build_progress,
    )?;

    let mut native_build_command = Command::new("cmake");
    native_build_command
        .arg("--build")
        .arg(native_build_directory)
        .arg("--target")
        .arg("mlxc");
    native_build_command
        .arg("--parallel")
        .arg(native_parallel_job_count());
    remove_uncontrolled_native_environment(&mut native_build_command);
    run_command(
        &mut native_build_command,
        "native-compile",
        native_build_progress,
    )?;

    if native_build_profile.should_build_memory_contract_probe() {
        let mut probe_build_command = Command::new("cmake");
        probe_build_command
            .arg("--build")
            .arg(native_build_directory)
            .arg("--target")
            .arg("mlx_memory_contract_probe")
            .arg("--parallel")
            .arg(native_parallel_job_count());
        remove_uncontrolled_native_environment(&mut probe_build_command);
        run_command(
            &mut probe_build_command,
            "native-memory-contract-probe",
            native_build_progress,
        )?;
    }
    Ok(())
}

fn append_native_archive_configuration(
    configure_command: &mut Command,
    native_build_inputs: &NativeRuntimeBuildInputs,
) {
    if let Some(native_dependency_cache_directory) =
        native_build_inputs.native_dependency_cache_directory()
    {
        configure_command.arg(format!(
            "-D{NATIVE_DEPENDENCY_CACHE_VARIABLE}={}",
            native_dependency_cache_directory.display()
        ));
    }
    for archive_variable in NATIVE_ARCHIVE_VARIABLES {
        match native_build_inputs.archive_path(archive_variable) {
            Some(archive_path) => {
                configure_command.arg(format!("-D{archive_variable}={}", archive_path.display()));
            }
            None => {
                configure_command.arg("-U").arg(archive_variable);
            }
        }
    }
}

pub fn write_native_build_status(
    native_build_artifacts: &NativeBuildArtifacts,
) -> Result<(), Box<dyn Error>> {
    let Some(status_file_path) = env::var_os(NATIVE_BUILD_STATUS_FILE_VARIABLE).map(PathBuf::from)
    else {
        return Ok(());
    };
    if !status_file_path.is_absolute() {
        return Err(format!(
            "{NATIVE_BUILD_STATUS_FILE_VARIABLE} must be an absolute path: {}",
            status_file_path.display()
        )
        .into());
    }
    if let Some(status_parent_directory) = status_file_path.parent() {
        std::fs::create_dir_all(status_parent_directory)?;
    }
    let status_text = if native_build_artifacts.was_built() {
        "built\n"
    } else {
        "reused\n"
    };
    let mut status_file = File::create(status_file_path)?;
    status_file.write_all(status_text.as_bytes())?;
    status_file.sync_all()?;
    Ok(())
}

pub fn required_path_variable(variable_name: &str) -> Result<PathBuf, Box<dyn Error>> {
    env::var_os(variable_name)
        .map(PathBuf::from)
        .ok_or_else(|| format!("required environment variable {variable_name} is missing").into())
}

// The standalone native-build tool has no Cargo context and detects the host
// triple itself; the Cargo build script reads TARGET straight from the
// environment, so this detector is dead code from the build script's view.
#[allow(dead_code)]
pub fn detect_host_target_triple() -> Result<String, Box<dyn Error>> {
    let rustc_version_output = Command::new("rustc")
        .arg("--version")
        .arg("--verbose")
        .output()?;
    if !rustc_version_output.status.success() {
        let diagnostic_text = String::from_utf8_lossy(&rustc_version_output.stderr);
        return Err(format!(
            "rustc --version --verbose failed with {}: {}",
            rustc_version_output.status,
            diagnostic_text.trim()
        )
        .into());
    }
    let verbose_text = String::from_utf8(rustc_version_output.stdout)?;
    for verbose_line in verbose_text.lines() {
        if let Some(host_triple) = verbose_line.strip_prefix("host: ") {
            return Ok(host_triple.trim().to_owned());
        }
    }
    Err("rustc --version --verbose did not report a host triple".into())
}

fn native_dependency_cache_directory() -> Option<PathBuf> {
    env::var_os(NATIVE_DEPENDENCY_CACHE_VARIABLE)
        .map(PathBuf::from)
        .or_else(|| {
            env::var_os("HOME")
                .map(PathBuf::from)
                .map(|home_directory| {
                    home_directory.join("Library/Caches/Astronomical/native-dependencies")
                })
        })
}

fn discover_xcrun_path(arguments: &[&str]) -> Result<PathBuf, Box<dyn Error>> {
    let path_output = Command::new("xcrun").args(arguments).output()?;
    if !path_output.status.success() {
        return Err(format!("xcrun failed to resolve {}", arguments.join(" ")).into());
    }
    let path_text = String::from_utf8(path_output.stdout)?;
    let resolved_path = PathBuf::from(path_text.trim());
    if !resolved_path.is_absolute() {
        return Err(format!(
            "xcrun reported a non-absolute path for {}",
            arguments.join(" ")
        )
        .into());
    }
    Ok(resolved_path)
}

fn remove_uncontrolled_native_environment(command: &mut Command) {
    for environment_variable in [
        "ARCHFLAGS",
        "CC",
        "CFLAGS",
        "CMAKE_GENERATOR",
        "CMAKE_OSX_ARCHITECTURES",
        "CMAKE_OSX_DEPLOYMENT_TARGET",
        "CMAKE_OSX_SYSROOT",
        "CMAKE_TOOLCHAIN_FILE",
        "CPPFLAGS",
        "CXX",
        "CXXFLAGS",
        "LDFLAGS",
        "MACOSX_DEPLOYMENT_TARGET",
        "SDKROOT",
    ] {
        command.env_remove(environment_variable);
    }
}

pub fn native_parallel_job_count() -> String {
    build_parallelism::resolve_native_parallel_job_count(
        env::var(build_parallelism::NATIVE_BUILD_JOBS_VARIABLE)
            .ok()
            .as_deref(),
        env::var("CARGO_BUILD_JOBS").ok().as_deref(),
        env::var("NUM_JOBS").ok().as_deref(),
        std::thread::available_parallelism()
            .map(|available_parallelism| available_parallelism.get())
            .unwrap_or(1),
    )
}

fn sccache_compiler_launcher_path() -> Option<PathBuf> {
    let rustc_wrapper_path = env::var_os(RUSTC_WRAPPER_VARIABLE).map(PathBuf::from)?;
    let wrapper_file_name = rustc_wrapper_path.file_name()?.to_str()?;
    (wrapper_file_name == SCCACHE_EXECUTABLE_NAME).then_some(rustc_wrapper_path)
}

fn cmake_boolean(boolean_value: bool) -> &'static str {
    if boolean_value { "ON" } else { "OFF" }
}

fn run_command(
    command: &mut Command,
    operation: &str,
    native_build_progress: &NativeBuildProgress,
) -> Result<(), Box<dyn Error>> {
    let operation_progress = native_build_progress.begin_operation(operation);
    let command_status = command.status()?;
    if !command_status.success() {
        let elapsed_seconds = operation_progress.complete("failed").as_secs_f64();
        return Err(format!(
            "native operation {operation} failed after {elapsed_seconds:.3} seconds: {command_status}"
        )
        .into());
    }
    operation_progress.complete("success");
    Ok(())
}
