//! Pinned native runtime compilation shared by the Cargo build script and the
//! standalone native-build tool.
//!
//! # The sharing mechanism
//!
//! Both entry points include this module through `#[path]`, so it must stay
//! free of `cargo:` directives and reach its sibling build modules only
//! through `crate::` paths that every includer declares. Like the other build
//! orchestration sources, this file is excluded from the native identity
//! fingerprint: semantic CMake changes require bumping
//! `native-build-store-schema-version` instead — progress-logging or
//! parallelism edits here must not invalidate anyone's native store entry.
//!
//! # The toolchain is fully controlled, never ambient
//!
//! The configure command pins the compilers, the SDK (Software Development
//! Kit), the architecture, and the build type, and
//! [`remove_uncontrolled_native_environment`] strips every ambient compiler
//! variable before CMake runs. Rationale: the native identity fingerprints
//! the toolchain it *probes*; if ambient `CC`/`CFLAGS`/`SDKROOT` could leak
//! into the actual build, the produced bytes would depend on state the
//! fingerprint never saw, and store reuse would serve mismatched artifacts.
//!
//! # Verified dependency provisioning
//!
//! `cargo xtask bootstrap-native-dependencies` provisions the pinned archive
//! filenames under one dependency cache directory. CMake resolves every
//! archive from that directory and re-verifies its SHA-256 before extraction
//! (see `native/CMakeLists.txt`); per-archive environment overrides are absent.

use std::{
    env,
    error::Error,
    fs::File,
    io::Write,
    path::{Path, PathBuf},
    process::Command,
};

use crate::native_parallelism;
use crate::native_progress::NativeBuildProgress;
use crate::native_store::{NativeBuildArtifacts, NativeBuildProfile};

/// Consulted so the CMake compile can share sccache with Cargo without any
/// other wrapper silently changing compiler semantics.
pub const RUSTC_WRAPPER_VARIABLE: &str = "RUSTC_WRAPPER";
pub const NATIVE_DEPENDENCY_CACHE_VARIABLE: &str = "ASTRONOMICAL_NATIVE_DEPENDENCY_CACHE_DIR";
pub const NATIVE_BUILD_STORE_VARIABLE: &str = "ASTRONOMICAL_NATIVE_BUILD_STORE_DIR";
pub const NATIVE_BUILD_STATUS_FILE_VARIABLE: &str = "ASTRONOMICAL_NATIVE_BUILD_STATUS_FILE";
const SCCACHE_EXECUTABLE_NAME: &str = "sccache";

/// Native inputs resolved from the environment once per run so the CMake
/// configuration and Cargo's path tracking observe the same values.
pub struct NativeRuntimeBuildInputs {
    native_dependency_cache_directory: Option<PathBuf>,
}

impl NativeRuntimeBuildInputs {
    pub fn from_environment() -> Self {
        Self {
            native_dependency_cache_directory: native_dependency_cache_directory(),
        }
    }

    pub fn native_dependency_cache_directory(&self) -> Option<&Path> {
        self.native_dependency_cache_directory.as_deref()
    }
}

/// Resolves the native build identity for a build of `native_build_profile`
/// on `target_triple`.
///
/// The identity library lives in `crates/native-identity`; this wrapper is
/// the single choke point both build paths (Cargo build script and
/// standalone tool) share, which is what guarantees they key the *same*
/// store entry. Overrides stay absent by construction: a production build
/// must always fingerprint the toolchain that CMake actually uses.
pub fn resolve_native_build_identity(
    repository_root: &Path,
    native_build_profile: &str,
    target_triple: &str,
) -> Result<String, Box<dyn Error>> {
    let profile =
        astronomical_native_identity::NativeBuildProfile::from_profile_text(native_build_profile)
            .map_err(|error| error.to_string())?;
    // The standalone native-build tool states its triple here because it has
    // no Cargo context; the identity stays identical in both paths.
    let identity_request = astronomical_native_identity::FingerprintRequest {
        repository_root,
        profile,
        scope: astronomical_native_identity::IdentityScope::Full,
        overrides: astronomical_native_identity::probes::ToolchainOverrides {
            target_triple: Some(target_triple.to_owned()),
            ..astronomical_native_identity::probes::ToolchainOverrides::none()
        },
    };
    astronomical_native_identity::native_build_identity(&identity_request)
        .map_err(|error| format!("native build identity failed: {error}").into())
}

/// Selects the machine-wide store root.
///
/// An explicit `ASTRONOMICAL_NATIVE_BUILD_STORE_DIR` wins; the default lives
/// under the user's cache directory so multiple checkouts and worktrees share
/// one warm store instead of rebuilding MLX per checkout.
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

/// Configures and builds the pinned MLX/MLX-C native runtime with CMake.
///
/// Sequence: configure (compilers, SDK, arm64, Release, sanitized
/// environment, archive locations, optional probe target) → build the
/// `mlxc` aggregate target → optionally build the memory-contract probe
/// binary. Every command reports through the progress recorder so hosted CI
/// can watch the silent multi-minute stretches.
pub fn build_pinned_native_runtime(
    native_source_directory: &Path,
    native_build_directory: &Path,
    native_build_profile: NativeBuildProfile,
    native_build_inputs: &NativeRuntimeBuildInputs,
    native_build_progress: &NativeBuildProgress,
) -> Result<(), Box<dyn Error>> {
    let mut configure_command = Command::new("cmake");
    // Resolve the toolchain through xcrun rather than trusting PATH so the
    // compilers are the ones the identity's toolchain probes measured.
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
    if let Some(native_dependency_cache_directory) =
        native_build_inputs.native_dependency_cache_directory()
    {
        configure_command.arg(format!(
            "-D{NATIVE_DEPENDENCY_CACHE_VARIABLE}={}",
            native_dependency_cache_directory.display()
        ));
    } else {
        configure_command
            .arg("-U")
            .arg(NATIVE_DEPENDENCY_CACHE_VARIABLE);
    }
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

/// Writes the one-word build status (`built` or `reused`) for CI steps that
/// must distinguish a fresh compile from a store hit.
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

/// Default location of the provisioned dependency cache shared by every
/// checkout on the machine.
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

/// Strips ambient compiler/toolchain variables from a CMake invocation.
///
/// These variables change native output bytes without changing anything the
/// native identity fingerprints; removing them keeps the produced artifact a
/// pure function of the pinned, probed toolchain. A missing variable is
/// simply absent — no error, because unset is the normal, controlled state.
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

/// Resolves the native parallel job count through the shared policy module.
pub fn native_parallel_job_count() -> String {
    native_parallelism::resolve_native_parallel_job_count(
        env::var(native_parallelism::NATIVE_BUILD_JOBS_VARIABLE)
            .ok()
            .as_deref(),
        env::var("CARGO_BUILD_JOBS").ok().as_deref(),
        env::var("NUM_JOBS").ok().as_deref(),
        std::thread::available_parallelism()
            .map(|available_parallelism| available_parallelism.get())
            .unwrap_or(1),
    )
}

/// Forwards sccache as the CMake compiler launcher when Cargo itself is
/// wrapped by it.
///
/// Only sccache qualifies: it is compile-cache-transparent (same bytes).
/// Any other `RUSTC_WRAPPER` is ignored rather than forwarded, because an
/// arbitrary wrapper could rewrite compilations in ways the fingerprinted
/// toolchain never promised.
fn sccache_compiler_launcher_path() -> Option<PathBuf> {
    let rustc_wrapper_path = env::var_os(RUSTC_WRAPPER_VARIABLE).map(PathBuf::from)?;
    let wrapper_file_name = rustc_wrapper_path.file_name()?.to_str()?;
    (wrapper_file_name == SCCACHE_EXECUTABLE_NAME).then_some(rustc_wrapper_path)
}

fn cmake_boolean(boolean_value: bool) -> &'static str {
    if boolean_value { "ON" } else { "OFF" }
}

/// Runs one native operation with lifecycle progress and a timing-rich
/// failure message (the elapsed time distinguishes a fast configuration
/// error from a mid-compile one).
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
