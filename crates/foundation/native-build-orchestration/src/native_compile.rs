//! Pinned native runtime compilation shared by the Cargo build script and the
//! standalone native-build tool.
//!
//! # One module, two entry points
//!
//! The build script consumes this crate through `[build-dependencies]` and
//! the standalone tool through a normal dependency, so this module must stay
//! free of `cargo:` directives (it must remain callable outside a build
//! script) and reaches its sibling build modules through ordinary `crate::`
//! paths. Like the other build orchestration sources, this file is excluded
//! from the native identity fingerprint: semantic CMake changes require
//! bumping `native-build-store-schema-version` instead — progress-logging or
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
//! filenames under one `astro_native_build_cache` dependency directory. CMake resolves every
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

use crate::native_host_platform::{
    MLX_PACKAGE_INSTALL_RELATIVE_DIRECTORY, MLX_SUBPROJECT_BUILD_RELATIVE_DIRECTORY,
    NativeTargetPlatform, compiler_configure_arguments, installs_mlx_package,
};
use crate::native_parallelism;
use crate::native_progress::NativeBuildProgress;
use crate::native_store::{NativeBuildArtifacts, NativeBuildProfile};

/// Consulted so the CMake compile can share sccache with Cargo without any
/// other wrapper silently changing compiler semantics.
pub const RUSTC_WRAPPER_VARIABLE: &str = "RUSTC_WRAPPER";
pub const ASTRO_NATIVE_BUILD_CACHE_DEPENDENCIES_VARIABLE: &str =
    "ASTRONOMICAL_NATIVE_DEPENDENCY_CACHE_DIR";
pub const NATIVE_BUILD_STORE_VARIABLE: &str = "ASTRONOMICAL_NATIVE_BUILD_STORE_DIR";
pub const NATIVE_BUILD_STATUS_FILE_VARIABLE: &str = "ASTRONOMICAL_NATIVE_BUILD_STATUS_FILE";
const SCCACHE_EXECUTABLE_NAME: &str = "sccache";

/// Native inputs resolved from the environment once per run so the CMake
/// configuration and Cargo's path tracking observe the same values.
pub struct NativeRuntimeBuildInputs {
    astro_native_build_cache_dependencies_directory: Option<PathBuf>,
}

impl NativeRuntimeBuildInputs {
    pub fn from_environment() -> Self {
        Self {
            astro_native_build_cache_dependencies_directory:
                astro_native_build_cache_dependencies_directory(),
        }
    }

    pub fn astro_native_build_cache_dependencies_directory(&self) -> Option<&Path> {
        self.astro_native_build_cache_dependencies_directory
            .as_deref()
    }
}

/// Resolves the native build identity for a build of `native_build_profile`
/// on `target_triple`.
///
/// The identity library lives in `crates/foundation/native-identity`; this wrapper is
/// the single choke point both build paths (Cargo build script and
/// standalone tool) share, which is what guarantees they key the *same*
/// store entry. Overrides stay absent by construction: a production build
/// must always fingerprint the toolchain that CMake actually uses.
pub fn resolve_native_build_identity(
    repository_root: &Path,
    native_build_profile: &str,
    target_triple: &str,
) -> Result<String, Box<dyn Error>> {
    let profile = native_identity::NativeBuildProfile::from_profile_text(native_build_profile)
        .map_err(|error| error.to_string())?;
    // The standalone native-build tool states its triple here because it has
    // no Cargo context; the identity stays identical in both paths.
    let identity_request = native_identity::FingerprintRequest {
        repository_root,
        profile,
        scope: native_identity::IdentityScope::Full,
        overrides: native_identity::probes::ToolchainOverrides {
            target_triple: Some(target_triple.to_owned()),
            ..native_identity::probes::ToolchainOverrides::none()
        },
    };
    native_identity::native_build_identity(&identity_request)
        .map_err(|error| format!("native build identity failed: {error}").into())
}

/// Selects the machine-wide store root.
///
/// An explicit `ASTRONOMICAL_NATIVE_BUILD_STORE_DIR` wins; the default lives
/// under the `astro_native_build_cache` folder so multiple checkouts and worktrees share
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
/// Sequence: configure (the platform's compilers, plus the SDK and arm64 on
/// macOS, Release, sanitized environment, archive locations, optional probe
/// target) → build the `mlxc` aggregate target → optionally build the
/// memory-contract probe binary → on Linux, install MLX's own CMake package,
/// whose exported targets the link reads. Every command reports through the progress recorder so hosted CI
/// can watch the silent multi-minute stretches.
pub fn build_pinned_native_runtime(
    native_source_directory: &Path,
    native_build_directory: &Path,
    native_target_platform: NativeTargetPlatform,
    native_build_profile: NativeBuildProfile,
    native_build_inputs: &NativeRuntimeBuildInputs,
    native_build_progress: &NativeBuildProgress,
) -> Result<(), Box<dyn Error>> {
    let mut configure_command = Command::new("cmake");
    configure_command
        .arg("-G")
        .arg("Unix Makefiles")
        .arg("-S")
        .arg(native_source_directory)
        .arg("-B")
        .arg(native_build_directory)
        .arg("-DCMAKE_BUILD_TYPE=Release")
        .args(compiler_configure_arguments(native_target_platform)?);
    remove_uncontrolled_native_environment(&mut configure_command);
    configure_command
        .args(compiler_launcher_configure_arguments(
            native_target_platform,
        ))
        .arg(format!(
            "-DASTRONOMICAL_BUILD_MEMORY_CONTRACT_PROBE={}",
            cmake_boolean(native_build_profile.should_build_memory_contract_probe())
        ));
    if let Some(astro_native_build_cache_dependencies_directory) =
        native_build_inputs.astro_native_build_cache_dependencies_directory()
    {
        configure_command.arg(format!(
            "-D{ASTRO_NATIVE_BUILD_CACHE_DEPENDENCIES_VARIABLE}={}",
            astro_native_build_cache_dependencies_directory.display()
        ));
    } else {
        configure_command
            .arg("-U")
            .arg(ASTRO_NATIVE_BUILD_CACHE_DEPENDENCIES_VARIABLE);
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

    build_profile_and_platform_extras(
        native_build_directory,
        native_target_platform,
        native_build_profile,
        native_build_progress,
    )
}

/// The steps beyond the `mlxc` target that a profile or platform adds: the
/// memory-contract probe binary, and on Linux the install of MLX's own CMake
/// package, whose exported targets the link reads.
fn build_profile_and_platform_extras(
    native_build_directory: &Path,
    native_target_platform: NativeTargetPlatform,
    native_build_profile: NativeBuildProfile,
    native_build_progress: &NativeBuildProgress,
) -> Result<(), Box<dyn Error>> {
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
    if installs_mlx_package(native_target_platform) {
        let mut package_install_command = Command::new("cmake");
        package_install_command
            .arg("--install")
            .arg(native_build_directory.join(MLX_SUBPROJECT_BUILD_RELATIVE_DIRECTORY))
            .arg("--prefix")
            .arg(native_build_directory.join(MLX_PACKAGE_INSTALL_RELATIVE_DIRECTORY));
        remove_uncontrolled_native_environment(&mut package_install_command);
        // An ambient staging root or install mode would move the package out
        // of the build directory or publish links instead of files.
        package_install_command
            .env_remove("DESTDIR")
            .env_remove("CMAKE_INSTALL_MODE");
        run_command(
            &mut package_install_command,
            "native-mlx-package-install",
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

/// Default location of the provisioned dependencies in the
/// `astro_native_build_cache`, shared by every checkout on the machine.
fn astro_native_build_cache_dependencies_directory() -> Option<PathBuf> {
    env::var_os(ASTRO_NATIVE_BUILD_CACHE_DEPENDENCIES_VARIABLE)
        .map(PathBuf::from)
        .or_else(|| {
            env::var_os("HOME")
                .map(PathBuf::from)
                .map(|home_directory| {
                    home_directory.join("Library/Caches/Astronomical/native-dependencies")
                })
        })
}

/// The compiler-launcher arguments: sccache, when it is installed, wraps
/// every compiler the recipe uses; an empty launcher otherwise, so a stale
/// launcher in a reused build tree is cleared.
///
/// Only the Linux CUDA recipe has a CUDA compiler to wrap; elsewhere CMake
/// would warn that the variable went unused. sccache wraps nvcc as it wraps
/// cc.
fn compiler_launcher_configure_arguments(
    native_target_platform: NativeTargetPlatform,
) -> Vec<String> {
    let compiler_launcher_argument = match sccache_compiler_launcher_path() {
        Some(launcher_path) => launcher_path.display().to_string(),
        None => String::new(),
    };
    let mut launcher_arguments = vec![
        format!("-DCMAKE_C_COMPILER_LAUNCHER={compiler_launcher_argument}"),
        format!("-DCMAKE_CXX_COMPILER_LAUNCHER={compiler_launcher_argument}"),
    ];
    if native_target_platform == NativeTargetPlatform::LinuxCuda {
        launcher_arguments.push(format!(
            "-DCMAKE_CUDA_COMPILER_LAUNCHER={compiler_launcher_argument}"
        ));
    }
    launcher_arguments
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
        // CMake seeds its CUDA flags, host compiler and architectures from
        // these, and nvcc reads its NVCC_* variables itself, so each would
        // change the CUDA recipe's kernels (`-G` debug code, say) under an
        // unchanged identity.
        "CUDAARCHS",
        "CUDAFLAGS",
        "CUDAHOSTCXX",
        "CXX",
        "CXXFLAGS",
        "LDFLAGS",
        "MACOSX_DEPLOYMENT_TARGET",
        "NVCC_APPEND_FLAGS",
        "NVCC_CCBIN",
        "NVCC_PREPEND_FLAGS",
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
/// Only sccache is accepted: it never changes the compiled bytes, only
/// whether they come from its store.
/// Any other `RUSTC_WRAPPER` is ignored rather than forwarded, because an
/// arbitrary wrapper could rewrite compilations in ways the fingerprinted
/// toolchain never promised.
fn sccache_compiler_launcher_path() -> Option<PathBuf> {
    let rustc_wrapper_path = env::var_os(RUSTC_WRAPPER_VARIABLE).map(PathBuf::from)?;
    let wrapper_file_name = rustc_wrapper_path.file_name()?.to_str()?;
    (wrapper_file_name == SCCACHE_EXECUTABLE_NAME).then_some(rustc_wrapper_path)
}

/// Starts the sccache server, when sccache is the compiler launcher, before
/// a build slot is taken. The slot lock stays open in every compiler this
/// build starts, so a server first started by one of those compiles would
/// inherit it and keep the slot busy for its whole lifetime, moving the next
/// build to another slot path where sccache keys miss. Starting it first
/// means the server never holds a slot. A failure is ignored: the compile
/// then starts the server itself, as before.
pub(crate) fn start_sccache_server_before_taking_a_slot() {
    let Some(sccache_path) = sccache_compiler_launcher_path() else {
        return;
    };
    let _ignored_status = Command::new(sccache_path)
        .arg("--start-server")
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status();
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
