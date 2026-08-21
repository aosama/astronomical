//! Cargo orchestration for Astronomical's pinned MLX and MLX-C runtime.
//!
//! Native products are compatibility-keyed outside `OUT_DIR` so release-version
//! changes do not rebuild MLX. Bindings remain in `OUT_DIR`, which preserves
//! Cargo's ownership of generated Rust source.

use std::{
    env,
    error::Error,
    fs::File,
    io::Write,
    path::{Path, PathBuf},
    process::Command,
    time::Instant,
};

mod build_bindings;
mod build_legacy_native_output;
mod build_native_linking;
mod build_native_store;

use build_bindings::generate_bindings;
use build_legacy_native_output::remove_legacy_cargo_native_build_directory;
use build_native_linking::configure_rust_linking;
use build_native_store::{NativeBuildArtifacts, NativeBuildProfile, NativeBuildStore};

const MLX_FEATURE_VARIABLE: &str = "CARGO_FEATURE_MLX";
const EXPERIMENTAL_ALIGNED_EXPERT_PACKS_FEATURE_VARIABLE: &str =
    "CARGO_FEATURE_EXPERIMENTAL_ALIGNED_EXPERT_PACKS";
const MLX_MEMORY_CONTRACT_PROBE_FEATURE_VARIABLE: &str = "CARGO_FEATURE_MLX_MEMORY_CONTRACT_PROBE";
const RUSTC_WRAPPER_VARIABLE: &str = "RUSTC_WRAPPER";
const NATIVE_DEPENDENCY_CACHE_VARIABLE: &str = "ASTRONOMICAL_NATIVE_DEPENDENCY_CACHE_DIR";
const NATIVE_BUILD_STORE_VARIABLE: &str = "ASTRONOMICAL_NATIVE_BUILD_STORE_DIR";
const NATIVE_BUILD_STATUS_FILE_VARIABLE: &str = "ASTRONOMICAL_NATIVE_BUILD_STATUS_FILE";
const SCCACHE_EXECUTABLE_NAME: &str = "sccache";
const NATIVE_ARCHIVE_VARIABLES: [&str; 5] = [
    "ASTRONOMICAL_MLX_SOURCE_ARCHIVE",
    "ASTRONOMICAL_MLX_C_SOURCE_ARCHIVE",
    "ASTRONOMICAL_METAL_CPP_SOURCE_ARCHIVE",
    "ASTRONOMICAL_JSON_SOURCE_ARCHIVE",
    "ASTRONOMICAL_FMT_SOURCE_ARCHIVE",
];

fn main() -> Result<(), Box<dyn Error>> {
    emit_environment_rerun_contracts();
    if env::var_os(MLX_FEATURE_VARIABLE).is_none() {
        return Ok(());
    }

    let manifest_directory = required_path_variable("CARGO_MANIFEST_DIR")?;
    let output_directory = required_path_variable("OUT_DIR")?;
    let repository_root = manifest_directory.join("../..").canonicalize()?;
    let native_source_directory = manifest_directory.join("native");
    let native_build_profile = selected_native_build_profile();
    let native_build_identity =
        resolve_native_build_identity(&repository_root, native_build_profile.identity_name())?;
    let native_build_store_directory = native_build_store_directory()?;
    let native_build_store = NativeBuildStore::new(
        &native_build_store_directory,
        &native_build_identity,
        native_build_profile,
    )?;
    let native_build_artifacts = native_build_store.resolve_or_build(|native_build_directory| {
        build_pinned_native_runtime(
            &manifest_directory,
            &native_source_directory,
            native_build_directory,
            native_build_profile,
        )
    })?;
    remove_legacy_cargo_native_build_directory(&output_directory)?;
    write_native_build_status(&native_build_artifacts)?;

    generate_bindings(
        &native_build_artifacts.include_directory(),
        &manifest_directory,
        &output_directory,
        native_build_profile.should_build_experimental_aligned_expert_packs(),
    )?;
    configure_rust_linking(&native_build_artifacts, native_build_profile)?;
    emit_native_source_rerun_contracts(&manifest_directory, &native_source_directory);
    Ok(())
}

fn emit_environment_rerun_contracts() {
    for environment_variable in [
        MLX_FEATURE_VARIABLE,
        EXPERIMENTAL_ALIGNED_EXPERT_PACKS_FEATURE_VARIABLE,
        MLX_MEMORY_CONTRACT_PROBE_FEATURE_VARIABLE,
        RUSTC_WRAPPER_VARIABLE,
        NATIVE_DEPENDENCY_CACHE_VARIABLE,
        NATIVE_BUILD_STORE_VARIABLE,
        NATIVE_BUILD_STATUS_FILE_VARIABLE,
    ] {
        println!("cargo:rerun-if-env-changed={environment_variable}");
    }
    for archive_variable in NATIVE_ARCHIVE_VARIABLES {
        println!("cargo:rerun-if-env-changed={archive_variable}");
    }
}

fn selected_native_build_profile() -> NativeBuildProfile {
    let should_build_memory_contract_probe =
        env::var_os(MLX_MEMORY_CONTRACT_PROBE_FEATURE_VARIABLE).is_some();
    let should_build_experimental_aligned_expert_packs =
        env::var_os(EXPERIMENTAL_ALIGNED_EXPERT_PACKS_FEATURE_VARIABLE).is_some();
    if !should_build_memory_contract_probe && !should_build_experimental_aligned_expert_packs {
        NativeBuildProfile::core()
    } else {
        NativeBuildProfile::new(
            should_build_memory_contract_probe,
            should_build_experimental_aligned_expert_packs,
        )
    }
}

fn resolve_native_build_identity(
    repository_root: &Path,
    native_build_profile: &str,
) -> Result<String, Box<dyn Error>> {
    let fingerprint_script_path = repository_root.join("scripts/native-build-cache-fingerprint.sh");
    let mut identity_command = Command::new(&fingerprint_script_path);
    identity_command
        .arg("--profile")
        .arg(native_build_profile)
        .arg(repository_root);
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

fn native_build_store_directory() -> Result<PathBuf, Box<dyn Error>> {
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

fn build_pinned_native_runtime(
    manifest_directory: &Path,
    native_source_directory: &Path,
    native_build_directory: &Path,
    native_build_profile: NativeBuildProfile,
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
        ))
        .arg(format!(
            "-DASTRONOMICAL_BUILD_EXPERIMENTAL_ALIGNED_EXPERT_PACKS={}",
            cmake_boolean(native_build_profile.should_build_experimental_aligned_expert_packs())
        ));
    append_native_archive_configuration(&mut configure_command)?;
    run_command(&mut configure_command, "native-configure")?;

    let mut native_build_command = Command::new("cmake");
    native_build_command
        .arg("--build")
        .arg(native_build_directory)
        .arg("--target")
        .arg("mlxc");
    if native_build_profile.should_build_experimental_aligned_expert_packs() {
        native_build_command.arg("astronomical_metal_expert_loader");
    }
    native_build_command
        .arg("--parallel")
        .arg(cargo_build_job_count());
    remove_uncontrolled_native_environment(&mut native_build_command);
    run_command(&mut native_build_command, "native-compile")?;

    if native_build_profile.should_build_memory_contract_probe() {
        let mut probe_build_command = Command::new("cmake");
        probe_build_command
            .arg("--build")
            .arg(native_build_directory)
            .arg("--target")
            .arg("mlx_memory_contract_probe")
            .arg("--parallel")
            .arg(cargo_build_job_count());
        remove_uncontrolled_native_environment(&mut probe_build_command);
        run_command(&mut probe_build_command, "native-memory-contract-probe")?;
    }
    println!(
        "cargo:rerun-if-changed={}",
        manifest_directory.join("build_native_store.rs").display()
    );
    Ok(())
}

fn append_native_archive_configuration(
    configure_command: &mut Command,
) -> Result<(), Box<dyn Error>> {
    if let Some(native_dependency_cache_directory) = native_dependency_cache_directory() {
        println!(
            "cargo:rerun-if-changed={}",
            native_dependency_cache_directory.display()
        );
        configure_command.arg(format!(
            "-D{NATIVE_DEPENDENCY_CACHE_VARIABLE}={}",
            native_dependency_cache_directory.display()
        ));
    }
    for archive_variable in NATIVE_ARCHIVE_VARIABLES {
        if let Some(archive_path) = env::var_os(archive_variable).map(PathBuf::from) {
            println!("cargo:rerun-if-changed={}", archive_path.display());
            configure_command.arg(format!("-D{archive_variable}={}", archive_path.display()));
        } else {
            configure_command.arg("-U").arg(archive_variable);
        }
    }
    Ok(())
}

fn write_native_build_status(
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

fn emit_native_source_rerun_contracts(manifest_directory: &Path, native_source_directory: &Path) {
    for source_path in [
        manifest_directory.join("build.rs"),
        manifest_directory.join("build_bindings.rs"),
        manifest_directory.join("build_native_linking.rs"),
        manifest_directory.join("build_native_store.rs"),
        manifest_directory.join("build_native_store_manifest.rs"),
        manifest_directory.join("native-build-store-schema-version"),
        native_source_directory.join("CMakeLists.txt"),
        native_source_directory.join("apply_patch_if_needed.cmake"),
        native_source_directory.join("tests/mlx_memory_contract_probe.cpp"),
        manifest_directory.join("../../scripts/native-build-cache-fingerprint.sh"),
        manifest_directory.join("../../third-party/native-dependency-manifest.cmake"),
        manifest_directory.join("../../third-party/pins"),
        manifest_directory.join("../../third-party/patches"),
    ] {
        println!("cargo:rerun-if-changed={}", source_path.display());
    }
}

fn required_path_variable(variable_name: &str) -> Result<PathBuf, Box<dyn Error>> {
    env::var_os(variable_name)
        .map(PathBuf::from)
        .ok_or_else(|| format!("required environment variable {variable_name} is missing").into())
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

fn cargo_build_job_count() -> String {
    env::var("CARGO_BUILD_JOBS")
        .or_else(|_| env::var("NUM_JOBS"))
        .unwrap_or_else(|_| "1".to_owned())
}

fn sccache_compiler_launcher_path() -> Option<PathBuf> {
    let rustc_wrapper_path = env::var_os(RUSTC_WRAPPER_VARIABLE).map(PathBuf::from)?;
    let wrapper_file_name = rustc_wrapper_path.file_name()?.to_str()?;
    (wrapper_file_name == SCCACHE_EXECUTABLE_NAME).then_some(rustc_wrapper_path)
}

fn cmake_boolean(boolean_value: bool) -> &'static str {
    if boolean_value { "ON" } else { "OFF" }
}

fn run_command(command: &mut Command, operation: &str) -> Result<(), Box<dyn Error>> {
    let operation_started_at = Instant::now();
    eprintln!("[native-build] operation={operation} status=start");
    let command_status = command.status()?;
    let elapsed_seconds = operation_started_at.elapsed().as_secs_f64();
    if !command_status.success() {
        return Err(format!(
            "native operation {operation} failed after {elapsed_seconds:.3} seconds: {command_status}"
        )
        .into());
    }
    eprintln!(
        "[native-build] operation={operation} status=success elapsed_seconds={elapsed_seconds:.3}"
    );
    Ok(())
}
