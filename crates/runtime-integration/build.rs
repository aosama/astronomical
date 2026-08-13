use std::{
    env,
    error::Error,
    fs::{self, File},
    io::Read,
    path::{Path, PathBuf},
    process::Command,
};

use sha2::{Digest, Sha256};

mod build_bindings;

use build_bindings::generate_bindings;

const MLX_FEATURE_VARIABLE: &str = "CARGO_FEATURE_MLX";
const EXPERIMENTAL_ALIGNED_EXPERT_PACKS_FEATURE_VARIABLE: &str =
    "CARGO_FEATURE_EXPERIMENTAL_ALIGNED_EXPERT_PACKS";
const MLX_MEMORY_CONTRACT_PROBE_FEATURE_VARIABLE: &str = "CARGO_FEATURE_MLX_MEMORY_CONTRACT_PROBE";
const RUSTC_WRAPPER_VARIABLE: &str = "RUSTC_WRAPPER";
const NATIVE_DEPENDENCY_CACHE_VARIABLE: &str = "ASTRONOMICAL_NATIVE_DEPENDENCY_CACHE_DIR";
const SCCACHE_EXECUTABLE_NAME: &str = "sccache";
const NATIVE_ARCHIVE_VARIABLES: [&str; 5] = [
    "ASTRONOMICAL_MLX_SOURCE_ARCHIVE",
    "ASTRONOMICAL_MLX_C_SOURCE_ARCHIVE",
    "ASTRONOMICAL_METAL_CPP_SOURCE_ARCHIVE",
    "ASTRONOMICAL_JSON_SOURCE_ARCHIVE",
    "ASTRONOMICAL_FMT_SOURCE_ARCHIVE",
];

fn main() -> Result<(), Box<dyn Error>> {
    println!("cargo:rerun-if-env-changed={MLX_FEATURE_VARIABLE}");
    println!("cargo:rerun-if-env-changed={EXPERIMENTAL_ALIGNED_EXPERT_PACKS_FEATURE_VARIABLE}");
    println!("cargo:rerun-if-env-changed={MLX_MEMORY_CONTRACT_PROBE_FEATURE_VARIABLE}");
    println!("cargo:rerun-if-env-changed={RUSTC_WRAPPER_VARIABLE}");
    println!("cargo:rerun-if-env-changed={NATIVE_DEPENDENCY_CACHE_VARIABLE}");
    for archive_variable in NATIVE_ARCHIVE_VARIABLES {
        println!("cargo:rerun-if-env-changed={archive_variable}");
    }
    if env::var_os(MLX_FEATURE_VARIABLE).is_none() {
        return Ok(());
    }

    let manifest_directory = required_path_variable("CARGO_MANIFEST_DIR")?;
    let output_directory = required_path_variable("OUT_DIR")?;
    let native_build_directory = output_directory.join("mlx-c-runtime-build");
    let native_source_directory = manifest_directory.join("native");
    let should_build_memory_contract_probe =
        env::var_os(MLX_MEMORY_CONTRACT_PROBE_FEATURE_VARIABLE).is_some();
    let should_build_experimental_aligned_expert_packs =
        env::var_os(EXPERIMENTAL_ALIGNED_EXPERT_PACKS_FEATURE_VARIABLE).is_some();

    let mut configure_command = Command::new("cmake");
    configure_command
        .arg("-S")
        .arg(&native_source_directory)
        .arg("-B")
        .arg(&native_build_directory)
        .arg("-DCMAKE_BUILD_TYPE=Release");
    let compiler_launcher_path = sccache_compiler_launcher_path();
    let compiler_launcher_argument = compiler_launcher_path
        .as_deref()
        .map_or_else(String::new, |launcher_path| {
            launcher_path.display().to_string()
        });
    configure_command
        .arg(format!(
            "-DCMAKE_C_COMPILER_LAUNCHER={compiler_launcher_argument}"
        ))
        .arg(format!(
            "-DCMAKE_CXX_COMPILER_LAUNCHER={compiler_launcher_argument}"
        ));
    if should_build_memory_contract_probe {
        configure_command.arg("-DASTRONOMICAL_BUILD_MEMORY_CONTRACT_PROBE=ON");
    } else {
        configure_command.arg("-DASTRONOMICAL_BUILD_MEMORY_CONTRACT_PROBE=OFF");
    }
    if should_build_experimental_aligned_expert_packs {
        configure_command.arg("-DASTRONOMICAL_BUILD_EXPERIMENTAL_ALIGNED_EXPERT_PACKS=ON");
    } else {
        configure_command.arg("-DASTRONOMICAL_BUILD_EXPERIMENTAL_ALIGNED_EXPERT_PACKS=OFF");
    }
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
    run_command(&mut configure_command, "configure the pinned MLX C runtime")?;

    let mut native_build_command = Command::new("cmake");
    native_build_command
        .arg("--build")
        .arg(&native_build_directory)
        .arg("--target")
        .arg("mlxc");
    if should_build_experimental_aligned_expert_packs {
        native_build_command.arg("astronomical_metal_expert_loader");
    }
    native_build_command
        .arg("--parallel")
        .arg(cargo_build_job_count());
    run_command(&mut native_build_command, "build the pinned MLX C runtime")?;

    let memory_contract_probe_path = if should_build_memory_contract_probe {
        let mut probe_build_command = Command::new("cmake");
        probe_build_command
            .arg("--build")
            .arg(&native_build_directory)
            .arg("--target")
            .arg("mlx_memory_contract_probe")
            .arg("--parallel")
            .arg(cargo_build_job_count());
        run_command(
            &mut probe_build_command,
            "build the pinned MLX memory contract probe",
        )?;
        let probe_path = native_build_directory
            .join("bin")
            .join("mlx_memory_contract_probe");
        require_file(&probe_path, "MLX memory contract probe")?;
        Some(probe_path)
    } else {
        None
    };

    let mlx_c_source_directory = native_build_directory.join("_deps/mlx_c-src");
    generate_bindings(
        &mlx_c_source_directory,
        &manifest_directory,
        &output_directory,
        should_build_experimental_aligned_expert_packs,
    )?;

    let native_library_directory = native_build_directory.join("lib");
    let experimental_native_library_path =
        native_library_directory.join("libastronomical_metal_expert_loader.a");
    let experimental_native_test_path = native_build_directory
        .join("bin")
        .join("astronomical_metal_expert_loader_native_tests");
    if !should_build_experimental_aligned_expert_packs {
        remove_stale_experimental_artifact(&experimental_native_library_path)?;
        remove_stale_experimental_artifact(&experimental_native_test_path)?;
    }
    require_file(
        &native_library_directory.join("libmlx.a"),
        "MLX static library",
    )?;
    require_file(
        &native_library_directory.join("libmlxc.a"),
        "MLX C static library",
    )?;
    if should_build_experimental_aligned_expert_packs {
        require_file(
            &experimental_native_library_path,
            "experimental Astronomical Metal expert loader static library",
        )?;
    }
    let metallib_path =
        native_build_directory.join("_deps/mlx-build/mlx/backend/metal/kernels/mlx.metallib");
    require_file(&metallib_path, "MLX AOT metallib")?;
    let metallib_size_bytes = metallib_path.metadata()?.len();
    let metallib_sha256_hex = sha256_file_hex(&metallib_path)?;

    println!(
        "cargo:rustc-link-search=native={}",
        native_library_directory.display()
    );
    if should_build_experimental_aligned_expert_packs {
        println!("cargo:rustc-link-lib=static=astronomical_metal_expert_loader");
    }
    println!("cargo:rustc-link-lib=static=mlxc");
    println!("cargo:rustc-link-lib=static=mlx");
    println!("cargo:rustc-link-lib=dylib=c++");
    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-link-lib=framework=IOKit");
    println!("cargo:rustc-link-lib=framework=CoreFoundation");
    println!("cargo:rustc-link-lib=framework=QuartzCore");
    println!("cargo:rustc-link-lib=framework=Accelerate");
    let clang_runtime_directory = discover_clang_runtime_directory()?;
    require_file(
        &clang_runtime_directory.join("libclang_rt.osx.a"),
        "Clang macOS runtime archive",
    )?;
    println!(
        "cargo:rustc-link-search=native={}",
        clang_runtime_directory.display()
    );
    println!("cargo:rustc-link-lib=static=clang_rt.osx");
    println!(
        "cargo:rustc-env=ASTRONOMICAL_MLX_METALLIB_PATH={}",
        metallib_path.display()
    );
    println!("cargo:rustc-env=ASTRONOMICAL_MLX_METALLIB_SIZE_BYTES={metallib_size_bytes}");
    println!("cargo:rustc-env=ASTRONOMICAL_MLX_METALLIB_SHA256={metallib_sha256_hex}");
    if let Some(memory_contract_probe_path) = memory_contract_probe_path {
        println!(
            "cargo:rustc-env=ASTRONOMICAL_MLX_MEMORY_CONTRACT_PROBE={}",
            memory_contract_probe_path.display()
        );
    }
    println!(
        "cargo:rerun-if-changed={}",
        native_source_directory.join("CMakeLists.txt").display()
    );
    if should_build_experimental_aligned_expert_packs {
        println!(
            "cargo:rerun-if-changed={}",
            native_source_directory
                .join("experimental/aligned_expert_packs/astronomical_metal_expert_loader.h")
                .display()
        );
        println!(
            "cargo:rerun-if-changed={}",
            native_source_directory
                .join("experimental/aligned_expert_packs/astronomical_metal_expert_loader.cpp")
                .display()
        );
    }
    println!(
        "cargo:rerun-if-changed={}",
        native_source_directory
            .join("tests/mlx_memory_contract_probe.cpp")
            .display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        native_source_directory
            .join("apply_patch_if_needed.cmake")
            .display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        manifest_directory
            .join("../../third-party/pins/mlx-v0.32.0.cmake")
            .display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        manifest_directory
            .join("../../third-party/pins/mlx-c-v0.6.0.cmake")
            .display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        manifest_directory
            .join("../../third-party/patches/mlx-0.32.0-nax-head-dim-256-attention.patch")
            .display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        manifest_directory
            .join("../../third-party/patches/mlx-0.32.0-stable-prefill-split-k.patch")
            .display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        manifest_directory
            .join("../../third-party/patches/mlx-0.32.0-astronomical-active-memory-ceiling.patch")
            .display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        manifest_directory
            .join("../../third-party/patches/mlx-0.32.0-adaptive-io-thread-pool.patch")
            .display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        manifest_directory
            .join("../../third-party/patches/mlx-0.32.0-streaming-safetensors-writer.patch")
            .display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        manifest_directory
            .join("../../third-party/patches/mlx-c-0.6.0-metallib-path.patch")
            .display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        manifest_directory
            .join("../../third-party/patches/mlx-c-0.6.0-mlx-0.32.0-fft-compatibility.patch")
            .display()
    );

    Ok(())
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

fn discover_clang_runtime_directory() -> Result<PathBuf, Box<dyn Error>> {
    let clang_runtime_output = Command::new("xcrun")
        .args(["clang", "--print-runtime-dir"])
        .output()?;
    if !clang_runtime_output.status.success() {
        return Err("xcrun clang could not report its runtime directory".into());
    }
    let clang_runtime_text = String::from_utf8(clang_runtime_output.stdout)?;
    let clang_runtime_directory = PathBuf::from(clang_runtime_text.trim());
    if !clang_runtime_directory.is_absolute() {
        return Err("xcrun clang reported a non-absolute runtime directory".into());
    }
    Ok(clang_runtime_directory)
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

fn run_command(command: &mut Command, operation: &str) -> Result<(), Box<dyn Error>> {
    let command_status = command.status()?;
    if !command_status.success() {
        return Err(format!("failed to {operation}: command exited with {command_status}").into());
    }
    Ok(())
}

fn require_file(file_path: &Path, description: &str) -> Result<(), Box<dyn Error>> {
    if !file_path.is_file() {
        return Err(format!("missing {description} at {}", file_path.display()).into());
    }
    Ok(())
}

fn remove_stale_experimental_artifact(artifact_path: &Path) -> Result<(), Box<dyn Error>> {
    match fs::remove_file(artifact_path) {
        Ok(()) => Ok(()),
        Err(remove_error) if remove_error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(remove_error) => Err(remove_error.into()),
    }
}

fn sha256_file_hex(file_path: &Path) -> Result<String, Box<dyn Error>> {
    let mut source_file = File::open(file_path)?;
    let mut digest = Sha256::new();
    let mut digest_buffer = [0_u8; 64 * 1024];
    loop {
        let bytes_read = source_file.read(&mut digest_buffer)?;
        if bytes_read == 0 {
            break;
        }
        digest.update(&digest_buffer[..bytes_read]);
    }
    Ok(hex_digest_bytes(&digest.finalize()))
}

fn hex_digest_bytes(digest_bytes: &[u8]) -> String {
    digest_bytes
        .iter()
        .map(|digest_byte| format!("{digest_byte:02x}"))
        .collect()
}
