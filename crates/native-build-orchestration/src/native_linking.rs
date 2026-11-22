//! Linker and runtime-resource metadata for a published native store entry.
//!
//! Keeping this separate from native compilation makes the Cargo owner small.
//! The link lines themselves are composed in `native_link_directives`, where
//! each platform's line is tested without a native build; this module checks
//! the store entry's products and prints those lines.
//!
//! On macOS the line is the two static archives, the C++ runtime, the six
//! Apple frameworks and the Clang compiler-rt archive. On Linux it is the two
//! static archives, the libraries MLX's configure chose (read from MLX's
//! installed `MLXTargets.cmake`), and the GNU C++ runtime.
//!
//! # The Metallib trust chain (macOS)
//!
//! The build-produced Metallib is hashed here at build time and injected as
//! three compile-time environment values
//! (`ASTRONOMICAL_MLX_METALLIB_{PATH,SIZE_BYTES,SHA256}`). The runtime's
//! validation (`src/mlx_runtime/metallib.rs`) compares every candidate at
//! process start against those values — a closed loop: the bytes the build
//! produced are the only bytes the process will load.
//!
//! The same function also copies the Metallib beside the cargo target
//! profile directory (see [`publish_metallib_beside_cargo_target`]) so an
//! unpackaged `cargo run` resolves a kernel archive without an environment
//! override.

use std::{
    env,
    error::Error,
    fs::{self, File},
    io::Read,
    path::{Path, PathBuf},
    process::Command,
};

use sha2::{Digest, Sha256};

use crate::native_host_platform::NativeTargetPlatform;
use crate::native_link_directives::{linux_link_directives, macos_link_directive_lines};
use crate::native_store::NativeBuildArtifacts;

/// Emits every `cargo:` directive needed to link and run against a store entry.
///
/// # Errors
///
/// Fails when any required product is missing from the published entry (a
/// corrupt store that slipped past validation) or, on macOS, when the Clang
/// runtime directory cannot be resolved.
pub fn configure_rust_linking(
    native_build_artifacts: &NativeBuildArtifacts,
    native_target_platform: NativeTargetPlatform,
) -> Result<(), Box<dyn Error>> {
    let native_library_directory = native_build_artifacts.native_library_directory();
    require_file(
        &native_build_artifacts.mlx_library_path(),
        "MLX static library",
    )?;
    require_file(
        &native_library_directory.join("libmlxc.a"),
        "MLX C static library",
    )?;
    match native_target_platform {
        NativeTargetPlatform::MacOs => {
            configure_macos_rust_linking(native_build_artifacts, &native_library_directory)?;
        }
        NativeTargetPlatform::Linux => {
            configure_linux_rust_linking(native_build_artifacts, &native_library_directory)?;
        }
    }
    if let Some(memory_contract_probe_path) = native_build_artifacts.memory_contract_probe_path() {
        println!(
            "cargo:rustc-env=ASTRONOMICAL_MLX_MEMORY_CONTRACT_PROBE={}",
            memory_contract_probe_path.display()
        );
    }
    Ok(())
}

/// Links the Metal build and hands the runtime its verified Metallib.
fn configure_macos_rust_linking(
    native_build_artifacts: &NativeBuildArtifacts,
    native_library_directory: &Path,
) -> Result<(), Box<dyn Error>> {
    let metallib_path = native_build_artifacts.metallib_path();
    require_file(&metallib_path, "MLX AOT metallib")?;
    let metallib_size_bytes = metallib_path.metadata()?.len();
    let metallib_sha256_hex = sha256_file_hex(&metallib_path)?;
    publish_metallib_beside_cargo_target(&metallib_path)?;

    let clang_runtime_directory = discover_clang_runtime_directory()?;
    require_file(
        &clang_runtime_directory.join("libclang_rt.osx.a"),
        "Clang macOS runtime archive",
    )?;
    for cargo_directive_line in
        macos_link_directive_lines(native_library_directory, &clang_runtime_directory)
    {
        println!("{cargo_directive_line}");
    }
    println!(
        "cargo:rustc-env=ASTRONOMICAL_MLX_METALLIB_PATH={}",
        metallib_path.display()
    );
    println!("cargo:rustc-env=ASTRONOMICAL_MLX_METALLIB_SIZE_BYTES={metallib_size_bytes}");
    println!("cargo:rustc-env=ASTRONOMICAL_MLX_METALLIB_SHA256={metallib_sha256_hex}");
    Ok(())
}

/// Links the CPU build against the libraries MLX was configured with, as its
/// installed CMake package records them. There is no Metallib on Linux, so no
/// Metallib values are emitted.
fn configure_linux_rust_linking(
    native_build_artifacts: &NativeBuildArtifacts,
    native_library_directory: &Path,
) -> Result<(), Box<dyn Error>> {
    let mlx_targets_path = native_build_artifacts.mlx_targets_path();
    require_file(&mlx_targets_path, "MLX CMake package targets")?;
    let mlx_targets_text = fs::read_to_string(&mlx_targets_path)?;
    let link_directives = linux_link_directives(native_library_directory, &mlx_targets_text);
    // The store entry is keyed by the BLAS search settings, not by the
    // library files themselves; a file removed since the entry was built
    // would otherwise surface as an obscure linker error.
    for linked_library_file in &link_directives.linked_library_files {
        if !linked_library_file.is_file() {
            return Err(format!(
                "the native runtime was configured against {}, which no longer exists; reinstall it or point CMake at another BLAS so a new native runtime is built",
                linked_library_file.display()
            )
            .into());
        }
    }
    for warning_message in &link_directives.warning_messages {
        println!("cargo:warning={warning_message}");
    }
    for cargo_directive_line in &link_directives.cargo_directive_lines {
        println!("{cargo_directive_line}");
    }
    Ok(())
}

/// Locates the Clang runtime archive directory through the active toolchain.
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

fn require_file(file_path: &Path, description: &str) -> Result<(), Box<dyn Error>> {
    if !file_path.is_file() {
        return Err(format!("missing {description} at {}", file_path.display()).into());
    }
    Ok(())
}

/// Streams a file through SHA-256 with a fixed buffer; the Metallib is tens
/// of megabytes and must not be slurped whole to be hashed.
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
    Ok(digest
        .finalize()
        .iter()
        .map(|digest_byte| format!("{digest_byte:02x}"))
        .collect())
}

/// Publishes the build-produced metallib beside the cargo target directory.
///
/// An unpackaged worker resolves its bundled metallib at `../Resources` relative
/// to its own directory, and a cargo-built worker lives under
/// `<target>/<profile>/`, so `OUT_DIR` (`<target>/<profile>/build/<pkg>-<hash>`)
/// up two directories is the profile directory that owns the executable. The
/// `~/Library/Caches` folder that stores the compile-time fallback is routinely
/// purged by third-party cleanup tools (issue #521); the cargo target directory
/// is workspace-local and survives those purges, so the packaged-app resolution
/// chain keeps working for unpackaged cargo runs without an environment
/// override.
fn publish_metallib_beside_cargo_target(metallib_path: &Path) -> Result<(), Box<dyn Error>> {
    let Some(out_directory) = env::var_os("OUT_DIR").map(PathBuf::from) else {
        return Err("cargo did not report OUT_DIR for the native build".into());
    };
    let Some(profile_directory) = out_directory
        .parent()
        .and_then(|package_build_directory| package_build_directory.parent())
        .and_then(|build_directory| build_directory.parent())
    else {
        return Err(format!(
            "OUT_DIR is not inside a cargo profile directory: {}",
            out_directory.display()
        )
        .into());
    };
    let published_metallib_path = profile_directory
        .join("../Resources/share/mlx/mlx.metallib")
        .components()
        .collect::<PathBuf>();
    if let Some(published_parent_directory) = published_metallib_path.parent() {
        fs::create_dir_all(published_parent_directory)?;
    }
    // Write-then-rename keeps concurrent cargo invocations from observing a
    // partially written metallib.
    let staged_metallib_path = published_metallib_path.with_extension("metallib.staged");
    fs::copy(metallib_path, &staged_metallib_path)?;
    fs::rename(&staged_metallib_path, &published_metallib_path)?;
    Ok(())
}
