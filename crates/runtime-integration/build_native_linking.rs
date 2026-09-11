//! Linker and runtime-resource metadata for a published native store entry.
//!
//! Keeping this separate from native compilation makes the Cargo owner small
//! while retaining one explicit inventory of Apple frameworks and archives.

use std::{
    env,
    error::Error,
    fs::{self, File},
    io::Read,
    path::{Path, PathBuf},
    process::Command,
};

use sha2::{Digest, Sha256};

use crate::build_native_store::{NativeBuildArtifacts, NativeBuildProfile};

pub fn configure_rust_linking(
    native_build_artifacts: &NativeBuildArtifacts,
    native_build_profile: NativeBuildProfile,
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
    if native_build_profile.should_build_experimental_aligned_expert_packs() {
        require_file(
            &native_library_directory.join("libastronomical_metal_expert_loader.a"),
            "experimental Astronomical Metal expert loader static library",
        )?;
        println!("cargo:rustc-link-lib=static=astronomical_metal_expert_loader");
    }
    let metallib_path = native_build_artifacts.metallib_path();
    require_file(&metallib_path, "MLX AOT metallib")?;
    let metallib_size_bytes = metallib_path.metadata()?.len();
    let metallib_sha256_hex = sha256_file_hex(&metallib_path)?;
    publish_metallib_beside_cargo_target(&metallib_path)?;

    println!(
        "cargo:rustc-link-search=native={}",
        native_library_directory.display()
    );
    println!("cargo:rustc-link-lib=static=mlxc");
    println!("cargo:rustc-link-lib=static=mlx");
    println!("cargo:rustc-link-lib=dylib=c++");
    for framework_name in [
        "Metal",
        "Foundation",
        "IOKit",
        "CoreFoundation",
        "QuartzCore",
        "Accelerate",
    ] {
        println!("cargo:rustc-link-lib=framework={framework_name}");
    }
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
    if let Some(memory_contract_probe_path) = native_build_artifacts.memory_contract_probe_path() {
        println!(
            "cargo:rustc-env=ASTRONOMICAL_MLX_MEMORY_CONTRACT_PROBE={}",
            memory_contract_probe_path.display()
        );
    }
    Ok(())
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

fn require_file(file_path: &Path, description: &str) -> Result<(), Box<dyn Error>> {
    if !file_path.is_file() {
        return Err(format!("missing {description} at {}", file_path.display()).into());
    }
    Ok(())
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
/// user cache directory that stores the compile-time fallback is routinely
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
