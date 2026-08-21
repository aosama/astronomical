//! Linker and runtime-resource metadata for a published native store entry.
//!
//! Keeping this separate from native compilation makes the Cargo owner small
//! while retaining one explicit inventory of Apple frameworks and archives.

use std::{
    error::Error,
    fs::File,
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
