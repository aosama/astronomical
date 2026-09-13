//! Compiles the Core ML predictor loader. Independent of MLX so the Neural
//! Engine path never enters the graphics-processor build graph.

use std::env;
use std::error::Error;
use std::path::Path;
use std::process::Command;

pub(crate) fn compile_macos_predictor_ane() -> Result<(), Box<dyn Error>> {
    if env::var("CARGO_CFG_TARGET_OS").ok().as_deref() != Some("macos") {
        return Ok(());
    }
    let manifest_directory = env::var("CARGO_MANIFEST_DIR")?;
    let output_directory = env::var("OUT_DIR")?;
    let source_path = Path::new(&manifest_directory).join("native/predictor_ane.m");
    println!("cargo:rerun-if-changed={}", source_path.display());
    let object_path = Path::new(&output_directory).join("predictor_ane.o");
    let archive_path = Path::new(&output_directory).join("libastronomical_predictor_ane.a");
    let compile_status = Command::new("clang")
        .args(["-c", "-fobjc-arc", "-fmodules", "-o"])
        .arg(&object_path)
        .arg(&source_path)
        .status()?;
    if !compile_status.success() {
        return Err("clang failed to compile native/predictor_ane.m".into());
    }
    let archive_status = Command::new("ar")
        .args(["crus"])
        .arg(&archive_path)
        .arg(&object_path)
        .status()?;
    if !archive_status.success() {
        return Err("ar failed to archive predictor_ane.o".into());
    }
    println!("cargo:rustc-link-search=native={output_directory}");
    println!("cargo:rustc-link-lib=static=astronomical_predictor_ane");
    println!("cargo:rustc-link-lib=framework=CoreML");
    println!("cargo:rustc-link-lib=framework=Foundation");
    Ok(())
}
