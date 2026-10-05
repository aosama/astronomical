//! Bindgen generation for the official MLX C API, from provisioned headers.
//!
//! The header include root comes from `scripts/provision-bindgen-headers.sh`,
//! keyed by the source-only native build identity. This build script never
//! invokes CMake and never builds native code: it only translates the already
//! extracted and patched headers into Rust declarations. The MLX and MLX-C
//! native libraries are linked by `astronomical-runtime-integration`'s build
//! script, so every final binary that contains this crate's objects also
//! contains the native image.

use std::{
    env,
    error::Error,
    path::{Path, PathBuf},
    process::Command,
};

const NATIVE_DEPENDENCY_CACHE_VARIABLE: &str = "ASTRONOMICAL_NATIVE_DEPENDENCY_CACHE_DIR";
const DEFAULT_NATIVE_DEPENDENCY_CACHE_SUFFIX: &str =
    "Library/Caches/Astronomical/native-dependencies";
const EXTRACTION_TREE_MLX_C: &str = "mlx_c-src";
const COMPLETION_MARKER_FILE_NAME: &str = "complete";

const BINDGEN_FUNCTION_ALLOWLIST: &str =
    "mlx_.*|astronomical_metal_expert_loader_(start|wait|free)";

const BINDGEN_TYPE_ALLOWLIST: &str =
    "mlx_.*|astronomical_metal_expert_loader_(output_tensor|load_range|metrics|handle)";

const EXPERT_LOADER_HEADER_REPOSITORY_RELATIVE_PATH: &str = "crates/runtime-integration/native/experimental/aligned_expert_packs/astronomical_metal_expert_loader.h";

/// The generated inventory module consumed by `raw.rs`; it exposes the exact
/// bridged symbol sets the coverage contract compares against the pinned
/// headers.
const BRIDGED_INVENTORY_FILE_NAME: &str = "bridged_inventory.rs";

fn main() -> Result<(), Box<dyn Error>> {
    let manifest_directory = required_path_variable("CARGO_MANIFEST_DIR")?;
    let output_directory = required_path_variable("OUT_DIR")?;
    let repository_root = manifest_directory.join("../..").canonicalize()?;
    println!("cargo:rerun-if-env-changed={NATIVE_DEPENDENCY_CACHE_VARIABLE}");
    emit_rerun_contracts(&repository_root);
    let headers_root = resolve_headers_root(&repository_root)?;
    let include_directory = headers_root.join(EXTRACTION_TREE_MLX_C);
    let mlx_c_header = include_directory.join("mlx/c/mlx.h");
    if !mlx_c_header.is_file() {
        return Err(format!(
            "missing MLX C umbrella header at {}; run scripts/provision-bindgen-headers.sh (see third-party/README.md)",
            mlx_c_header.display()
        )
        .into());
    }

    println!("cargo:rerun-if-changed={}", headers_root.display());

    // The Astronomical Metal expert-loader header is part of the same native
    // image and its functions take MLX C types, so it must be bound in this
    // same bindgen unit: C types declared in two bindgen outputs are nominal
    // Rust types that do not unify. Its symbols are referenced only by the
    // experimental loader, so profiles that exclude it never demand them.
    let expert_loader_header = repository_root.join(EXPERT_LOADER_HEADER_REPOSITORY_RELATIVE_PATH);
    if !expert_loader_header.is_file() {
        return Err(format!(
            "missing Astronomical Metal expert loader header at {}",
            expert_loader_header.display()
        )
        .into());
    }
    println!("cargo:rerun-if-changed={}", expert_loader_header.display());

    let bindings = bindgen::Builder::default()
        .header(mlx_c_header.to_string_lossy())
        .header(expert_loader_header.to_string_lossy())
        .clang_arg(format!("-I{}", include_directory.display()))
        .clang_arg(format!(
            "-I{}",
            expert_loader_header
                .parent()
                .ok_or("Astronomical Metal expert loader header has no parent directory")?
                .display()
        ))
        .allowlist_function(BINDGEN_FUNCTION_ALLOWLIST)
        .allowlist_type(BINDGEN_TYPE_ALLOWLIST)
        .generate_comments(false)
        .generate()?;
    let bindings_content = bindings.to_string();
    write_bridged_inventory(&bindings_content, &output_directory)?;
    bindings.write_to_file(output_directory.join("mlx_c_bindings.rs"))?;
    Ok(())
}

/// Extracts the bridged `mlx_*` functions and types from the generated
/// bindings and emits them as public constant slices so the coverage contract
/// can compare the bridge against the pinned headers without re-running
/// bindgen.
fn write_bridged_inventory(
    bindings_content: &str,
    output_directory: &Path,
) -> Result<(), Box<dyn Error>> {
    let identifier_end = |segment: &str| {
        segment
            .find(|character: char| !(character.is_ascii_alphanumeric() || character == '_'))
            .unwrap_or(segment.len())
    };
    let mut bridged_functions: Vec<String> = bindings_content
        .split("pub fn ")
        .skip(1)
        .map(|segment| &segment[..identifier_end(segment)])
        .filter(|name| name.starts_with("mlx_"))
        .map(str::to_owned)
        .collect();
    bridged_functions.sort();
    bridged_functions.dedup();

    // bindgen 0.73 skips C variadic functions; the hand-written declaration
    // in `raw.rs` restores the one variadic MLX-C entry point.
    bridged_functions.push("mlx_error".to_owned());

    let mut bridged_types: Vec<String> = ["pub struct ", "pub enum ", "pub union ", "pub type "]
        .iter()
        .flat_map(|marker| {
            bindings_content
                .split(marker)
                .skip(1)
                .map(|segment| &segment[..identifier_end(segment)])
                .filter(|name| name.starts_with("mlx_"))
                .map(str::to_owned)
                .collect::<Vec<String>>()
        })
        .collect();
    // bindgen aliases constified C enums through `pub use self::mlx_x_ as
    // mlx_x;`; the alias is the name the headers expose, so it belongs in the
    // inventory alongside the tag.
    for alias_segment in bindings_content.split("pub use self::").skip(1) {
        let Some((_tag, alias_with_rest)) = alias_segment.split_once(" as ") else {
            continue;
        };
        let alias = &alias_with_rest[..identifier_end(alias_with_rest)];
        if alias.starts_with("mlx_") {
            bridged_types.push(alias.to_owned());
        }
    }
    bridged_types.sort();
    bridged_types.dedup();

    let inventory_source = format!(
        "// Generated by this crate's build script from the bindgen output.\n\
         // Every listed symbol exists in the compiled bridge; the coverage\n\
         // contract compares this inventory against the pinned headers.\n\n\
         /// Every MLX-C function bridged by the generated raw declarations.\n\
         pub const BRIDGED_FUNCTIONS: [&str; {}] = {:?};\n\n\
         /// Every MLX-C type bridged by the generated raw declarations.\n\
         pub const BRIDGED_TYPES: [&str; {}] = {:?};\n",
        bridged_functions.len(),
        bridged_functions,
        bridged_types.len(),
        bridged_types,
    );
    std::fs::write(
        output_directory.join(BRIDGED_INVENTORY_FILE_NAME),
        inventory_source,
    )?;
    Ok(())
}

/// Emits rerun contracts for the same tracked inputs the source identity
/// covers, so a pin or patch edit reruns this script and resolves the new
/// extraction directory. Directory paths are watched recursively.
fn emit_rerun_contracts(repository_root: &Path) {
    for tracked_input in [
        "third-party/native-dependency-manifest.cmake",
        "third-party/pins",
        "third-party/patches",
        "crates/runtime-integration/native",
    ] {
        println!(
            "cargo:rerun-if-changed={}",
            repository_root.join(tracked_input).display()
        );
    }
}

/// Resolves the provisioned bindgen header extraction directory for the
/// current source identity. The identity computation mirrors
/// `astronomical-runtime-integration`'s native identity resolution, including
/// the removal of toolchain override variables for production parity.
fn resolve_headers_root(repository_root: &Path) -> Result<PathBuf, Box<dyn Error>> {
    let extraction_identity = resolve_extraction_identity(repository_root)?;
    let headers_root = native_dependency_cache_directory()?
        .join("bindgen-headers")
        .join(extraction_identity);
    if !headers_root.join(COMPLETION_MARKER_FILE_NAME).is_file() {
        return Err(format!(
            "no provisioned bindgen headers for the current source identity at {}; run scripts/provision-bindgen-headers.sh (see third-party/README.md)",
            headers_root.display()
        )
        .into());
    }
    Ok(headers_root)
}

fn resolve_extraction_identity(repository_root: &Path) -> Result<String, Box<dyn Error>> {
    let fingerprint_script_path = repository_root.join("scripts/native-build-cache-fingerprint.sh");
    let mut identity_command = Command::new(&fingerprint_script_path);
    identity_command
        .arg("--source-only")
        .arg("--profile")
        .arg("core")
        .arg(repository_root);
    // Fixture overrides make identity contracts hermetic, but a production
    // build must always fingerprint the real tracked inputs, mirroring the
    // native identity resolution discipline.
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
        return Err(format!(
            "native source identity fingerprint failed: {}",
            String::from_utf8_lossy(&identity_output.stderr)
        )
        .into());
    }
    let identity = String::from_utf8(identity_output.stdout)?;
    let identity = identity.trim();
    if identity.len() != 64
        || !identity
            .bytes()
            .all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f'))
    {
        return Err(format!(
            "native source identity fingerprint returned an invalid identity: {identity}"
        )
        .into());
    }
    Ok(identity.to_owned())
}

fn native_dependency_cache_directory() -> Result<PathBuf, Box<dyn Error>> {
    if let Some(cache_directory) = env::var_os(NATIVE_DEPENDENCY_CACHE_VARIABLE) {
        return Ok(PathBuf::from(cache_directory));
    }
    let home_directory = env::var_os("HOME")
        .map(PathBuf::from)
        .ok_or("set ASTRONOMICAL_NATIVE_DEPENDENCY_CACHE_DIR or HOME to select the native dependency cache")?;
    Ok(home_directory.join(DEFAULT_NATIVE_DEPENDENCY_CACHE_SUFFIX))
}

fn required_path_variable(variable_name: &str) -> Result<PathBuf, Box<dyn Error>> {
    env::var_os(variable_name)
        .map(PathBuf::from)
        .ok_or_else(|| format!("required environment variable is unset: {variable_name}").into())
}
