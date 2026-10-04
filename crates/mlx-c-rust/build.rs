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

const BINDGEN_FUNCTION_ALLOWLIST: &str = concat!(
    "mlx_(set_error_handler|metal_set_metallib_path|version|string_(new|data|free)|clear_cache|",
    "device_(new_type|free)|device_info_(new|get|get_size|free)|",
    "compile|closure_(new|new_func|free|apply)|",
    "get_(active_memory|cache_memory|memory_limit|peak_memory)|reset_peak_memory|set_(cache|memory)_limit|",
    "array_(new|new_data|free|eval|shape|ndim|dtype|size|nbytes|data_(float32|uint8|uint32)|item_uint32|set)|",
    "default_(cpu|gpu)_stream_new|stream_free|synchronize|",
    "(add(mm)?|arange|argmax_axis|argpartition_axis|argsort_axis|astype|broadcast_to|clip|concatenate_axis|contiguous|conv(1d|2d|3d)|cos|cumsum_axis|dequantize|divide|erf|exp|expand_dims|floor_divide|full|gather_(mm|qmm)|greater|greater_equal|log1p|logaddexp|matmul|pad|power|",
    "max_axis|multiply|negative|put_along_axis|scatter_add_single|quantize|quantized_matmul|repeat_axis|reshape|sigmoid|sin|slice(_update)?|softmax_axis|sqrt|subtract|sum_axis|tanh|less|",
    "squeeze_axis|stack_axis|take_along_axis|take_axis|topk_axis|transpose_axes|where|zeros)|",
    "fast_(rms_norm|layer_norm|rope(_dynamic)?|scaled_dot_product_attention)|fast_metal_kernel(_config)?_(new|free|apply|add_output_arg|set_grid|set_thread_group|set_init_value|add_template_arg_(dtype|int|bool))|random_(categorical|key|normal|split)|eval|async_eval|",
    "vector_array_(new|new_data|free|get|size|set_value|set_data)|vector_string_(new_data|free)|io_(reader|writer)_(new|free)|",
    "save_safetensors_writer|",
    "load_safetensors_reader|map_string_to_array_(new|free|get)|",
    "map_string_to_array_insert|map_string_to_string_(new|free|insert))|",
    "astronomical_metal_expert_loader_(start|wait|free)"
);

const BINDGEN_TYPE_ALLOWLIST: &str = concat!(
    "mlx_(error_handler_func|string|array|dtype|stream|closure|device|device_info|device_type|io_reader|io_vtable|",
    "optional_(dtype|float|int)|vector_array|vector_string|fast_metal_kernel(_config)?|map_string_to_array|map_string_to_string)|",
    "astronomical_metal_expert_loader_(output_tensor|load_range|metrics|handle)"
);

const EXPERT_LOADER_HEADER_REPOSITORY_RELATIVE_PATH: &str = "crates/runtime-integration/native/experimental/aligned_expert_packs/astronomical_metal_expert_loader.h";

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
    bindings.write_to_file(output_directory.join("mlx_c_bindings.rs"))?;
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
