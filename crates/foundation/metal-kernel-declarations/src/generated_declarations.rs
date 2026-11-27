//! Writes, from the build script of a crate that owns kernels, one
//! [`crate::MetalKernelSource`] constant a registered kernel.
//!
//! The launcher then declares its kernel from that constant alone, so the
//! input and output names, their order and the bytes MLX receives all come
//! from the `.metal` files and cannot drift from them. The generated texts
//! are `include_str!` of the files themselves, so the files stay the only
//! copy of the Metal code.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use crate::kernel_files::{
    RegisteredKernel, kernel_folder_files, registered_kernels, repository_root_above,
};
use crate::kernel_naming::{
    SHARED_HEADER_DIRECTORY, crate_kernel_scope, declaration_constant_name,
};

/// The file, in `OUT_DIR`, holding the declarations of the kernels under the
/// crate's `src/`; the crate includes it in a module of its own.
pub const SOURCE_DECLARATIONS_FILE_NAME: &str = "metal_kernels.rs";
/// The file, in `OUT_DIR`, holding the declarations of the kernels under the
/// crate's `tests/`; integration tests include it.
pub const TEST_DECLARATIONS_FILE_NAME: &str = "test_metal_kernels.rs";

/// The build script of a crate that owns kernels, entire: reads the crate
/// and output folders Cargo gives every build script, and stops the build
/// with the reason when the declarations cannot be generated, so each
/// owning crate's `build.rs` is one call.
///
/// # Panics
/// When Cargo did not set `CARGO_MANIFEST_DIR` or `OUT_DIR`, or when
/// [`generate_kernel_declarations_for_build`] fails.
pub fn run_kernel_declarations_build_script() {
    let crate_directory =
        PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("Cargo sets CARGO_MANIFEST_DIR"));
    let out_directory = PathBuf::from(std::env::var("OUT_DIR").expect("Cargo sets OUT_DIR"));
    if let Err(generation_error) =
        generate_kernel_declarations_for_build(&crate_directory, &out_directory)
    {
        panic!("the Metal kernel declarations cannot be generated: {generation_error}");
    }
}

/// The whole build-script step: finds the crate's kernels under `src/` and
/// `tests/`, writes their declarations into `out_directory`, and tells Cargo
/// to rerun when anything under `src/`, `tests/` or the shared header folder
/// changes, so a Metal folder added later is found too.
///
/// # Errors
/// Fails when the crate may own no kernels, when a kernel file breaks the
/// convention or its block cannot be read, or when a file cannot be written.
pub fn generate_kernel_declarations_for_build(
    crate_directory: &Path,
    out_directory: &Path,
) -> Result<(), String> {
    let repository_root = repository_root_above(crate_directory)
        .ok_or_else(|| format!("no workspace above {}", crate_directory.display()))?;
    let crate_relative_directory = relative_slash_path(crate_directory, &repository_root)?;
    let crate_scope = crate_kernel_scope(&crate_relative_directory).ok_or_else(|| {
        format!("{crate_relative_directory} is not a crate that may own Metal kernels")
    })?;
    let shared_header_directory = repository_root.join(SHARED_HEADER_DIRECTORY);
    println!(
        "cargo:rerun-if-changed={}",
        shared_header_directory.display()
    );
    for (search_folder, declarations_file_name) in [
        ("src", SOURCE_DECLARATIONS_FILE_NAME),
        ("tests", TEST_DECLARATIONS_FILE_NAME),
    ] {
        let search_directory = crate_directory.join(search_folder);
        // The whole folder, not each Metal file: Cargo scans a folder for any
        // change, so a `metal/` folder added later reruns the script too, and
        // any change under it recompiles the crate anyway.
        println!("cargo:rerun-if-changed={}", search_directory.display());
        let kernel_files = kernel_folder_files(&search_directory);
        let kernels = registered_kernels(
            &kernel_files,
            &crate_scope.name_prefix(),
            &shared_header_directory,
        )?;
        let declarations_text = declarations_source(&kernels)?;
        std::fs::write(
            out_directory.join(declarations_file_name),
            declarations_text,
        )
        .map_err(|write_error| format!("writing {declarations_file_name}: {write_error}"))?;
    }
    Ok(())
}

fn relative_slash_path(directory: &Path, repository_root: &Path) -> Result<String, String> {
    let relative_directory = directory.strip_prefix(repository_root).map_err(|_| {
        format!(
            "{} is outside {}",
            directory.display(),
            repository_root.display()
        )
    })?;
    let mut path_words = Vec::new();
    for path_component in relative_directory.components() {
        path_words.push(path_component.as_os_str().to_string_lossy().into_owned());
    }
    Ok(path_words.join("/"))
}

/// The Rust text declaring every kernel in `kernels`.
///
/// # Errors
/// Fails only when a path is not valid UTF-8, which `include_str!` cannot name.
pub fn declarations_source(kernels: &[RegisteredKernel]) -> Result<String, String> {
    let mut declarations_text = String::from(
        "// Generated by metal-kernel-declarations from the crate's .metal files; do not edit.\n",
    );
    for kernel in kernels {
        declarations_text.push_str(&kernel_declaration(kernel)?);
    }
    Ok(declarations_text)
}

fn kernel_declaration(kernel: &RegisteredKernel) -> Result<String, String> {
    let signature = &kernel.signature;
    let mut input_names = Vec::new();
    for input in &signature.inputs {
        input_names.push(input.name.as_str());
    }
    let mut output_names = Vec::new();
    for output in &signature.outputs {
        output_names.push(output.name.as_str());
    }
    let mut template_parameter_names = Vec::new();
    for template_parameter in &signature.template_parameters {
        template_parameter_names.push(template_parameter.name.as_str());
    }
    let mut declaration = String::new();
    let _ = writeln!(declaration, "/// The registered kernel `{}`.", kernel.name);
    let _ = writeln!(
        declaration,
        "pub const {}: ::metal_kernel_declarations::MetalKernelSource = ::metal_kernel_declarations::MetalKernelSource {{",
        declaration_constant_name(&kernel.name)
    );
    let _ = writeln!(declaration, "    name: {:?},", kernel.name);
    let _ = writeln!(declaration, "    input_names: &{input_names:?},");
    let _ = writeln!(declaration, "    output_names: &{output_names:?},");
    let _ = writeln!(
        declaration,
        "    template_parameter_names: &{template_parameter_names:?},"
    );
    let _ = writeln!(
        declaration,
        "    define_names: &{:?},",
        signature.define_names
    );
    let _ = writeln!(
        declaration,
        "    has_atomic_outputs: {},",
        signature.has_atomic_outputs
    );
    let _ = writeln!(
        declaration,
        "    header_parts: &[{}],",
        included_texts(&kernel.header_files)?
    );
    let _ = writeln!(
        declaration,
        "    source_parts: &[{}],",
        included_texts(&kernel.source_files)?
    );
    declaration.push_str("};\n");
    Ok(declaration)
}

fn included_texts(files: &[PathBuf]) -> Result<String, String> {
    let mut included_texts = Vec::new();
    for included_file in files {
        let file_text = included_file
            .to_str()
            .ok_or_else(|| format!("{} is not a UTF-8 path", included_file.display()))?;
        included_texts.push(format!("include_str!({file_text:?})"));
    }
    Ok(included_texts.join(", "))
}
