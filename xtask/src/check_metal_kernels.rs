//! `cargo xtask check-metal-kernels`: Astronomical's own Metal kernels are
//! first-class files, checked without a GPU (issue #2712).
//!
//! 1. **Naming:** every file in a `metal/` folder, and every shared header in
//!    `model-primitives/src/metal_common/`, follows
//!    `astro_<scope>_[<owner>_]<operation>[_<variant>]` with the scope and
//!    owner its crate gives it; every registered kernel names its GPU
//!    contract and probe; every helper and shared body is used.
//! 2. **No kernel in Rust text:** `MlxMetalKernel::new` appears only in the
//!    one declarer, so every kernel's Metal is in a `.metal` file, and no
//!    Rust string literal holds C++ or Metal program text at all (see
//!    [`embedded_program_text`]).
//! 3. **Offline compile:** every registered kernel is wrapped exactly as MLX
//!    wraps it at run time and compiled with Apple's offline compiler, for
//!    every `@compile-with` instantiation, every way MLX may pass each input
//!    and every Metal language version MLX may use. This catches what used to
//!    wait for a kernel's first GPU run: syntax and type errors, a pointer in
//!    the wrong address space, a missing helper, a template that does not
//!    instantiate. It does not catch what only a pipeline on a GPU can show
//!    (threadgroup memory limits, matrix units missing on the chip): the
//!    GPU contracts and capability probes do.
//!
//! A source that compiled is remembered by digest under
//! `target/metal-kernel-check`, so the hook compiles only what changed;
//! `--fresh` compiles everything, as CI does. `--skip-compile` runs only the
//! first two checks, for a machine without the Metal toolchain.

mod compile_jobs;
mod embedded_kernel_ratchet;
mod embedded_program_text;
mod kernel_inventory;
mod offline_compiler;
mod rust_file_walk;

#[cfg(test)]
mod check_metal_kernels_arguments_tests;
#[cfg(test)]
mod compile_jobs_tests;
#[cfg(test)]
mod embedded_kernel_ratchet_tests;
#[cfg(test)]
mod embedded_program_text_tests;
#[cfg(test)]
mod kernel_inventory_tests;
#[cfg(test)]
mod offline_compiler_tests;

use std::path::Path;
use std::time::Instant;

use compile_jobs::kernel_compile_jobs;
use embedded_kernel_ratchet::embedded_kernel_sites;
use embedded_program_text::embedded_program_text_sites;
use kernel_inventory::{KernelInventory, kernel_inventory};
use offline_compiler::OfflineCompiler;

const TOOLCHAIN_HINT: &str = "the Metal toolchain is missing: install it with `xcodebuild -downloadComponent MetalToolchain`";

/// The parsed command line.
#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct MetalKernelCheckInvocation {
    pub(crate) is_fresh_run: bool,
    pub(crate) skips_compile: bool,
}

pub(crate) fn parse_arguments(arguments: &[String]) -> Result<MetalKernelCheckInvocation, String> {
    let mut invocation = MetalKernelCheckInvocation::default();
    for argument in arguments {
        match argument.as_str() {
            "--fresh" => invocation.is_fresh_run = true,
            "--skip-compile" => invocation.skips_compile = true,
            unknown_argument => {
                return Err(format!(
                    "usage: cargo xtask check-metal-kernels [--fresh] [--skip-compile]; unknown argument {unknown_argument}"
                ));
            }
        }
    }
    Ok(invocation)
}

pub(crate) fn run(repository_root: &Path, arguments: &[String]) -> Result<(), String> {
    let invocation = parse_arguments(arguments)?;
    let started_at = Instant::now();
    let inventory = kernel_inventory(repository_root);
    let mut violations = inventory.violations.clone();
    for embedded_site in embedded_kernel_sites(repository_root) {
        violations.push(format!(
            "{embedded_site}: declares a Metal kernel from Rust text; move its Metal into a .metal file and declare it with declare_metal_kernel"
        ));
    }
    for program_text_site in embedded_program_text_sites(repository_root) {
        violations.push(format!(
            "{program_text_site}: a Rust string holds C++ or Metal program text; move it into a .metal or C++ file"
        ));
    }
    // Kernels declared from Rust text do not stop the compile of the ones in
    // files; a broken file does, because its kernels cannot be built.
    if !invocation.skips_compile && inventory.violations.is_empty() {
        violations.extend(compile_failures(
            repository_root,
            &inventory,
            invocation.is_fresh_run,
        )?);
    }
    let elapsed_seconds = started_at.elapsed().as_secs_f64();
    if violations.is_empty() {
        println!(
            "[check-metal-kernels] status=success kernels={} elapsed_seconds={elapsed_seconds:.1}",
            inventory.registered_kernels.len()
        );
        return Ok(());
    }
    for violation in &violations {
        eprintln!("[check-metal-kernels] {violation}");
    }
    Err(format!(
        "[check-metal-kernels] status=failed violations={} elapsed_seconds={elapsed_seconds:.1}",
        violations.len()
    ))
}

fn compile_failures(
    repository_root: &Path,
    inventory: &KernelInventory,
    is_fresh_run: bool,
) -> Result<Vec<String>, String> {
    let provisioned_sources = native_build_orchestration::provisioned_mlx_sources(repository_root)
        .map_err(|provisioning_error| provisioning_error.to_string())?;
    let scratch_directory =
        std::env::temp_dir().join(format!("astro-metal-kernel-check-{}", std::process::id()));
    std::fs::create_dir_all(&scratch_directory).map_err(|create_error| create_error.to_string())?;
    let compiler = OfflineCompiler::new(
        repository_root,
        provisioned_sources.mlx_source_directory,
        scratch_directory.clone(),
    );
    let compile_outcome = compile_with(&compiler, inventory, is_fresh_run);
    // Removed whatever the outcome, so a failed run leaves nothing behind.
    let _ = std::fs::remove_dir_all(&scratch_directory);
    compile_outcome
}

fn compile_with(
    compiler: &OfflineCompiler,
    inventory: &KernelInventory,
    is_fresh_run: bool,
) -> Result<Vec<String>, String> {
    std::fs::create_dir_all(&compiler.compiled_source_store)
        .map_err(|create_error| create_error.to_string())?;
    let mut compile_jobs = Vec::new();
    for registered_kernel in &inventory.registered_kernels {
        compile_jobs.extend(kernel_compile_jobs(registered_kernel)?);
    }
    let job_count = compile_jobs.len();
    let pending_jobs = compiler.pending_jobs(compile_jobs, is_fresh_run);
    if !pending_jobs.is_empty() {
        check_toolchain()?;
    }
    println!(
        "[check-metal-kernels] compiling {} of {job_count} kernel sources (the rest compiled before)",
        pending_jobs.len()
    );
    let worker_count = std::thread::available_parallelism().map_or(4, std::num::NonZero::get);
    let mut violations = Vec::new();
    for failure in compiler.compile_all(&pending_jobs, worker_count) {
        violations.push(format!(
            "{} does not compile:\n{}",
            failure.description, failure.compiler_output
        ));
    }
    Ok(violations)
}

/// CI and the hook fail, never skip, when the compiler is missing: a check
/// that silently does nothing is the failure this check exists to prevent.
fn check_toolchain() -> Result<(), String> {
    let version_run = std::process::Command::new("xcrun")
        .args(["-sdk", "macosx", "metal", "--version"])
        .output()
        .map_err(|_| TOOLCHAIN_HINT.to_owned())?;
    if version_run.status.success() {
        Ok(())
    } else {
        Err(TOOLCHAIN_HINT.to_owned())
    }
}
