//! Compiled Neural Engine programs kept on disk between model loads.
//!
//! Compiling a program and loading it for the first time is nearly all of what a
//! load costs: 86 s for a 9B model's MLP programs, against about 2 s once macOS has
//! seen them (issue #161). macOS keeps that work keyed by the compiled model's
//! path, so a program is compiled straight to the path it will keep and is first
//! loaded from there. Loading it from the temporary directory and moving it
//! afterwards pays the slow first load twice.

use std::path::{Path, PathBuf};

use astronomical_neural_engine::core_ml_projection::{
    ComputePlacement, CoreMlComputeUnits, CoreMlProjection,
};
use astronomical_neural_engine::projection_shape::ProjectionShape;

use super::neural_engine_mlp_prefill::ProgramPackageDirectory;

/// One family's programs for one model revision, in the shape they run at.
pub(crate) struct ProgramSetCache {
    set_directory: PathBuf,
}

impl ProgramSetCache {
    /// Opens `model_cache_directory/<family>-<set_detail>` and removes the same
    /// family's other sets. A set is several gigabytes, and one left behind by an
    /// older chunk size or program format would otherwise stay on disk for good.
    pub(crate) fn open(
        model_cache_directory: &Path,
        family: &str,
        set_detail: &str,
    ) -> Option<Self> {
        let set_directory_name = format!("{family}-{set_detail}");
        let set_directory = model_cache_directory.join(&set_directory_name);
        std::fs::create_dir_all(&set_directory).ok()?;
        remove_other_sets_of_family(model_cache_directory, family, &set_directory_name);
        Some(Self { set_directory })
    }

    pub(crate) fn program_path(&self, layer_index: usize) -> PathBuf {
        self.set_directory
            .join(format!("layer-{layer_index}.mlmodelc"))
    }
}

fn remove_other_sets_of_family(model_cache_directory: &Path, family: &str, kept_set_name: &str) {
    let Ok(directory_entries) = std::fs::read_dir(model_cache_directory) else {
        return;
    };
    let family_prefix = format!("{family}-");
    for directory_entry in directory_entries.flatten() {
        let entry_name = directory_entry.file_name();
        let entry_name = entry_name.to_string_lossy();
        if !entry_name.starts_with(&family_prefix) || entry_name == kept_set_name {
            continue;
        }
        if let Err(removal_error) = std::fs::remove_dir_all(directory_entry.path()) {
            tracing::warn!(error = %removal_error, "could not remove an outdated Neural Engine program set");
        }
    }
}

/// What one layer's program is built from. The MLP and the output gate differ
/// only here, so loading, verifying and caching are written once.
pub(crate) trait ProgramSource {
    fn shape(&self) -> ProjectionShape;

    /// Compiles the program and leaves it at `destination` without loading it.
    fn compile_to(&self, package_directory: &Path, destination: &Path) -> Result<(), &'static str>;

    /// Compiles into the temporary directory and loads from there.
    fn compile_and_load(&self, package_directory: &Path) -> Result<CoreMlProjection, &'static str>;
}

/// A program is usable only when all of it sits on the Neural Engine and it has
/// run once. The first prediction wires its weights and is far slower than the
/// rest, so running it here keeps that cost out of a user's first long prompt, and
/// a program that cannot run fails the load instead of a request.
fn verify_program(program: &mut CoreMlProjection) -> Result<(), &'static str> {
    let placement = program.placement();
    if placement.preferred != ComputePlacement::NeuralEngine
        || placement.operations_off_neural_engine != 0
    {
        return Err("Core ML did not place the whole program on the Neural Engine");
    }
    if program.predict().is_err() {
        return Err("the Neural Engine could not run the program");
    }
    Ok(())
}

/// The program at its kept path, compiled there first when it is missing. `None`
/// sends the caller to the uncached path, so a cache that cannot be used costs a
/// slower load and nothing else.
fn load_kept_program(
    source: &dyn ProgramSource,
    kept_program_path: &Path,
    layer_index: usize,
) -> Option<CoreMlProjection> {
    if !kept_program_path.is_dir() {
        let package_directory = ProgramPackageDirectory::create(layer_index)?;
        source
            .compile_to(&package_directory.path, kept_program_path)
            .ok()?;
    }
    if let Ok(mut program) = CoreMlProjection::load_compiled_model(
        kept_program_path,
        source.shape(),
        CoreMlComputeUnits::CpuAndNeuralEngine,
    ) && verify_program(&mut program).is_ok()
    {
        return Some(program);
    }
    // A kept program that will not load or run is damaged or from another macOS,
    // and removing it lets the next load compile a fresh one.
    if let Err(removal_error) = std::fs::remove_dir_all(kept_program_path) {
        tracing::warn!(error = %removal_error, "could not remove an unusable kept Neural Engine program");
    }
    None
}

/// One verified program, from `cache` when there is one.
pub(crate) fn load_verified_program(
    source: &dyn ProgramSource,
    cache: Option<&ProgramSetCache>,
    layer_index: usize,
) -> Result<CoreMlProjection, &'static str> {
    if let Some(cache) = cache
        && let Some(program) =
            load_kept_program(source, &cache.program_path(layer_index), layer_index)
    {
        return Ok(program);
    }
    let Some(package_directory) = ProgramPackageDirectory::create(layer_index) else {
        return Err("no scratch directory for the Core ML program");
    };
    let mut program = source.compile_and_load(&package_directory.path)?;
    verify_program(&mut program)?;
    // The daemon stops a worker with SIGKILL on a swap or shutdown, so the
    // bridge's release never runs there; left to it, every uncached load wrote a
    // whole layer's weights per program into the temporary directory for good.
    program.discard_compiled_model_files();
    Ok(program)
}

#[cfg(test)]
#[path = "neural_engine_program_cache_tests.rs"]
mod tests;
