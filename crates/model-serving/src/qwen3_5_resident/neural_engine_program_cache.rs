//! Compiled Neural Engine programs kept on disk between model loads.
//!
//! Compiling a program and loading it for the first time is nearly all of what a
//! load costs: 86 s for a 9B model's MLP programs, against about 2 s once macOS has
//! seen them (issue #161). macOS keeps that work keyed by the compiled model's
//! path, so a program is compiled straight to the path it will keep and is first
//! loaded from there. Moving one to a new path afterwards repeats about 75 percent of
//! the slow first load, which is why these are built where they will stay and are not
//! shipped.

use std::fs;
use std::path::{Path, PathBuf};

use astronomical_config::{
    NEURAL_ENGINE_PROGRAM_MANIFEST_FILE_NAME, NeuralEngineProgramSetManifest, directory_byte_count,
};
use astronomical_neural_engine::core_ml::{ComputePlacement, CoreMlComputeUnits, CoreMlProgram};
use astronomical_neural_engine::program_shape::ProgramShape;

use super::neural_engine_mlp_prefill::ProgramPackageDirectory;

const MANIFEST_TEMPORARY_FILE_NAME: &str = ".programs.json.incomplete";
const BUILT_WITH: &str = concat!("Astronomical ", env!("CARGO_PKG_VERSION"));

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
        fs::create_dir_all(&set_directory).ok()?;
        remove_other_sets_of_family(model_cache_directory, family, &set_directory_name);
        Some(Self { set_directory })
    }

    pub(crate) fn program_path(&self, layer_index: usize) -> PathBuf {
        self.set_directory
            .join(format!("layer-{layer_index}.mlmodelc"))
    }

    /// Records a finished set in `programs.json`. Only a report depends on it, so a
    /// failure is logged and never fails the load. Skips the work when nothing changed,
    /// because this runs on every load and sizing a set walks thousands of files.
    pub(crate) fn record_finished_set(&self, program_count: u64) {
        let manifest_path = self
            .set_directory
            .join(NEURAL_ENGINE_PROGRAM_MANIFEST_FILE_NAME);
        if self.manifest_already_records(&manifest_path, program_count) {
            return;
        }
        if let Err(write_error) = self.write_manifest(&manifest_path, program_count) {
            tracing::warn!(error = %write_error, "could not record the Neural Engine program set");
        }
    }

    fn manifest_already_records(&self, manifest_path: &Path, program_count: u64) -> bool {
        let Ok(manifest_bytes) = fs::read(manifest_path) else {
            return false;
        };
        match serde_json::from_slice::<NeuralEngineProgramSetManifest>(&manifest_bytes) {
            Ok(manifest) => manifest.program_count == program_count,
            Err(_) => false,
        }
    }

    fn write_manifest(&self, manifest_path: &Path, program_count: u64) -> std::io::Result<()> {
        let manifest = NeuralEngineProgramSetManifest {
            model_id: self.ancestor_name(2),
            model_revision: self.ancestor_name(1),
            set_name: self.ancestor_name(0),
            program_count,
            byte_count: directory_byte_count(&self.set_directory),
            macos_version: macos_product_version(),
            built_with: BUILT_WITH.to_owned(),
        };
        let temporary_path = self.set_directory.join(MANIFEST_TEMPORARY_FILE_NAME);
        fs::write(&temporary_path, serde_json::to_vec_pretty(&manifest)?)?;
        fs::rename(&temporary_path, manifest_path)
    }

    /// The set's own name at 0, its revision at 1, its model at 2.
    fn ancestor_name(&self, levels_up: usize) -> String {
        let mut ancestors = self.set_directory.ancestors();
        match ancestors.nth(levels_up).and_then(Path::file_name) {
            Some(name) => name.to_string_lossy().into_owned(),
            None => String::new(),
        }
    }
}

fn remove_other_sets_of_family(model_cache_directory: &Path, family: &str, kept_set_name: &str) {
    let Ok(directory_entries) = fs::read_dir(model_cache_directory) else {
        return;
    };
    let family_prefix = format!("{family}-");
    for directory_entry in directory_entries.flatten() {
        let entry_name = directory_entry.file_name();
        let entry_name = entry_name.to_string_lossy();
        if !entry_name.starts_with(&family_prefix) || entry_name == kept_set_name {
            continue;
        }
        if let Err(removal_error) = fs::remove_dir_all(directory_entry.path()) {
            tracing::warn!(error = %removal_error, "could not remove an outdated Neural Engine program set");
        }
    }
}

/// macOS may redo its compilation after an update, so the version helps explain a
/// slow load later.
fn macos_product_version() -> Option<String> {
    let output = std::process::Command::new("/usr/bin/sw_vers")
        .arg("-productVersion")
        .output()
        .ok()?;
    let version = String::from_utf8(output.stdout).ok()?;
    let version = version.trim();
    (output.status.success() && !version.is_empty()).then_some(version.to_owned())
}

/// What one layer's program is built from. The MLP and the output gate differ
/// only here, so loading, verifying and caching are written once.
pub(crate) trait ProgramSource {
    fn shape(&self) -> ProgramShape;

    /// Compiles the program and leaves it at `destination` without loading it.
    fn compile_to(&self, package_directory: &Path, destination: &Path) -> Result<(), &'static str>;

    /// Compiles into the temporary directory and loads from there.
    fn compile_and_load(&self, package_directory: &Path) -> Result<CoreMlProgram, &'static str>;
}

/// A program is usable only when all of it sits on the Neural Engine and it has
/// run once. The first prediction wires its weights and is far slower than the
/// rest, so running it here keeps that cost out of a user's first long prompt, and
/// a program that cannot run fails the load instead of a request.
fn verify_program(program: &mut CoreMlProgram) -> Result<(), &'static str> {
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

fn load_and_verify_kept_program(
    source: &dyn ProgramSource,
    kept_program_path: &Path,
) -> Result<CoreMlProgram, String> {
    let mut program = match CoreMlProgram::load_from_path(
        kept_program_path,
        source.shape(),
        CoreMlComputeUnits::CpuAndNeuralEngine,
    ) {
        Ok(program) => program,
        Err(load_error) => return Err(load_error.to_string()),
    };
    match verify_program(&mut program) {
        Ok(()) => Ok(program),
        Err(verification_failure) => Err(verification_failure.to_owned()),
    }
}

/// The program at its kept path, compiled there first when it is missing. `None`
/// sends the caller to the uncached path, so a cache that cannot be used costs a
/// slower load and nothing else.
fn load_kept_program(
    source: &dyn ProgramSource,
    kept_program_path: &Path,
    layer_index: usize,
) -> Option<CoreMlProgram> {
    if !kept_program_path.is_dir() {
        let package_directory = ProgramPackageDirectory::create(layer_index)?;
        source
            .compile_to(&package_directory.path, kept_program_path)
            .ok()?;
    }
    let failure_reason = match load_and_verify_kept_program(source, kept_program_path) {
        Ok(program) => return Some(program),
        Err(failure_reason) => failure_reason,
    };
    // Removing it lets the next load compile a fresh one, which has been stable. On one
    // Mac 4 to 6 of a small model's 42 programs failed this way on the first reload after
    // they were compiled, and the same ones failed again if kept, so waiting does not help.
    // Logged, because the recompile costs the next load time that nothing else explains.
    tracing::warn!(
        layer_index,
        reason = %failure_reason,
        "removing a kept Neural Engine program that could not be used"
    );
    if let Err(removal_error) = fs::remove_dir_all(kept_program_path) {
        tracing::warn!(error = %removal_error, "could not remove an unusable kept Neural Engine program");
    }
    None
}

/// One verified program, from `cache` when there is one.
pub(crate) fn load_verified_program(
    source: &dyn ProgramSource,
    cache: Option<&ProgramSetCache>,
    layer_index: usize,
) -> Result<CoreMlProgram, &'static str> {
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
