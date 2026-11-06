//! Compiled Neural Engine programs kept on disk between model loads.
//!
//! Compiling a program and loading it for the first time is nearly all of what the
//! set costs to prepare: 86 s for a 9B model's MLP programs, against about 2 s once macOS has
//! seen them (issue #161). macOS keeps that work keyed by the compiled model's
//! path, so a program is compiled straight to the path it will keep and is first
//! loaded from there. Moving one to a new path afterwards repeats about 75 percent of
//! the slow first load, which is why these are built where they will stay and are not
//! shipped.
//!
//! macOS keeps its copies per app and drops them when the app's binary changes: a 9B
//! model's 56 programs took 94 s to load after a rebuild, then 43 s, 13 s and 2.5 s
//! over the next three loads of the same build, with nothing else changed. Writing a
//! program to its kept path is not the slow part, at about 0.02 s each.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use astronomical_config::{
    NEURAL_ENGINE_PROGRAM_MANIFEST_FILE_NAME, NeuralEngineProgramSetManifest, directory_byte_count,
};
use astronomical_neural_engine::core_ml::{
    ComputePlacement, CoreMlComputeUnits, CoreMlProgram, compile_model_to,
};
use astronomical_neural_engine::program_shape::ProgramShape;

use super::host_description::{current_host_environment, macos_product_version};
use super::neural_engine_mlp_prefill::NEURAL_ENGINE_FINISH_TIMEOUT;
use super::neural_engine_program_placement_record::{
    PlacementCheck, forget_placement_verified, placement_check_for, record_placement_verified,
};
use super::neural_engine_program_preparation::{
    ProgramBuildJob, ProgramBuildOutcome, ProgramBuilder,
};

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

    pub(crate) fn set_directory(&self) -> &Path {
        &self.set_directory
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
            macos_version: macos_product_version().map(str::to_owned),
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

/// Where a layer's program stays between loads, when there is a cache to keep it in.
pub(crate) fn kept_program_path(
    program_cache: Option<&ProgramSetCache>,
    decoder_layer_index: usize,
) -> Option<PathBuf> {
    Some(program_cache?.program_path(decoder_layer_index))
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

/// A program is usable only when all of it sits on the Neural Engine, so one whose
/// placement Core ML cannot show is refused.
fn verify_placement(program: &CoreMlProgram) -> Result<(), &'static str> {
    let placement = program.placement();
    if placement.preferred != ComputePlacement::NeuralEngine
        || placement.operations_off_neural_engine != 0
    {
        tracing::warn!(
            ?placement,
            "Core ML reported an unexpected program placement"
        );
        return Err("Core ML did not place the whole program on the Neural Engine");
    }
    Ok(())
}

/// The first prediction wires a program's weights and is far slower than the rest,
/// so running it here keeps that cost out of a user's first long prompt, and a
/// program that cannot run is left out instead of failing a request. Returns how
/// long that prediction took.
fn run_first_prediction(program: &mut CoreMlProgram) -> Result<Duration, &'static str> {
    let first_prediction_started_at = Instant::now();
    if program
        .predict_within(NEURAL_ENGINE_FINISH_TIMEOUT)
        .is_err()
    {
        return Err("the Neural Engine could not run the program");
    }
    Ok(first_prediction_started_at.elapsed())
}

/// A reload costs from a few milliseconds to seconds per program depending on what
/// macOS still holds, and only per-step times show which step a slow one spent it in.
fn log_program_load_timing(
    build_job: &ProgramBuildJob,
    program: &CoreMlProgram,
    is_kept_program: bool,
    load_duration: Duration,
    first_prediction_duration: Duration,
) {
    let (model_load_seconds, placement_seconds) = program.load_phase_seconds();
    tracing::info!(
        family = ?build_job.identity.family,
        decoder_layer_index = build_job.identity.decoder_layer_index,
        is_kept_program,
        load_seconds = load_duration.as_secs_f64(),
        model_load_seconds,
        placement_seconds,
        first_prediction_seconds = first_prediction_duration.as_secs_f64(),
        "Neural Engine program loaded"
    );
}

/// Loads a kept program, proving its placement only when no earlier load in this
/// environment did. See `neural_engine_program_placement_record` for what the proof
/// costs. An unproven load cannot report placement, so the record is the only
/// thing that lets one through.
fn load_and_verify_kept_program(
    build_job: &ProgramBuildJob,
    kept_program_path: &Path,
) -> Result<CoreMlProgram, String> {
    let placement_environment = current_host_environment();
    let placement_check = placement_check_for(kept_program_path, placement_environment);
    let load_started_at = Instant::now();
    let load_outcome = match placement_check {
        PlacementCheck::AlreadyProven => CoreMlProgram::load_from_path_without_placement_check(
            kept_program_path,
            build_job.shape,
            CoreMlComputeUnits::CpuAndNeuralEngine,
        ),
        PlacementCheck::Required => CoreMlProgram::load_from_path(
            kept_program_path,
            build_job.shape,
            CoreMlComputeUnits::CpuAndNeuralEngine,
        ),
    };
    let mut program = match load_outcome {
        Ok(program) => program,
        Err(load_error) => return Err(load_error.to_string()),
    };
    let load_duration = load_started_at.elapsed();
    if placement_check == PlacementCheck::Required {
        verify_placement(&program)?;
        if let Some(placement_environment) = placement_environment {
            record_placement_verified(kept_program_path, placement_environment);
        }
    }
    let first_prediction_duration = run_first_prediction(&mut program)?;
    log_program_load_timing(
        build_job,
        &program,
        true,
        load_duration,
        first_prediction_duration,
    );
    Ok(program)
}

/// Builds programs with Core ML. Holds nothing from the model, because it runs on
/// the build thread and every weight it needs arrives in a package on disk.
pub(crate) struct CoreMlProgramBuilder;

impl CoreMlProgramBuilder {
    /// The program at its kept path, compiled there first when it is missing. A cache
    /// that cannot be used costs a slower load and nothing else.
    fn build_kept(
        build_job: &ProgramBuildJob,
        kept_program_path: &Path,
    ) -> ProgramBuildOutcome<CoreMlProgram> {
        if !kept_program_path.is_dir() {
            // A record left by a program that is gone must not vouch for the one
            // about to be compiled to the same path.
            forget_placement_verified(kept_program_path);
            let Some(package) = &build_job.package else {
                return ProgramBuildOutcome::NeedsUncachedPackage;
            };
            let compile_started_at = Instant::now();
            if compile_model_to(&package.package_path, kept_program_path).is_err() {
                return ProgramBuildOutcome::NeedsUncachedPackage;
            }
            tracing::info!(
                family = ?build_job.identity.family,
                decoder_layer_index = build_job.identity.decoder_layer_index,
                compile_seconds = compile_started_at.elapsed().as_secs_f64(),
                "Neural Engine program compiled to its kept path"
            );
        }
        let failure_reason = match load_and_verify_kept_program(build_job, kept_program_path) {
            Ok(program) => return ProgramBuildOutcome::Built(program),
            Err(failure_reason) => failure_reason,
        };
        // Removing it lets the next load compile a fresh one, which has been stable. On one
        // Mac 4 to 6 of a small model's 42 programs failed this way on the first reload after
        // they were compiled, and the same ones failed again if kept, so waiting does not help.
        // On a 9B model 17 of 56 failed on one reload, every one in the placement check after
        // its model had loaded. A program proven once is not checked again, so this now
        // happens at most on the loads before a program has its record.
        //
        // Logged, because the recompile costs the next load time that nothing else explains.
        tracing::warn!(
            family = ?build_job.identity.family,
            decoder_layer_index = build_job.identity.decoder_layer_index,
            reason = %failure_reason,
            "removing a kept Neural Engine program that could not be used"
        );
        forget_placement_verified(kept_program_path);
        if let Err(removal_error) = fs::remove_dir_all(kept_program_path) {
            tracing::warn!(error = %removal_error, "could not remove an unusable kept Neural Engine program");
        }
        ProgramBuildOutcome::NeedsUncachedPackage
    }

    fn build_uncached(build_job: &ProgramBuildJob) -> ProgramBuildOutcome<CoreMlProgram> {
        let Some(package) = &build_job.package else {
            return ProgramBuildOutcome::Failed("no package exists for the Core ML program");
        };
        let load_started_at = Instant::now();
        let mut program = match CoreMlProgram::load_from_path(
            &package.package_path,
            build_job.shape,
            CoreMlComputeUnits::CpuAndNeuralEngine,
        ) {
            Ok(program) => program,
            Err(_) => {
                return ProgramBuildOutcome::Failed(
                    "Core ML could not load the program on this macOS",
                );
            }
        };
        let load_duration = load_started_at.elapsed();
        if let Err(placement_failure) = verify_placement(&program) {
            return ProgramBuildOutcome::Failed(placement_failure);
        }
        let first_prediction_duration = match run_first_prediction(&mut program) {
            Ok(first_prediction_duration) => first_prediction_duration,
            Err(prediction_failure) => return ProgramBuildOutcome::Failed(prediction_failure),
        };
        log_program_load_timing(
            build_job,
            &program,
            false,
            load_duration,
            first_prediction_duration,
        );
        // The daemon stops a worker with SIGKILL on a swap or shutdown, so the
        // bridge's release never runs there; left to it, every uncached load wrote a
        // whole layer's weights per program into the temporary directory for good.
        program.discard_compiled_model_files();
        ProgramBuildOutcome::Built(program)
    }
}

impl ProgramBuilder for CoreMlProgramBuilder {
    type Program = CoreMlProgram;

    fn build(&self, build_job: &ProgramBuildJob) -> ProgramBuildOutcome<CoreMlProgram> {
        match &build_job.kept_program_path {
            Some(kept_program_path) => Self::build_kept(build_job, kept_program_path),
            None => Self::build_uncached(build_job),
        }
    }
}

#[cfg(test)]
#[path = "neural_engine_program_cache_tests.rs"]
mod tests;
