//! `quantize drafter`: a DFlash drafter rewritten in the packing
//! Astronomical serves it in (feature `quantize-convert`).
//!
//! The engine quantizes a bf16 drafter's linears to 4 bits, group 64, each
//! time it loads one. A package that ships the drafter already packed this
//! way downloads about a third of the bytes and serves bit for bit the same
//! weights, because the arrays are made here by the engine's own load-time
//! call (`drafter_tensors_at_serving_packing`). Every other file of the
//! drafter's folder is copied unchanged.

use std::path::Path;
use std::time::Instant;

use engine_core::speculative_decoding::dflash_drafter::drafter_tensors_at_serving_packing;
use memory_policy::serving_ceiling::machine_ceiling_from_physical_ram_bytes;
use runtime_integration::MlxRuntime;

use super::quantize_convert::{memory_limits_for, sample_physical_ram_bytes};
use super::{QuantizeDependencies, QuantizeDrafterArguments, QuantizeError};

/// The drafter's weights file, in its source folder and in the output.
const DRAFTER_WEIGHTS_FILE_NAME: &str = "model.safetensors";
/// Recorded in the packed file's header, as MLX's own packed files say.
const PACKED_FORMAT_METADATA: (&str, &str) = ("format", "mlx");
const BYTES_PER_GIGABYTE: f64 = 1_073_741_824.0;

/// Packs the drafter's linears, writes the packed weights file and copies
/// the rest of the drafter's folder beside it.
pub fn run_quantize_drafter(
    drafter_arguments: &QuantizeDrafterArguments,
    dependencies: &mut QuantizeDependencies<'_>,
) -> Result<(), QuantizeError> {
    let started_at = Instant::now();
    let source_weights_path = drafter_arguments
        .drafter_directory
        .join(DRAFTER_WEIGHTS_FILE_NAME);
    let output_directory = &drafter_arguments.output_directory;
    std::fs::create_dir(output_directory).map_err(packing_failure)?;
    let machine_ceiling_bytes =
        machine_ceiling_from_physical_ram_bytes(sample_physical_ram_bytes()?);
    let runtime = MlxRuntime::initialize(memory_limits_for(machine_ceiling_bytes)?)
        .map_err(packing_failure)?;
    report_stage(dependencies, "packing the drafter's linears", started_at);
    let packed_tensors = drafter_tensors_at_serving_packing(&runtime, &source_weights_path)
        .map_err(packing_failure)?;
    report_stage(dependencies, "writing the packed weights", started_at);
    let named_tensors: Vec<(&str, &mlx_bridge::MlxArray)> = packed_tensors
        .iter()
        .map(|(tensor_name, tensor)| (tensor_name.as_str(), tensor))
        .collect();
    let packed_weights_path = output_directory.join(DRAFTER_WEIGHTS_FILE_NAME);
    let packed_weights_file =
        std::fs::File::create(&packed_weights_path).map_err(packing_failure)?;
    runtime
        .save_safetensors(
            packed_weights_file,
            &named_tensors,
            &[PACKED_FORMAT_METADATA],
        )
        .map_err(packing_failure)?;
    copy_drafter_folder(&drafter_arguments.drafter_directory, output_directory)?;
    report_summary(
        dependencies,
        (&source_weights_path, &packed_weights_path),
        started_at,
    )
}

/// Copies every file and folder of the drafter's own folder except its
/// weights, which the packed file replaces.
fn copy_drafter_folder(
    source_directory: &Path,
    output_directory: &Path,
) -> Result<(), QuantizeError> {
    for entry in std::fs::read_dir(source_directory).map_err(packing_failure)? {
        let entry = entry.map_err(packing_failure)?;
        if entry.file_name() == DRAFTER_WEIGHTS_FILE_NAME {
            continue;
        }
        copy_entry(&entry.path(), &output_directory.join(entry.file_name()))?;
    }
    Ok(())
}

fn copy_entry(source_path: &Path, output_path: &Path) -> Result<(), QuantizeError> {
    if source_path.is_dir() {
        std::fs::create_dir(output_path).map_err(packing_failure)?;
        return copy_drafter_folder(source_path, output_path);
    }
    std::fs::copy(source_path, output_path).map_err(packing_failure)?;
    Ok(())
}

fn report_stage(dependencies: &mut QuantizeDependencies<'_>, stage: &str, started_at: Instant) {
    let _ = writeln!(
        dependencies.stderr,
        "[quantize-drafter] {stage} (elapsed {}s)",
        started_at.elapsed().as_secs()
    );
}

/// One line: the packed file, its size against the source's, and the time.
fn report_summary(
    dependencies: &mut QuantizeDependencies<'_>,
    (source_weights_path, packed_weights_path): (&Path, &Path),
    started_at: Instant,
) -> Result<(), QuantizeError> {
    let source_bytes = std::fs::metadata(source_weights_path)
        .map_err(packing_failure)?
        .len();
    let packed_bytes = std::fs::metadata(packed_weights_path)
        .map_err(packing_failure)?
        .len();
    writeln!(
        dependencies.stdout,
        "wrote {}: {:.2} GB, from {:.2} GB, in {}s",
        packed_weights_path.display(),
        packed_bytes as f64 / BYTES_PER_GIGABYTE,
        source_bytes as f64 / BYTES_PER_GIGABYTE,
        started_at.elapsed().as_secs()
    )
    .map_err(QuantizeError::stdout_unwritable)
}

/// Named so `map_err` takes a function rather than a closure.
fn packing_failure(failure: impl std::fmt::Display) -> QuantizeError {
    QuantizeError::DrafterPacking {
        description: failure.to_string(),
    }
}
