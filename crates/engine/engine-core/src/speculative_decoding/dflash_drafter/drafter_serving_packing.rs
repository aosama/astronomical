//! A DFlash drafter checkpoint rewritten in the packing Astronomical serves
//! it in, for packages that ship the drafter already quantized.
//!
//! The engine quantizes a bf16 drafter's linears to 4 bits, group 64, when
//! it loads (`DrafterLinear::quantize_in_place`). This writes those same
//! arrays ahead of time, made by that same call, so a package that ships
//! them serves bit for bit the weights the engine would have made from the
//! bf16 file, at about a third of the download. Every other tensor (norms,
//! convolution kernels, the selector's codebooks) is kept as stored, as the
//! engine keeps it.

use std::path::Path;

use mlx_bridge::MlxArray;
use runtime_integration::{MlxRuntime, MlxRuntimeError};

use super::drafter_linear::DrafterLinear;

/// The suffix of a linear's weight in a drafter checkpoint.
const WEIGHT_SUFFIX: &str = ".weight";

/// Every tensor of the checkpoint at `checkpoint_path` under the name a packed drafter stores it
/// by: a linear's 4-bit codes as `<module>.weight` beside its
/// `<module>.scales` and `<module>.biases`, every other tensor unchanged.
/// A drafter's linears are exactly its two-dimensional `.weight` tensors:
/// its norms are one-dimensional, its convolution kernels three-dimensional,
/// and its codebooks carry no `.weight` suffix.
pub fn drafter_tensors_at_serving_packing(
    runtime: &MlxRuntime,
    checkpoint_path: &Path,
) -> Result<Vec<(String, MlxArray)>, MlxRuntimeError> {
    let checkpoint_header = crate::expert_paging::parse_safetensors_header(checkpoint_path)
        .map_err(|header_error| MlxRuntimeError::RuntimeOperation {
            operation: "read a drafter checkpoint's header",
            description: header_error.to_string(),
        })?;
    let checkpoint_file = std::fs::File::open(checkpoint_path).map_err(|open_error| {
        MlxRuntimeError::RuntimeOperation {
            operation: "open a drafter checkpoint",
            description: open_error.to_string(),
        }
    })?;
    let checkpoint = runtime.load_safetensors(checkpoint_file, None)?;
    let mut tensor_names: Vec<String> = checkpoint_header
        .tensor_entries
        .into_iter()
        .map(|tensor_entry| tensor_entry.tensor_name)
        .collect();
    tensor_names.sort();
    let mut packed_tensors = Vec::with_capacity(tensor_names.len() * 2);
    for tensor_name in tensor_names {
        let tensor = checkpoint.tensor(&tensor_name)?;
        match tensor_name.strip_suffix(WEIGHT_SUFFIX) {
            Some(module_name) if tensor.shape().len() == 2 => {
                packed_tensors.extend(packed_linear_tensors(runtime, module_name, &tensor)?);
            }
            _ => packed_tensors.push((tensor_name, tensor)),
        }
    }
    Ok(packed_tensors)
}

/// One linear quantized by the engine's own load-time call, named for the
/// packed checkpoint.
fn packed_linear_tensors(
    runtime: &MlxRuntime,
    module_name: &str,
    full_precision_weight: &MlxArray,
) -> Result<[(String, MlxArray); 3], MlxRuntimeError> {
    let mut linear = DrafterLinear::full_precision(runtime, full_precision_weight)?;
    linear.quantize_in_place(runtime, false)?;
    let DrafterLinear::Quantized {
        packed_weight,
        scales,
        biases,
    } = linear
    else {
        return Err(MlxRuntimeError::RuntimeOperation {
            operation: "pack a drafter linear at the serving packing",
            description: format!("{module_name} did not quantize to MLX's affine packing"),
        });
    };
    Ok([
        (format!("{module_name}{WEIGHT_SUFFIX}"), packed_weight),
        (format!("{module_name}.scales"), scales),
        (format!("{module_name}.biases"), biases),
    ])
}
