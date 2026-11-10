//! Collects the multi-token-prediction head as the requantizer walks the
//! source and writes it as one file at the end.
//!
//! The projections are encoded with stock min/max at 4 bits, never the
//! range search: OptiQ quantizes the head after its search encoder is out of
//! scope, and a head quantized any other way would not be the one its
//! runtime was tuned against. Everything else in the head keeps its stored
//! dtype and values; nothing is renamed or re-laid-out, because the head's
//! own loader reads the release's names.
//!
//! Ported from OptiQ's `_quantize_mtp` and `_unfuse_experts`
//! (`optiq/runtime/mtp_convert.py`).

use std::fs::File;
use std::path::Path;

use astronomical_runtime_integration::MlxRuntime;
use mlx_bridge::{MlxArray, MlxDtype};

use crate::error::QuantizationError;

use super::mtp_sidecar::{
    FusedMtpExpertTensor, MTP_SIDECAR_BITS, MTP_SIDECAR_RELATIVE_PATH, MtpSidecarSummary,
    fused_mtp_expert_tensor, is_mtp_quantize_target, unfused_mtp_expert_weight_name,
};

/// The head's tensors, held until the walk ends. A head is one small block;
/// holding it does not move the requantizer's single-tensor peak.
pub struct MtpSidecarWriter {
    group_elements: i32,
    tensors: Vec<(String, MlxArray)>,
}

impl MtpSidecarWriter {
    #[must_use]
    pub const fn new(group_elements: u32) -> Self {
        Self {
            group_elements: group_elements as i32,
            tensors: Vec::new(),
        }
    }

    /// Adds one source tensor of the head: a fused expert tensor is split
    /// per expert and each slice quantized, a projection is quantized, and
    /// anything else is kept as stored.
    pub fn add(
        &mut self,
        runtime: &MlxRuntime,
        tensor_name: &str,
        source_tensor: MlxArray,
    ) -> Result<(), QuantizationError> {
        let source_shape = source_tensor.shape();
        let mut unsigned_shape = Vec::with_capacity(source_shape.len());
        for extent in &source_shape {
            unsigned_shape.push((*extent).max(0) as usize);
        }
        match fused_mtp_expert_tensor(tensor_name, &unsigned_shape) {
            Some(FusedMtpExpertTensor::GateUp) => {
                self.add_fused_gate_up(runtime, tensor_name, &source_tensor)
            }
            Some(FusedMtpExpertTensor::Down) => {
                self.add_fused_down(runtime, tensor_name, &source_tensor)
            }
            None if is_mtp_quantize_target(tensor_name) => {
                self.add_quantized(runtime, tensor_name.to_owned(), &source_tensor)
            }
            None => {
                source_tensor.evaluate()?;
                self.tensors.push((tensor_name.to_owned(), source_tensor));
                Ok(())
            }
        }
    }

    /// `[experts, 2 * intermediate, hidden]`: each expert's lower half is
    /// its gate projection and its upper half its up projection.
    fn add_fused_gate_up(
        &mut self,
        runtime: &MlxRuntime,
        tensor_name: &str,
        fused_tensor: &MlxArray,
    ) -> Result<(), QuantizationError> {
        let fused_shape = fused_tensor.shape();
        let (expert_count, half_rows, columns) =
            (fused_shape[0], fused_shape[1] / 2, fused_shape[2]);
        for expert_index in 0..expert_count {
            for (projection, first_row) in [("gate_proj", 0), ("up_proj", half_rows)] {
                let expert_weight = expert_rows(
                    runtime,
                    fused_tensor,
                    expert_index,
                    first_row,
                    first_row + half_rows,
                    columns,
                )?;
                let weight_name = unfused_mtp_expert_weight_name(
                    tensor_name,
                    "gate_up_proj",
                    expert_index as usize,
                    projection,
                );
                self.add_quantized(runtime, weight_name, &expert_weight)?;
            }
        }
        Ok(())
    }

    /// `[experts, hidden, intermediate]`: one down projection per expert.
    fn add_fused_down(
        &mut self,
        runtime: &MlxRuntime,
        tensor_name: &str,
        fused_tensor: &MlxArray,
    ) -> Result<(), QuantizationError> {
        let fused_shape = fused_tensor.shape();
        for expert_index in 0..fused_shape[0] {
            let expert_weight = expert_rows(
                runtime,
                fused_tensor,
                expert_index,
                0,
                fused_shape[1],
                fused_shape[2],
            )?;
            let weight_name = unfused_mtp_expert_weight_name(
                tensor_name,
                "down_proj",
                expert_index as usize,
                "down_proj",
            );
            self.add_quantized(runtime, weight_name, &expert_weight)?;
        }
        Ok(())
    }

    /// Stock affine quantization of one projection, written as the packed
    /// weight with its scales and biases.
    fn add_quantized(
        &mut self,
        runtime: &MlxRuntime,
        weight_name: String,
        weight: &MlxArray,
    ) -> Result<(), QuantizationError> {
        let is_float = matches!(
            weight.dtype(),
            MlxDtype::Float16 | MlxDtype::BFloat16 | MlxDtype::Float32
        );
        let float_weight = if is_float {
            weight.retain()?
        } else {
            runtime.astype(weight, MlxDtype::BFloat16)?
        };
        let (packed_weight, scales, biases) =
            runtime.quantize_affine(&float_weight, self.group_elements, MTP_SIDECAR_BITS as i32)?;
        runtime.evaluate_arrays(&[&packed_weight, &scales, &biases])?;
        let module_name = weight_name
            .strip_suffix(".weight")
            .unwrap_or(&weight_name)
            .to_owned();
        self.tensors
            .push((format!("{module_name}.weight"), packed_weight));
        self.tensors.push((format!("{module_name}.scales"), scales));
        self.tensors.push((format!("{module_name}.biases"), biases));
        Ok(())
    }

    /// Writes `optiq/mtp.safetensors` under `output_directory` and returns
    /// what the config registration needs; `None` when the source had no
    /// head. The file carries no metadata, as OptiQ's does not.
    pub fn finish(
        self,
        runtime: &MlxRuntime,
        output_directory: &Path,
        layer_count: usize,
    ) -> Result<Option<MtpSidecarSummary>, QuantizationError> {
        if self.tensors.is_empty() {
            return Ok(None);
        }
        let sidecar_path = output_directory.join(MTP_SIDECAR_RELATIVE_PATH);
        let sidecar_directory = sidecar_path.parent().unwrap_or(output_directory);
        if let Err(create_error) = std::fs::create_dir_all(sidecar_directory) {
            return Err(sidecar_write_failure(output_directory, &create_error));
        }
        let sidecar_file = match File::create(&sidecar_path) {
            Ok(sidecar_file) => sidecar_file,
            Err(create_error) => {
                return Err(sidecar_write_failure(output_directory, &create_error));
            }
        };
        let mut named_arrays = Vec::with_capacity(self.tensors.len());
        for (tensor_name, tensor) in &self.tensors {
            named_arrays.push((tensor_name.as_str(), tensor));
        }
        if let Err(writer_error) = runtime.save_safetensors(sidecar_file, &named_arrays, &[]) {
            return Err(sidecar_write_failure(output_directory, &writer_error));
        }
        Ok(Some(MtpSidecarSummary {
            tensor_count: self.tensors.len(),
            bits: MTP_SIDECAR_BITS,
            group_size: self.group_elements as u32,
            layer_count,
        }))
    }
}

/// Rows `first_row..last_row` of one expert of a fused tensor, as a matrix.
fn expert_rows(
    runtime: &MlxRuntime,
    fused_tensor: &MlxArray,
    expert_index: i32,
    first_row: i32,
    last_row: i32,
    columns: i32,
) -> Result<MlxArray, QuantizationError> {
    let expert_slice = runtime.slice(
        fused_tensor,
        &[expert_index, first_row, 0],
        &[expert_index + 1, last_row, columns],
        &[1, 1, 1],
    )?;
    Ok(runtime.reshape(&expert_slice, &[last_row - first_row, columns])?)
}

fn sidecar_write_failure(
    output_directory: &Path,
    failure: &dyn std::error::Error,
) -> QuantizationError {
    QuantizationError::ArtifactWriteFailure {
        output_directory: output_directory.to_path_buf(),
        description: format!("the multi-token-prediction sidecar was not written: {failure}"),
    }
}
