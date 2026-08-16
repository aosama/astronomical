//! One Laguna projection: unquantized rows or MLX affine packed rows.

use astronomical_runtime_integration::{MlxArray, MlxDtype, MlxRuntime};

use super::error::LagunaExecutionError;

/// Resident projection weights for one canonical linear module.
#[derive(Debug)]
pub(in crate::laguna) enum LagunaBoundLinear {
    Native {
        weight: MlxArray,
    },
    Affine {
        packed_weight: MlxArray,
        scales: MlxArray,
        biases: MlxArray,
        bits: i32,
        group_size: i32,
    },
}

impl LagunaBoundLinear {
    /// Returns the physical payload bytes owned by this bound projection.
    #[must_use]
    pub(in crate::laguna) fn payload_byte_count(&self) -> u64 {
        match self {
            Self::Native { weight } => u64::try_from(weight.byte_count()).unwrap_or(u64::MAX),
            Self::Affine {
                packed_weight,
                scales,
                biases,
                ..
            } => [packed_weight, scales, biases]
                .into_iter()
                .map(|array| u64::try_from(array.byte_count()).unwrap_or(u64::MAX))
                .fold(0, u64::saturating_add),
        }
    }

    /// Materializes storage so an evaluated fused projection no longer retains its sources.
    pub(in crate::laguna) fn materialize_storage(
        &self,
        runtime: &MlxRuntime,
    ) -> Result<(), LagunaExecutionError> {
        match self {
            Self::Native { weight } => runtime.evaluate_arrays(&[weight])?,
            Self::Affine {
                packed_weight,
                scales,
                biases,
                ..
            } => runtime.evaluate_arrays(&[packed_weight, scales, biases])?,
        }
        Ok(())
    }

    /// Applies `activation @ weight_transpose` for native or affine storage.
    pub(in crate::laguna) fn project(
        &self,
        runtime: &MlxRuntime,
        activations: &MlxArray,
    ) -> Result<MlxArray, LagunaExecutionError> {
        match self {
            Self::Native { weight } => {
                let transposed_weight = runtime.transpose_axes(weight, &[1, 0])?;
                Ok(runtime.matmul(activations, &transposed_weight)?)
            }
            Self::Affine {
                packed_weight,
                scales,
                biases,
                bits,
                group_size,
            } => Ok(runtime.quantized_matmul_affine(
                activations,
                packed_weight,
                scales,
                biases,
                true,
                *group_size,
                *bits,
            )?),
        }
    }

    /// Selects stacked expert rows inside the matrix product.
    pub(in crate::laguna) fn project_gathered(
        &self,
        runtime: &MlxRuntime,
        activations: &MlxArray,
        selected_indices: &MlxArray,
        are_indices_sorted: bool,
    ) -> Result<MlxArray, LagunaExecutionError> {
        match self {
            Self::Native { weight } => {
                let transposed_weight = runtime.transpose_axes(weight, &[0, 2, 1])?;
                Ok(runtime.gather_dense_matmul(
                    activations,
                    &transposed_weight,
                    None,
                    Some(selected_indices),
                    are_indices_sorted,
                )?)
            }
            Self::Affine {
                packed_weight,
                scales,
                biases,
                bits,
                group_size,
            } => Ok(runtime.gather_quantized_matmul_affine(
                activations,
                packed_weight,
                scales,
                biases,
                None,
                Some(selected_indices),
                true,
                *group_size,
                *bits,
                are_indices_sorted,
            )?),
        }
    }

    /// Concatenates matching affine gate and up output rows into one projection.
    pub(in crate::laguna) fn fuse_matching_affine_output_rows(
        runtime: &MlxRuntime,
        gate: &Self,
        up: &Self,
    ) -> Result<Option<Self>, LagunaExecutionError> {
        let (
            Self::Affine {
                packed_weight: gate_weight,
                scales: gate_scales,
                biases: gate_biases,
                bits: gate_bits,
                group_size: gate_group_size,
            },
            Self::Affine {
                packed_weight: up_weight,
                scales: up_scales,
                biases: up_biases,
                bits: up_bits,
                group_size: up_group_size,
            },
        ) = (gate, up)
        else {
            return Ok(None);
        };
        if gate_bits != up_bits || gate_group_size != up_group_size {
            return Ok(None);
        }
        let output_row_axis = match gate_weight.shape().len() {
            // Ordinary dense/shared projections use `[out, packed_in]`.
            2 => 0,
            // Stacked routed projections use `[experts, out, packed_in]`.
            3 => 1,
            _ => return Ok(None),
        };
        Ok(Some(Self::Affine {
            packed_weight: runtime.concatenate_axis(&[gate_weight, up_weight], output_row_axis)?,
            scales: runtime.concatenate_axis(&[gate_scales, up_scales], output_row_axis)?,
            biases: runtime.concatenate_axis(&[gate_biases, up_biases], output_row_axis)?,
            bits: *gate_bits,
            group_size: *gate_group_size,
        }))
    }

    pub(in crate::laguna) fn split_fused_gate_up(
        runtime: &MlxRuntime,
        fused_output: &MlxArray,
    ) -> Result<(MlxArray, MlxArray), LagunaExecutionError> {
        let output_shape = fused_output.shape();
        let Some(last_dimension_index) = output_shape.len().checked_sub(1) else {
            return Err(LagunaExecutionError::invalid_geometry(
                "fused gate/up output must have a trailing dimension",
            ));
        };
        let output_dimension = output_shape[last_dimension_index];
        if output_dimension <= 0 || output_dimension % 2 != 0 {
            return Err(LagunaExecutionError::invalid_geometry(
                "fused gate/up output dimension must be positive and even",
            ));
        }
        let projection_dimension = output_dimension / 2;
        let gate_starts = vec![0; output_shape.len()];
        let mut gate_stops = output_shape.clone();
        gate_stops[last_dimension_index] = projection_dimension;
        let mut up_starts = gate_starts.clone();
        up_starts[last_dimension_index] = projection_dimension;
        let slice_strides = vec![1; output_shape.len()];
        Ok((
            runtime.slice(fused_output, &gate_starts, &gate_stops, &slice_strides)?,
            runtime.slice(fused_output, &up_starts, &output_shape, &slice_strides)?,
        ))
    }
}

pub(in crate::laguna) fn require_supported_affine_profile(
    bits: i32,
    group_size: i32,
) -> Result<(), LagunaExecutionError> {
    if !matches!(bits, 2 | 3 | 4 | 5 | 6 | 8) || !matches!(group_size, 32 | 64 | 128) {
        return Err(LagunaExecutionError::invalid_geometry(
            "affine projections must use an MLX-supported bit width and group size",
        ));
    }
    Ok(())
}

pub(in crate::laguna) fn is_floating_weight(dtype: MlxDtype) -> bool {
    matches!(
        dtype,
        MlxDtype::Float16 | MlxDtype::BFloat16 | MlxDtype::Float32
    )
}
