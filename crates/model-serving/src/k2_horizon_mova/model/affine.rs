//! Affine quantized linear modules for stacked MLX weights.

use astronomical_runtime_integration::{MlxArray, MlxRuntime, MlxRuntimeError};

/// One affine quantized linear (weight, scales, biases) plus optional dense bias.
#[derive(Debug)]
pub struct K2HorizonMoVAAffineLinear {
    packed_weight: MlxArray,
    scales: MlxArray,
    biases: MlxArray,
    bits: i32,
    group_size: i32,
    dense_bias: Option<MlxArray>,
}

impl K2HorizonMoVAAffineLinear {
    #[must_use]
    pub fn new(
        packed_weight: MlxArray,
        scales: MlxArray,
        biases: MlxArray,
        bits: u32,
        group_size: u32,
        dense_bias: Option<MlxArray>,
    ) -> Self {
        Self {
            packed_weight,
            scales,
            biases,
            bits: bits as i32,
            group_size: group_size as i32,
            dense_bias,
        }
    }

    pub fn project(
        &self,
        runtime: &MlxRuntime,
        activations: &MlxArray,
    ) -> Result<MlxArray, MlxRuntimeError> {
        let projected = runtime.quantized_matmul_affine(
            activations,
            &self.packed_weight,
            &self.scales,
            &self.biases,
            true,
            self.group_size,
            self.bits,
        )?;
        match &self.dense_bias {
            Some(dense_bias) => runtime.add(&projected, dense_bias),
            None => Ok(projected),
        }
    }

    pub fn project_without_dense_bias(
        &self,
        runtime: &MlxRuntime,
        activations: &MlxArray,
    ) -> Result<MlxArray, MlxRuntimeError> {
        runtime.quantized_matmul_affine(
            activations,
            &self.packed_weight,
            &self.scales,
            &self.biases,
            true,
            self.group_size,
            self.bits,
        )
    }

    #[must_use]
    pub fn packed_weight(&self) -> &MlxArray {
        &self.packed_weight
    }
    #[must_use]
    pub fn scales(&self) -> &MlxArray {
        &self.scales
    }
    #[must_use]
    pub fn biases(&self) -> &MlxArray {
        &self.biases
    }
    #[must_use]
    pub const fn bits(&self) -> i32 {
        self.bits
    }
    #[must_use]
    pub const fn group_size(&self) -> i32 {
        self.group_size
    }
    #[must_use]
    pub fn dense_bias(&self) -> Option<&MlxArray> {
        self.dense_bias.as_ref()
    }

    /// Concatenates matching gate and up output rows into one affine module.
    pub fn fuse_matching_output_rows(
        runtime: &MlxRuntime,
        gate: &Self,
        up: &Self,
    ) -> Result<Option<Self>, MlxRuntimeError> {
        Self::fuse_output_rows(runtime, &[gate, up])
    }

    /// Concatenates any number of same-geometry affine modules along the
    /// output-row axis so one quantized projection can produce all of their
    /// outputs. Rows are packed independently, so the fused route is
    /// bit-identical to projecting the parts separately.
    pub fn fuse_output_rows(
        runtime: &MlxRuntime,
        parts: &[&Self],
    ) -> Result<Option<Self>, MlxRuntimeError> {
        let Some(first) = parts.first() else {
            return Ok(None);
        };
        if parts
            .iter()
            .any(|part| part.bits != first.bits || part.group_size != first.group_size)
        {
            return Ok(None);
        }
        let output_row_axis = match first.packed_weight.shape().len() {
            2 => 0,
            3 => 1,
            _ => return Ok(None),
        };
        let packed_parts = parts
            .iter()
            .map(|part| &part.packed_weight)
            .collect::<Vec<_>>();
        let scale_parts = parts.iter().map(|part| &part.scales).collect::<Vec<_>>();
        let bias_parts = parts.iter().map(|part| &part.biases).collect::<Vec<_>>();
        let packed_weight = runtime.concatenate_axis(&packed_parts, output_row_axis)?;
        let scales = runtime.concatenate_axis(&scale_parts, output_row_axis)?;
        let biases = runtime.concatenate_axis(&bias_parts, output_row_axis)?;
        runtime.evaluate_arrays(&[&packed_weight, &scales, &biases])?;
        Ok(Some(Self {
            packed_weight,
            scales,
            biases,
            bits: first.bits,
            group_size: first.group_size,
            dense_bias: None,
        }))
    }

    /// Splits one fused projection's output on the trailing axis into
    /// consecutive ranges with the caller's row counts.
    pub fn split_projection_output(
        runtime: &MlxRuntime,
        fused_output: &MlxArray,
        row_counts: &[usize],
    ) -> Result<Vec<MlxArray>, MlxRuntimeError> {
        let output_shape = fused_output.shape();
        let last_dimension_index = output_shape.len().saturating_sub(1);
        let slice_strides = vec![1; output_shape.len()];
        let mut parts = Vec::with_capacity(row_counts.len());
        let mut start_row = 0_usize;
        for row_count in row_counts {
            let stop_row = start_row + row_count;
            let mut starts = vec![0; output_shape.len()];
            let mut stops = output_shape.clone();
            starts[last_dimension_index] = start_row as i32;
            stops[last_dimension_index] = stop_row as i32;
            parts.push(runtime.slice(fused_output, &starts, &stops, &slice_strides)?);
            start_row = stop_row;
        }
        Ok(parts)
    }

    /// Splits a fused gate/up projection on the trailing hidden axis.
    pub fn split_fused_output(
        runtime: &MlxRuntime,
        fused_output: &MlxArray,
    ) -> Result<(MlxArray, MlxArray), MlxRuntimeError> {
        let output_shape = fused_output.shape();
        let last_dimension_index = output_shape.len().saturating_sub(1);
        let output_dimension = output_shape.get(last_dimension_index).copied().unwrap_or(0);
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

    /// Logical tensor payload held by this affine module.
    #[must_use]
    pub fn payload_bytes(&self) -> u64 {
        mlx_array_payload_bytes(&self.packed_weight)
            .saturating_add(mlx_array_payload_bytes(&self.scales))
            .saturating_add(mlx_array_payload_bytes(&self.biases))
            .saturating_add(
                self.dense_bias
                    .as_ref()
                    .map(mlx_array_payload_bytes)
                    .unwrap_or(0),
            )
    }
}

fn mlx_array_payload_bytes(array: &MlxArray) -> u64 {
    u64::try_from(array.byte_count()).unwrap_or(u64::MAX)
}
