//! Affine-quantized projection weights in the form Astronomical's checkpoints
//! use, and their re-encodings for Core ML's compressed-weight operations.
//!
//! A checkpoint stores `w = scale · q + bias` with one fp16 `scale` and `bias`
//! per block of weights and a code `q` per weight; Astronomical's 4-bit
//! checkpoints use blocks of one output row by 64 input channels
//! ([`QuantizationLayout::CHECKPOINT_INT4`]). Other layouts exist here to map
//! which forms the Neural Engine accepts.
//!
//! Core ML offers two encodings of the same codes:
//!
//! - `constexpr_blockwise_shift_scale` computes `scale · (q − offset)`, so only
//!   the bias is re-expressed, as `offset = −bias / scale`; storing that offset
//!   in fp16 is a rounding the checkpoint does not have, which
//!   [`AffineQuantizedWeights::shift_scale_dequantized_fp16`] exposes.
//! - `constexpr_lut_to_dense` indexes a per-block table by the unchanged code;
//!   a block's codes can only take `2^bits` values, so tables holding exactly
//!   those values reproduce the checkpoint bit for bit.

use thiserror::Error;

use crate::weights::half_precision::{f16_bits_to_f32, f32_to_f16_bits};
use crate::weights::row_major_block::row_block;

const SMALLEST_NORMAL_FP16_BITS: u16 = 0x0400;

/// Code width and block shape of an affine quantization.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QuantizationLayout {
    /// 4 or 8.
    pub code_bits: u8,
    pub block_output_rows: usize,
    pub block_input_columns: usize,
}

impl QuantizationLayout {
    /// Astronomical's 4-bit checkpoints: one scale per 64 inputs of a row.
    pub const CHECKPOINT_INT4: Self = Self {
        code_bits: 4,
        block_output_rows: 1,
        block_input_columns: 64,
    };

    /// Largest code; codes span `0..=maximum_code`. Widths beyond eight bits
    /// saturate rather than overflow, and quantization rejects them anyway.
    #[must_use]
    pub fn maximum_code(self) -> u8 {
        ((1_u16 << self.code_bits.min(8)) - 1) as u8
    }

    /// Distinct codes, the length of one lookup table.
    #[must_use]
    pub fn code_count(self) -> usize {
        usize::from(self.maximum_code()) + 1
    }
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum AffineQuantizationError {
    #[error("{output_channels} × {input_channels} weights expected, got {actual}")]
    WeightCountMismatch {
        output_channels: usize,
        input_channels: usize,
        actual: usize,
    },
    #[error("{output_channels} × {input_channels} weights cannot be cut into {layout:?} blocks")]
    UnevenBlocks {
        output_channels: usize,
        input_channels: usize,
        layout: QuantizationLayout,
    },
}

/// Row-major `[output_channels][input_channels]` codes with row-major
/// `[output_channels / block rows][input_channels / block columns]` fp16
/// scales and biases.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AffineQuantizedWeights {
    output_channels: usize,
    input_channels: usize,
    layout: QuantizationLayout,
    codes: Vec<u8>,
    fp16_scales: Vec<u16>,
    fp16_biases: Vec<u16>,
}

impl AffineQuantizedWeights {
    /// Quantizes each block to its own `[min, max]` range: `bias = min`,
    /// `scale = (max − min) / maximum_code`. This is the affine form the
    /// checkpoints use; the measurement compares engines against these codes'
    /// own dequantization, so matching any particular quantizer's
    /// tie-breaking is unnecessary.
    pub fn quantize(
        row_major_weights: &[f32],
        output_channels: usize,
        input_channels: usize,
        layout: QuantizationLayout,
    ) -> Result<Self, AffineQuantizationError> {
        validate_layout(
            row_major_weights.len(),
            output_channels,
            input_channels,
            layout,
        )?;
        let mut quantized_weights = Self {
            output_channels,
            input_channels,
            layout,
            codes: Vec::with_capacity(row_major_weights.len()),
            fp16_scales: Vec::new(),
            fp16_biases: Vec::new(),
        };
        let (block_minima, block_maxima) = quantized_weights.block_ranges(row_major_weights);
        for (minimum_weight, maximum_weight) in block_minima.iter().zip(&block_maxima) {
            let (fp16_scale, fp16_bias) =
                block_scale_and_bias(*minimum_weight, *maximum_weight, layout.maximum_code());
            quantized_weights.fp16_scales.push(fp16_scale);
            quantized_weights.fp16_biases.push(fp16_bias);
        }
        for (weight_index, weight) in row_major_weights.iter().enumerate() {
            let block_index = quantized_weights.block_index_of(weight_index);
            quantized_weights.codes.push(nearest_code(
                *weight,
                quantized_weights.fp16_scales[block_index],
                quantized_weights.fp16_biases[block_index],
                layout.maximum_code(),
            ));
        }
        Ok(quantized_weights)
    }

    #[must_use]
    pub fn output_channels(&self) -> usize {
        self.output_channels
    }

    #[must_use]
    pub fn input_channels(&self) -> usize {
        self.input_channels
    }

    #[must_use]
    pub fn layout(&self) -> QuantizationLayout {
        self.layout
    }

    /// `[output_channels / block rows, input_channels / block columns]`.
    #[must_use]
    pub fn block_grid(&self) -> [usize; 2] {
        [
            self.output_channels / self.layout.block_output_rows,
            self.input_channels / self.layout.block_input_columns,
        ]
    }

    #[must_use]
    pub fn codes(&self) -> &[u8] {
        &self.codes
    }

    #[must_use]
    pub fn fp16_scales(&self) -> &[u16] {
        &self.fp16_scales
    }

    /// The checkpoint's own weights, `fp16(scale · q + bias)`, row-major.
    #[must_use]
    pub fn dequantized_fp16(&self) -> Vec<u16> {
        let mut fp16_weights = Vec::with_capacity(self.codes.len());
        for (weight_index, code) in self.codes.iter().enumerate() {
            let block_index = self.block_index_of(weight_index);
            fp16_weights.push(checkpoint_weight_fp16(
                self.fp16_scales[block_index],
                self.fp16_biases[block_index],
                *code,
            ));
        }
        fp16_weights
    }

    /// `code_count` fp16 entries per block, entry `k` being the checkpoint
    /// weight for code `k`, so a lookup indexed by the unchanged codes
    /// reproduces [`Self::dequantized_fp16`] exactly.
    #[must_use]
    pub fn block_lookup_tables_fp16(&self) -> Vec<u16> {
        let mut lookup_tables =
            Vec::with_capacity(self.fp16_scales.len() * self.layout.code_count());
        for (fp16_scale, fp16_bias) in self.fp16_scales.iter().zip(&self.fp16_biases) {
            for code in 0..=self.layout.maximum_code() {
                lookup_tables.push(checkpoint_weight_fp16(*fp16_scale, *fp16_bias, code));
            }
        }
        lookup_tables
    }

    /// `fp16(−bias / scale)` per block, the offset Core ML subtracts.
    #[must_use]
    pub fn shift_scale_offsets_fp16(&self) -> Vec<u16> {
        let mut fp16_offsets = Vec::with_capacity(self.fp16_scales.len());
        for (fp16_scale, fp16_bias) in self.fp16_scales.iter().zip(&self.fp16_biases) {
            fp16_offsets.push(shift_scale_offset(*fp16_scale, *fp16_bias));
        }
        fp16_offsets
    }

    /// The weights Core ML reconstructs, `fp16(scale · (q − offset))`.
    #[must_use]
    pub fn shift_scale_dequantized_fp16(&self) -> Vec<u16> {
        let fp16_offsets = self.shift_scale_offsets_fp16();
        let mut fp16_weights = Vec::with_capacity(self.codes.len());
        for (weight_index, code) in self.codes.iter().enumerate() {
            let block_index = self.block_index_of(weight_index);
            let scale = f16_bits_to_f32(self.fp16_scales[block_index]);
            let offset = f16_bits_to_f32(fp16_offsets[block_index]);
            fp16_weights.push(f32_to_f16_bits(scale * (f32::from(*code) - offset)));
        }
        fp16_weights
    }

    /// Input channels `first_input_channel .. + part_width` of every output
    /// row, for one part of a contraction split. Both bounds must sit on
    /// quantization-block boundaries, because a block's scale belongs to
    /// exactly one part.
    #[must_use]
    pub fn contraction_part(&self, first_input_channel: usize, part_width: usize) -> Self {
        let block_columns = self.layout.block_input_columns;
        assert!(
            first_input_channel.is_multiple_of(block_columns)
                && part_width.is_multiple_of(block_columns)
                && first_input_channel + part_width <= self.input_channels,
            "the part must cover whole quantization blocks inside the row"
        );
        let [_, blocks_per_row] = self.block_grid();
        let first_block = first_input_channel / block_columns;
        let part_block_count = part_width / block_columns;
        Self {
            output_channels: self.output_channels,
            input_channels: part_width,
            layout: self.layout,
            codes: row_block(
                &self.codes,
                self.input_channels,
                first_input_channel,
                part_width,
            ),
            fp16_scales: row_block(
                &self.fp16_scales,
                blocks_per_row,
                first_block,
                part_block_count,
            ),
            fp16_biases: row_block(
                &self.fp16_biases,
                blocks_per_row,
                first_block,
                part_block_count,
            ),
        }
    }

    fn block_index_of(&self, weight_index: usize) -> usize {
        let output_channel = weight_index / self.input_channels;
        let input_channel = weight_index % self.input_channels;
        let [_, blocks_per_row] = self.block_grid();
        (output_channel / self.layout.block_output_rows) * blocks_per_row
            + input_channel / self.layout.block_input_columns
    }

    /// Per-block minimum and maximum, in block order.
    fn block_ranges(&self, row_major_weights: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let [block_rows, blocks_per_row] = self.block_grid();
        let mut block_minima = vec![f32::INFINITY; block_rows * blocks_per_row];
        let mut block_maxima = vec![f32::NEG_INFINITY; block_rows * blocks_per_row];
        for (weight_index, weight) in row_major_weights.iter().enumerate() {
            let block_index = self.block_index_of(weight_index);
            block_minima[block_index] = block_minima[block_index].min(*weight);
            block_maxima[block_index] = block_maxima[block_index].max(*weight);
        }
        (block_minima, block_maxima)
    }
}

fn validate_layout(
    weight_count: usize,
    output_channels: usize,
    input_channels: usize,
    layout: QuantizationLayout,
) -> Result<(), AffineQuantizationError> {
    if output_channels.checked_mul(input_channels) != Some(weight_count) {
        return Err(AffineQuantizationError::WeightCountMismatch {
            output_channels,
            input_channels,
            actual: weight_count,
        });
    }
    let is_supported_code_width = matches!(layout.code_bits, 4 | 8);
    let tiles_the_matrix = layout.block_output_rows > 0
        && layout.block_input_columns > 0
        && output_channels > 0
        && output_channels.is_multiple_of(layout.block_output_rows)
        && input_channels.is_multiple_of(layout.block_input_columns);
    if !is_supported_code_width || !tiles_the_matrix {
        return Err(AffineQuantizationError::UnevenBlocks {
            output_channels,
            input_channels,
            layout,
        });
    }
    Ok(())
}

fn block_scale_and_bias(minimum_weight: f32, maximum_weight: f32, maximum_code: u8) -> (u16, u16) {
    // A constant block still needs a non-zero scale so `−bias / scale` is
    // finite; the smallest normal fp16 keeps every code at zero.
    let scale = ((maximum_weight - minimum_weight) / f32::from(maximum_code))
        .max(f16_bits_to_f32(SMALLEST_NORMAL_FP16_BITS));
    (f32_to_f16_bits(scale), f32_to_f16_bits(minimum_weight))
}

fn nearest_code(weight: f32, fp16_scale: u16, fp16_bias: u16, maximum_code: u8) -> u8 {
    let scale = f16_bits_to_f32(fp16_scale);
    let bias = f16_bits_to_f32(fp16_bias);
    ((weight - bias) / scale)
        .round()
        .clamp(0.0, f32::from(maximum_code)) as u8
}

fn checkpoint_weight_fp16(fp16_scale: u16, fp16_bias: u16, code: u8) -> u16 {
    f32_to_f16_bits(f16_bits_to_f32(fp16_scale) * f32::from(code) + f16_bits_to_f32(fp16_bias))
}

fn shift_scale_offset(fp16_scale: u16, fp16_bias: u16) -> u16 {
    f32_to_f16_bits(-f16_bits_to_f32(fp16_bias) / f16_bits_to_f32(fp16_scale))
}

#[cfg(test)]
#[path = "affine_quantized_tests.rs"]
mod tests;
