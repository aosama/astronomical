//! 3-axis (frame, height, width) rotary positional embedding for Qwen-Image-2.1.
//!
//! Derived from diffusers `QwenImage21Rope`. Text tokens advance a shared position on all
//! three axes; each image block freezes the frame axis at the position reached by the
//! preceding text and lays its tokens out on a zero-centered height/width grid, so a block's
//! spatial positions do not depend on where the block sits in the joint sequence.
//!
//! This is the highest-risk novel piece relative to FLUX.2 Klein's 4-axis RoPE, so the index
//! construction and the frequency tables are pure f64/f32 math with no runtime dependency —
//! the hermetic test compares the output against the diffusers Python reference bit-for-bit.

/// Positive-index table length (diffusers `arange(8192)`).
const POS_TABLE_LEN: usize = 8192;
/// Negative-index table length (diffusers `arange(1024).flip() * -1 - 1`, i.e. -1024..=-1).
const NEG_TABLE_LEN: usize = 1024;
/// Default rotary base, matching diffusers `theta=10000`.
const DEFAULT_THETA: f64 = 10_000.0;

/// One axis of the RoPE frequency table: `real`/`imag` each hold `table_len * per_pos` f32
/// values laid out row-major (one index position, then `per_pos` complex pairs).
struct AxisTable {
    real: Vec<f32>,
    imag: Vec<f32>,
    per_pos: usize,
}

impl AxisTable {
    /// Appends this axis's `per_pos` complex pairs for `value` to both output vectors.
    fn append_frequencies(
        &self,
        value: isize,
        real_values: &mut Vec<f32>,
        imaginary_values: &mut Vec<f32>,
    ) {
        let table_index = table_index(value);
        let start = table_index * self.per_pos;
        let end = start + self.per_pos;
        real_values.extend_from_slice(&self.real[start..end]);
        imaginary_values.extend_from_slice(&self.imag[start..end]);
    }
}

/// Map a sequence position to a frequency-table slot.
///
/// Positions `0..=8191` index the positive table directly; negative positions `-1..=-1024`
/// index the appended negative table, whose `j`-th entry holds value `-j - 1`.
#[must_use]
const fn table_index(value: isize) -> usize {
    if value >= 0 {
        value as usize
    } else {
        POS_TABLE_LEN + (-value - 1) as usize
    }
}

/// Build the zero-centered grid index `[-(n - n/2), n/2)` for one axis length `n`.
#[must_use]
fn centered_range(n: usize) -> Vec<isize> {
    let start = -(n as isize) + (n as isize) / 2;
    let end = (n as isize) / 2;
    (start..end).collect()
}

/// The 3-axis RoPE for Qwen-Image-2.1.
pub struct QwenImage21Rope {
    axes_dim: [usize; 3],
    tables: [AxisTable; 3],
}

impl Default for QwenImage21Rope {
    fn default() -> Self {
        Self::new(DEFAULT_THETA, [16, 56, 56])
    }
}

impl QwenImage21Rope {
    /// Create the RoPE for the given `theta` and `[frame, height, width]` axis dimensions.
    ///
    /// Each axis dimension must be even (frequencies pair real/imaginary halves).
    #[must_use]
    pub fn new(theta: f64, axes_dim: [usize; 3]) -> Self {
        assert!(
            axes_dim.iter().all(|d| *d > 0 && *d % 2 == 0),
            "each RoPE axis must be even and positive"
        );
        let tables = [
            build_axis_table(axes_dim[0], theta),
            build_axis_table(axes_dim[1], theta),
            build_axis_table(axes_dim[2], theta),
        ];
        Self { axes_dim, tables }
    }

    /// The `[frame, height, width]` axis dimensions this RoPE was built for.
    #[must_use]
    pub fn axes_dim(&self) -> [usize; 3] {
        self.axes_dim
    }

    /// Build the frame/height/width index arrays from image shapes and the image pad mask.
    ///
    /// Block boundaries come from `img_shapes` token counts, not from runs of `true` in the
    /// pad mask: two condition images adjacent with no text between them still form separate
    /// blocks. Returns arrays of length `seq_len`; height/width fall back to the frame index at
    /// text positions.
    #[must_use]
    pub fn construct_indices(
        &self,
        img_shapes: &[(usize, usize)],
        image_pad_mask: &[bool],
    ) -> (Vec<isize>, Vec<isize>, Vec<isize>) {
        let total_len = image_pad_mask.len();
        let mut frame_index = Vec::with_capacity(total_len);
        let mut height_index = Vec::with_capacity(total_len);
        let mut width_index = Vec::with_capacity(total_len);
        let mut image_height_index = Vec::new();
        let mut image_width_index = Vec::new();
        let mut cursor = 0usize;
        let mut position: isize = 0;

        for &(height, width) in img_shapes {
            let mut block_start = cursor;
            while block_start < total_len && !image_pad_mask[block_start] {
                block_start += 1;
            }
            let text_len = block_start - cursor;
            for offset in 0..text_len {
                frame_index.push(position + offset as isize);
            }
            position += text_len as isize;

            cursor = block_start + height * width;
            for _ in 0..(height * width) {
                // Every token in the block shares the frozen frame position.
                frame_index.push(position);
            }
            // The reference advances the shared position by the block's largest axis, so the
            // next block starts beyond this block's zero-centered grid extent on every axis.
            position += height.max(width) as isize;

            for height_coordinate in centered_range(height) {
                for _ in 0..width {
                    image_height_index.push(height_coordinate);
                }
            }
            for _ in 0..height {
                for width_coordinate in centered_range(width) {
                    image_width_index.push(width_coordinate);
                }
            }
        }

        if cursor < total_len {
            let remaining = total_len - cursor;
            for offset in 0..remaining {
                frame_index.push(position + offset as isize);
            }
        }

        // Height/width start as the frame index and are overwritten at image-token positions.
        height_index.extend_from_slice(&frame_index);
        width_index.extend_from_slice(&frame_index);
        let mut image_height_cursor = 0usize;
        let mut image_width_cursor = 0usize;
        for (position, &is_image) in image_pad_mask.iter().enumerate() {
            if is_image {
                height_index[position] = image_height_index[image_height_cursor];
                width_index[position] = image_width_index[image_width_cursor];
                image_height_cursor += 1;
                image_width_cursor += 1;
            }
        }

        (frame_index, height_index, width_index)
    }

    /// Compute the `[seq_len, head_dim / 2]` complex frequencies as `(real, imaginary)` f32
    /// vectors, each of length `seq_len * head_dim / 2`.
    #[must_use]
    pub fn frequencies(
        &self,
        img_shapes: &[(usize, usize)],
        image_pad_mask: &[bool],
    ) -> (Vec<f32>, Vec<f32>) {
        let (frame_index, height_index, width_index) =
            self.construct_indices(img_shapes, image_pad_mask);
        let head_half = (self.axes_dim[0] + self.axes_dim[1] + self.axes_dim[2]) / 2;
        let seq_len = frame_index.len();
        let mut real = Vec::with_capacity(seq_len * head_half);
        let mut imag = Vec::with_capacity(seq_len * head_half);

        for position in 0..seq_len {
            self.tables[0].append_frequencies(frame_index[position], &mut real, &mut imag);
            self.tables[1].append_frequencies(height_index[position], &mut real, &mut imag);
            self.tables[2].append_frequencies(width_index[position], &mut real, &mut imag);
        }
        (real, imag)
    }
}

/// Build the real/imaginary frequency table for one axis, matching diffusers `rope_params`:
/// `freqs = outer(index, theta**-arange(0, dim, 2) / dim)`, stored as `polar(1, freqs)`.
fn build_axis_table(dim: usize, theta: f64) -> AxisTable {
    let per_pos = dim / 2;
    let powers: Vec<f64> = (0..per_pos).map(|k| (2 * k) as f64 / dim as f64).collect();
    let inv: Vec<f64> = powers.iter().map(|&power| theta.powf(-power)).collect();

    let mut real = Vec::with_capacity((POS_TABLE_LEN + NEG_TABLE_LEN) * per_pos);
    let mut imag = Vec::with_capacity((POS_TABLE_LEN + NEG_TABLE_LEN) * per_pos);

    for index in 0..POS_TABLE_LEN {
        let index = index as f64;
        for &factor in &inv {
            let angle = index * factor;
            real.push(angle.cos() as f32);
            imag.push(angle.sin() as f32);
        }
    }
    // Negative table: the j-th entry holds value -j - 1.
    for j in 0..NEG_TABLE_LEN {
        let index = -(j as f64) - 1.0;
        for &factor in &inv {
            let angle = index * factor;
            real.push(angle.cos() as f32);
            imag.push(angle.sin() as f32);
        }
    }

    AxisTable {
        real,
        imag,
        per_pos,
    }
}

/// Test oracle: expose frequencies for the hermetic test without widening the public API.
#[doc(hidden)]
#[must_use]
pub fn frequencies_for_tests(
    rope: &QwenImage21Rope,
    img_shapes: &[(usize, usize)],
    image_pad_mask: &[bool],
) -> (Vec<f32>, Vec<f32>) {
    rope.frequencies(img_shapes, image_pad_mask)
}
