//! An image as a text request carries it: preprocessed on the CPU, before a
//! family's vision tower reads it.

/// Patch grid produced by the Qwen3VL image processor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageGrid {
    pub temporal_patch_count: u32,
    pub height_patch_count: u32,
    pub width_patch_count: u32,
}

/// CPU-side image processor output before the MLX vision tower consumes it.
///
/// Each row is one flattened patch in C,T,H,W order. Rows themselves use the
/// spatial-merger block order expected by the vision tower. A still image is
/// duplicated conceptually across the temporal patch dimension: both temporal
/// slots read the same source RGB pixel, matching upstream `np.repeat`.
#[derive(Debug, Clone, PartialEq)]
pub struct ProcessedImage {
    /// SHA-256 of the exact encoded PNG, JPEG, or WebP bytes received at the request boundary.
    pub encoded_image_sha256: [u8; 32],
    pub pixel_values: Vec<f32>,
    pub pixel_values_row_count: usize,
    pub pixel_values_column_count: usize,
    pub image_grid: ImageGrid,
    pub resized_height_pixels: u32,
    pub resized_width_pixels: u32,
    pub image_token_count_after_spatial_merge: usize,
}
