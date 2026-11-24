//! Converts a tower from upstream's layout to the MLX layout loaders expect.
//!
//! Upstream checkpoints name towers `model.visual.*` and store the patch embedding
//! as PyTorch's channels-first convolution, `[out, in, depth, height, width]`. MLX
//! names them `vision_tower.*` and wants channels last, `[out, depth, height,
//! width, in]`. For Qwen3.5 those are the only two differences, confirmed by comparing
//! all 333 tensors of a converted tower with an independent conversion, so a rename
//! and one reorder are enough and the other tensors keep their exact bytes. Other
//! families may need more, and have not been checked.

const MLX_TOWER_ROOT: &str = "vision_tower";
const UPSTREAM_TOWER_ROOT: &str = "visual";
const MODEL_WRAPPER_PREFIX: &str = "model.";
const PATCH_EMBEDDING_SUFFIX: &str = ".patch_embed.proj.weight";
const PATCH_EMBEDDING_DIMENSIONS: usize = 5;

/// The MLX name of a tower tensor, or `None` for a layout this cannot convert.
#[must_use]
pub fn mlx_tower_tensor_name(source_tensor_name: &str) -> Option<String> {
    let unwrapped_name = source_tensor_name
        .strip_prefix(MODEL_WRAPPER_PREFIX)
        .unwrap_or(source_tensor_name);
    let (root_segment, tower_path) = unwrapped_name.split_once('.')?;
    if root_segment == MLX_TOWER_ROOT || root_segment == UPSTREAM_TOWER_ROOT {
        return Some(format!("{MLX_TOWER_ROOT}.{tower_path}"));
    }
    None
}

/// Whether this tensor is an upstream patch embedding that needs reordering. A
/// tower already named for MLX is taken to be in MLX's layout already.
#[must_use]
pub fn is_channels_first_patch_embedding(source_tensor_name: &str, tensor_shape: &[u64]) -> bool {
    let unwrapped_name = source_tensor_name
        .strip_prefix(MODEL_WRAPPER_PREFIX)
        .unwrap_or(source_tensor_name);
    unwrapped_name.starts_with(UPSTREAM_TOWER_ROOT)
        && unwrapped_name.ends_with(PATCH_EMBEDDING_SUFFIX)
        && tensor_shape.len() == PATCH_EMBEDDING_DIMENSIONS
}

/// `[out, in, depth, height, width]` as `[out, depth, height, width, in]`.
#[must_use]
pub fn channels_last_shape(channels_first_shape: &[u64]) -> Vec<u64> {
    let mut channels_last = channels_first_shape.to_vec();
    channels_last[1..].rotate_left(1);
    channels_last
}

/// Reorders a channels-first patch embedding's bytes without touching any value.
#[must_use]
pub fn channels_first_to_channels_last(
    source_bytes: &[u8],
    channels_first_shape: &[u64],
    element_bytes: usize,
) -> Vec<u8> {
    let output_count = channels_first_shape[0] as usize;
    let input_count = channels_first_shape[1] as usize;
    let depth_count = channels_first_shape[2] as usize;
    let height_count = channels_first_shape[3] as usize;
    let width_count = channels_first_shape[4] as usize;
    let mut reordered_bytes = Vec::with_capacity(source_bytes.len());
    for output_index in 0..output_count {
        for depth_index in 0..depth_count {
            for height_index in 0..height_count {
                for width_index in 0..width_count {
                    for input_index in 0..input_count {
                        let element_index = (((output_index * input_count + input_index)
                            * depth_count
                            + depth_index)
                            * height_count
                            + height_index)
                            * width_count
                            + width_index;
                        let first_byte = element_index * element_bytes;
                        reordered_bytes.extend_from_slice(
                            &source_bytes[first_byte..first_byte + element_bytes],
                        );
                    }
                }
            }
        }
    }
    reordered_bytes
}

#[cfg(test)]
#[path = "tower_layout_tests.rs"]
mod tests;
