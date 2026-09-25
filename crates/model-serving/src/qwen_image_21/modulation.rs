//! Target-token labelling and `causal_condition` modulation-row selection for Qwen-Image-2.1.
//!
//! The final adaptive norm (`QwenImage21AdaLayerNormContinuous`) modulates each token from a timestep
//! embedding. Under `causal_condition` the conditioning tensor carries `batch_size + 1` rows: the real
//! timestep row per sample plus one extra `t = 0` row. Text and condition-image tokens take the `t = 0`
//! row (making their modulation step-independent, which is what enables KV-cache reuse), while only the
//! **target image's** tokens take their own sample's row.
//!
//! This is subtle, high-risk bookkeeping (an off-by-one here silently corrupts every denoise step), so it
//! is pure integer logic with no runtime dependency and is tested hermetically against the diffusers
//! reference.

/// Mark the target image's tokens in the joint sequence.
///
/// Returns `true` only at the last image block's positions (the target image), `false` at every text and
/// condition-image position. This is the `target_token_mask` output of diffusers
/// [`build_token_metadata`], computed from the same `img_shapes` token counts that drive `image_ids`, so
/// adjacent condition images stay distinct and only the final block is the target.
pub fn build_target_token_mask(
    img_shapes: &[(usize, usize)],
    image_pad_mask: &[bool],
) -> Vec<bool> {
    let seq_len = image_pad_mask.len();
    let mut target = vec![false; seq_len];
    if img_shapes.is_empty() {
        return target;
    }

    let mut image_positions = Vec::with_capacity(seq_len);
    for (position, &is_image) in image_pad_mask.iter().enumerate() {
        if is_image {
            image_positions.push(position);
        }
    }

    let total_image_tokens: usize = img_shapes.iter().map(|&(h, w)| h * w).sum();
    debug_assert!(
        total_image_tokens == image_positions.len(),
        "img_shapes accounts for {total_image_tokens} image tokens but image_pad_mask marks {}",
        image_positions.len()
    );

    // The target image is the last block; its tokens are the final `target_len` image positions.
    let &(target_h, target_w) = img_shapes.last().expect("img_shapes is non-empty");
    let target_len = target_h * target_w;
    let start = image_positions.len().saturating_sub(target_len);
    for &position in &image_positions[start..] {
        target[position] = true;
    }

    target
}

/// Row index (per sample, per token) that `causal_condition` modulation selects.
///
/// Models the `_select_modulation_rows` output as an index map over the `[0, batch_size]` parameter
/// rows, where row `batch_size` is the trailing `t = 0` row:
/// - No `target_token_mask` (`None`): every token uses its own sample's real row (`sample`).
/// - With a `target_token_mask`: a target-image token uses its sample's row; every other token uses the
///   shared `t = 0` row (`batch_size`).
///
/// Returning indices rather than gathered values keeps this boundary pure and independently testable; the
/// gather over the real modulation tensor is a one-line consumer step once weights are loaded.
#[must_use]
pub fn causal_modulation_row_map(
    target_token_mask: Option<&[bool]>,
    batch_size: usize,
    seq_len: usize,
) -> Vec<Vec<usize>> {
    let zero_row = batch_size; // the trailing `t = 0` row in the `batch_size + 1` tensor
    match target_token_mask {
        None => (0..batch_size)
            .map(|sample| vec![sample; seq_len])
            .collect(),
        Some(mask) => (0..batch_size)
            .map(|sample| {
                (0..seq_len)
                    .map(|token| if mask[token] { sample } else { zero_row })
                    .collect()
            })
            .collect(),
    }
}
