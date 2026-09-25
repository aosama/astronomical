//! Block-causal attention mask and prefix segmentation for Qwen-Image-2.1.
//!
//! The joint text/image sequence is **block-causal**: `(q >= kv)` everywhere, except each image
//! block (every condition image and the target image) is internally bidirectional so its tokens
//! may attend to one another regardless of order. Text tokens are strictly causal. This is the
//! second-highest-risk novel component after the 3-axis RoPE, so the logic is pure integer
//! bookkeeping with no runtime dependency and is tested hermetically against the diffusers oracle.
//!
//! Block boundaries come from `image_ids` counts, not from runs of `true` in the pad mask: two
//! condition images adjacent with no text between them still form separate blocks, otherwise they
//! would attend to each other bidirectionally.

/// Build the per-token image-block id used by the block-causal mask.
///
/// Returns `-1` at text positions and a unique non-negative id per image block, in the same form
/// diffusers [`build_token_metadata`] produces. Block lengths come from `img_shapes`
/// `(height, width)` counts, so adjacent condition images stay distinct blocks.
pub fn build_image_ids(img_shapes: &[(usize, usize)], image_pad_mask: &[bool]) -> Vec<i32> {
    let seq_len = image_pad_mask.len();
    let mut image_ids = vec![-1i32; seq_len];
    let mut image_positions =
        Vec::with_capacity(img_shapes.iter().map(|&(h, w)| h * w).sum::<usize>());
    for (position, &is_image) in image_pad_mask.iter().enumerate() {
        if is_image {
            image_positions.push(position);
        }
    }

    let mut block_id = 0i32;
    let mut pos_cursor = 0usize;
    for &(height, width) in img_shapes {
        let block_len = height * width;
        for _ in 0..block_len {
            // The `debug_assert` guards the invariant that `img_shapes` accounts for exactly the
            // marked image tokens; a mismatch here is a construction bug, not a runtime case.
            debug_assert!(
                pos_cursor < image_positions.len(),
                "img_shapes accounts for more image tokens than image_pad_mask marks"
            );
            image_ids[image_positions[pos_cursor]] = block_id;
            pos_cursor += 1;
        }
        block_id += 1;
    }

    image_ids
}

/// Build the effective `[seq_len, seq_len]` block-causal mask, row-major (`q * seq_len + kv`).
///
/// `allowed(q, kv) = ((q >= kv) OR same_image_block(q, kv)) AND key_valid[kv]`, where
/// `same_image_block = (image_ids[q] == image_ids[kv]) AND image_ids[q] >= 0`. Passing `None` for
/// `key_valid` keeps every key valid (no encoder padding). Positions masked out of `key_valid` are
/// excluded as *keys* so right-padded prompts cannot be attended to, while staying valid as
/// queries so their rows are never fully masked.
pub fn build_block_causal_mask(image_ids: &[i32], key_valid: Option<&[bool]>) -> Vec<bool> {
    let seq_len = image_ids.len();
    let keys_valid: Vec<bool> = match key_valid {
        Some(valid) => valid.to_vec(),
        None => vec![true; seq_len],
    };

    let mut mask = Vec::with_capacity(seq_len * seq_len);
    for (q, &q_id) in image_ids.iter().enumerate() {
        let block = q_id >= 0;
        for (kv, &kv_id) in image_ids.iter().enumerate() {
            let same_image_block = block && (q_id == kv_id);
            let allowed = ((q >= kv) || same_image_block) && keys_valid[kv];
            mask.push(allowed);
        }
    }
    mask
}

/// Split the prefix `[0, prefix_len)` into `(start, end, is_text)` runs of equal `image_ids`.
///
/// This is the block-causal structure in the form the segment-based attention processor consumes
/// it: consecutive tokens sharing an `image_ids` value form one run, and a run is text when the
/// shared id is `-1`.
pub fn prefix_segments(image_ids: &[i32], prefix_len: usize) -> Vec<(usize, usize, bool)> {
    let mut segments = Vec::new();
    let mut start = 0usize;
    for index in 1..=prefix_len {
        if index == prefix_len || image_ids[index] != image_ids[start] {
            segments.push((start, index, image_ids[start] < 0));
            start = index;
        }
    }
    segments
}
