//! Replaces text embeddings at `<|image_pad|>` positions with vision outputs.
//!
//! This is the multimodal splice between the vision and language graphs. It is
//! equivalent to the masked/scatter embedding replacement used by Qwen-VL model
//! wrappers, but operates on contiguous pad runs so each run becomes one MLX-C
//! `mlx_slice` plus `mlx_slice_update` graph operation. The cursor makes the
//! splice correct when prefill chunk boundaries split one image's pad run.

use astronomical_runtime_integration::{MlxArray, MlxRuntime};

use super::Qwen3_5MoEExecutionError;

/// Replaces image-pad token embeddings in one prefill chunk with ordered visual embeddings.
pub fn qwen3_5_moe_inject_visual_embeddings(
    runtime: &MlxRuntime,
    text_embeddings: &MlxArray,
    chunk_token_ids: &[u32],
    visual_embeddings: &MlxArray,
    starting_visual_embedding_index: usize,
    image_pad_token_id: u32,
) -> Result<(MlxArray, usize), Qwen3_5MoEExecutionError> {
    let text_embedding_shape = text_embeddings.shape();
    let visual_embedding_shape = visual_embeddings.shape();
    if text_embedding_shape.len() != 3
        || text_embedding_shape[0] != 1
        || text_embedding_shape[1] as usize != chunk_token_ids.len()
    {
        return Err(Qwen3_5MoEExecutionError::InvalidInput {
            description: "text embedding shape does not match the prefill chunk",
        });
    }
    if visual_embedding_shape.len() != 2
        || visual_embedding_shape[1] != text_embedding_shape[2]
        || visual_embeddings.dtype() != text_embeddings.dtype()
    {
        return Err(Qwen3_5MoEExecutionError::InvalidInput {
            description: "visual embeddings do not match the text embedding width and dtype",
        });
    }

    // Count before constructing updates so an undersized vision tensor fails
    // atomically rather than after partially assembling a lazy replacement graph.
    let image_pad_token_count = chunk_token_ids
        .iter()
        .filter(|token_id| **token_id == image_pad_token_id)
        .count();
    let ending_visual_embedding_index = starting_visual_embedding_index
        .checked_add(image_pad_token_count)
        .ok_or(Qwen3_5MoEExecutionError::InvalidInput {
            description: "visual embedding cursor overflowed",
        })?;
    if ending_visual_embedding_index > visual_embedding_shape[0] as usize {
        return Err(Qwen3_5MoEExecutionError::InvalidInput {
            description: "image-pad tokens exceed the available visual embeddings",
        });
    }

    // `reshape` creates a distinct MLX graph value with identical shape. Every
    // `slice_update` below returns another immutable graph value; no tensor is
    // mutated behind Rust aliases.
    let mut injected_embeddings = runtime.reshape(text_embeddings, &text_embedding_shape)?;
    let mut chunk_token_index = 0_usize;
    let mut visual_embedding_index = starting_visual_embedding_index;
    while chunk_token_index < chunk_token_ids.len() {
        if chunk_token_ids[chunk_token_index] != image_pad_token_id {
            chunk_token_index += 1;
            continue;
        }
        // Group adjacent pad tokens. Updating an entire run avoids one MLX graph
        // node per visual token, which matters for images with thousands of pads.
        let image_pad_run_start = chunk_token_index;
        while chunk_token_index < chunk_token_ids.len()
            && chunk_token_ids[chunk_token_index] == image_pad_token_id
        {
            chunk_token_index += 1;
        }
        let image_pad_run_end = chunk_token_index;
        let image_pad_run_length = image_pad_run_end - image_pad_run_start;
        let visual_embedding_run_end = visual_embedding_index + image_pad_run_length;
        // Vision rows and image-pad tokens are both ordered by image, then by
        // spatial-merge block. Therefore the next N rows replace the next N pad
        // positions without an index map.
        let visual_embedding_run = runtime.slice(
            visual_embeddings,
            &[usize_to_i32(visual_embedding_index)?, 0],
            &[
                usize_to_i32(visual_embedding_run_end)?,
                visual_embedding_shape[1],
            ],
            &[1, 1],
        )?;
        let batched_visual_embedding_run = runtime.expand_dims(&visual_embedding_run, 0)?;
        injected_embeddings = runtime.slice_update(
            &injected_embeddings,
            &batched_visual_embedding_run,
            &[0, usize_to_i32(image_pad_run_start)?, 0],
            &[1, usize_to_i32(image_pad_run_end)?, text_embedding_shape[2]],
            &[1, 1, 1],
        )?;
        visual_embedding_index = visual_embedding_run_end;
    }

    Ok((injected_embeddings, image_pad_token_count))
}

fn usize_to_i32(dimension_size: usize) -> Result<i32, Qwen3_5MoEExecutionError> {
    i32::try_from(dimension_size).map_err(|_conversion_error| {
        Qwen3_5MoEExecutionError::InvalidInput {
            description: "visual embedding index exceeds the MLX integer range",
        }
    })
}
