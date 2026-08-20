//! Artifact-derived memory geometry and reference-compatible four-axis positions.

use std::collections::BTreeMap;

use astronomical_ipc_protocol::ImageGenerationCapabilities;
use astronomical_runtime_integration::{MlxArray, MlxRuntime};

use super::super::{
    Flux2KleinMemoryGeometry, Flux2KleinPackedLatentLayout, ValidatedFlux2KleinArtifact,
};
use super::FLUX2_KLEIN_CONDITIONING_SEQUENCE_LENGTH;

const MAXIMUM_IMAGE_EDGE_PIXELS: u32 = 1_024;
pub(super) fn official_capabilities() -> ImageGenerationCapabilities {
    ImageGenerationCapabilities {
        minimum_width_pixels: 64,
        maximum_width_pixels: MAXIMUM_IMAGE_EDGE_PIXELS,
        minimum_height_pixels: 64,
        maximum_height_pixels: MAXIMUM_IMAGE_EDGE_PIXELS,
        dimension_multiple_pixels: 16,
        maximum_steps: 4,
        maximum_guidance_thousandths: 1_000,
        output_mime_types: vec!["image/png".to_owned()],
    }
}

pub(super) fn memory_geometry(
    artifact: &ValidatedFlux2KleinArtifact,
) -> Result<Flux2KleinMemoryGeometry, String> {
    let text_encoder_payload_bytes = artifact.text_encoder_inventory().payload_bytes();
    let largest_text_owner_bytes = largest_text_streamed_owner_bytes(artifact)?;
    let mut transformer_block_payload_bytes = vec![0_u64; 25];
    let mut largest_tensor_payload_bytes = 0_u64;
    for descriptor in artifact.transformer_inventory().descriptors() {
        largest_tensor_payload_bytes = largest_tensor_payload_bytes.max(descriptor.payload_bytes());
        if let Some(block_index) = transformer_block_index(descriptor.tensor_name()) {
            transformer_block_payload_bytes[block_index] = transformer_block_payload_bytes
                [block_index]
                .checked_add(descriptor.payload_bytes())
                .ok_or_else(|| "transformer block accounting overflowed".to_owned())?;
        }
    }
    let vae_payload_bytes = artifact
        .vae_inventory()
        .vae_decoder_owned_payload_bytes()
        .ok_or_else(|| "VAE decoder payload accounting overflowed".to_owned())?;
    for descriptor in artifact
        .vae_inventory()
        .descriptors()
        .iter()
        .filter(|descriptor| descriptor.is_owned_by_vae_decoder())
    {
        largest_tensor_payload_bytes = largest_tensor_payload_bytes.max(descriptor.payload_bytes());
    }
    let maximum_pixel_count = u64::from(MAXIMUM_IMAGE_EDGE_PIXELS).pow(2);
    let host_rgb_bytes = maximum_pixel_count * 3;
    let maximum_png_bytes = host_rgb_bytes + u64::from(MAXIMUM_IMAGE_EDGE_PIXELS) + 65_536;
    Ok(Flux2KleinMemoryGeometry {
        text_encoder_payload_bytes,
        transformer_payload_bytes: artifact.transformer_inventory().payload_bytes(),
        transformer_block_payload_bytes,
        vae_payload_bytes,
        largest_component_load_page_bytes: largest_text_owner_bytes
            .max(largest_tensor_payload_bytes),
        conditioning_bytes: 512 * 7_680 * 2 + 512,
        latent_state_bytes: maximum_pixel_count / 256 * 128 * 2,
        denoising_workspace_bytes: 256_000_000,
        vae_workspace_bytes: maximum_pixel_count * 512 * 4,
        host_rgb_bytes,
        maximum_png_bytes,
        maximum_base64_bytes: maximum_png_bytes.div_ceil(3) * 4,
    })
}

fn largest_text_streamed_owner_bytes(
    artifact: &ValidatedFlux2KleinArtifact,
) -> Result<u64, String> {
    let mut layer_payload_bytes = BTreeMap::<usize, u64>::new();
    let mut embedding_payload_bytes = None;
    let mut largest_layer_payload_bytes = 0_u64;
    for descriptor in artifact.text_encoder_inventory().descriptors() {
        if let Some(layer_index) = text_layer_index(descriptor.tensor_name()) {
            let layer_total = layer_payload_bytes.entry(layer_index).or_default();
            *layer_total = layer_total
                .checked_add(descriptor.payload_bytes())
                .ok_or_else(|| "text layer payload accounting overflowed".to_owned())?;
            largest_layer_payload_bytes = largest_layer_payload_bytes.max(*layer_total);
        } else if descriptor.tensor_name() == "model.embed_tokens.weight" {
            embedding_payload_bytes = Some(descriptor.payload_bytes());
        }
    }
    let embedding_payload_bytes = embedding_payload_bytes
        .ok_or_else(|| "text encoder inventory has no embedding payload".to_owned())?;
    (largest_layer_payload_bytes > 0)
        .then_some(largest_layer_payload_bytes)
        .ok_or_else(|| "text encoder inventory has no layer payload".to_owned())?
        .checked_add(embedding_payload_bytes)
        .ok_or_else(|| "streamed text owner payload accounting overflowed".to_owned())
}

fn text_layer_index(tensor_name: &str) -> Option<usize> {
    tensor_name
        .strip_prefix("model.layers.")?
        .split_once('.')?
        .0
        .parse()
        .ok()
}

pub(super) fn signed_shape<const DIMENSIONS: usize>(
    shape: &[usize; DIMENSIONS],
) -> Result<Vec<i32>, String> {
    shape
        .iter()
        .map(|dimension| {
            i32::try_from(*dimension).map_err(|_| "MLX image shape exceeds i32".to_owned())
        })
        .collect()
}

pub(super) fn build_position_ids(
    runtime: &MlxRuntime,
    layout: Flux2KleinPackedLatentLayout,
) -> Result<(MlxArray, MlxArray), String> {
    let packed_shape = layout.packed_spatial_shape();
    let mut image_ids = Vec::with_capacity(packed_shape[1] * packed_shape[2] * 4);
    for row_index in 0..packed_shape[1] {
        for column_index in 0..packed_shape[2] {
            image_ids.extend_from_slice(&[0.0, row_index as f32, column_index as f32, 0.0]);
        }
    }
    let mut text_ids = Vec::with_capacity(FLUX2_KLEIN_CONDITIONING_SEQUENCE_LENGTH * 4);
    for token_index in 0..FLUX2_KLEIN_CONDITIONING_SEQUENCE_LENGTH {
        text_ids.extend_from_slice(&[0.0, 0.0, 0.0, token_index as f32]);
    }
    let image_token_count = i32::try_from(packed_shape[1] * packed_shape[2])
        .map_err(|_| "image position count exceeds i32".to_owned())?;
    let text_token_count = i32::try_from(FLUX2_KLEIN_CONDITIONING_SEQUENCE_LENGTH)
        .map_err(|_| "text position count exceeds i32".to_owned())?;
    Ok((
        runtime
            .array_from_f32(&image_ids, &[image_token_count, 4])
            .map_err(|error| error.to_string())?,
        runtime
            .array_from_f32(&text_ids, &[text_token_count, 4])
            .map_err(|error| error.to_string())?,
    ))
}

fn transformer_block_index(tensor_name: &str) -> Option<usize> {
    if let Some(suffix) = tensor_name.strip_prefix("transformer_blocks.") {
        return suffix.split('.').next()?.parse().ok();
    }
    tensor_name
        .strip_prefix("single_transformer_blocks.")?
        .split('.')
        .next()?
        .parse::<usize>()
        .ok()
        .map(|index| index + 5)
}
