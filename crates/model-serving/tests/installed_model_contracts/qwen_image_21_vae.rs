//! Direct-MLX acceptance journeys for the native Qwen-Image-2.1 VAE decoder.
//!
//! These ignored journeys load the reviewed artifact's real `vae/model.safetensors` (238
//! unquantized F32 tensors, ~1.35 GB) and drive the ported decoder end to end on the GPU. They
//! prove execution-level properties: the ported graph loads, builds in bounded stages, decodes a
//! latent grid to the geometry the family's own constants predict, respects the reference's
//! `[-1, 1]` clamp, and is deterministic across repeated decodes. Pixel equality with the diffusers
//! reference needs a reference decode this machine cannot produce, so the end-to-end render
//! journey carries the image-level acceptance and these journeys pin the decoder's own contract.
//!
//! Resolves the artifact through `ASTRONOMICAL_QWEN_IMAGE_21_ARTIFACT_DIRECTORY` — the same
//! env-var acceptance pattern as the text-conditioning journeys — so no developer path is
//! hardcoded. The wired-memory limit is computed from the artifact's own weight file size plus a
//! fixed activation headroom, so the journey adapts to any artifact revision instead of any
//! particular machine.

use std::fs::File;
use std::time::Duration;

use astronomical_model_serving::{
    QWEN_IMAGE_21_LATENT_CHANNEL_COUNT, QWEN_IMAGE_21_OUTPUT_CHANNEL_COUNT, QwenImage21VaeDecoder,
    decoded_pixel_dimensions,
};
use astronomical_runtime_integration::MlxRuntime;

use crate::common::qwen_image_21::{component_weights_path, shared_journey_runtime};

/// The whole journey must finish inside this budget; the wrapper fails it otherwise.
const VAE_DECODE_JOURNEY_TIMEOUT: Duration = Duration::from_secs(115);
/// Latent side of the journey's decode: 8 → the smallest spatial grid every up block doubles.
const LATENT_SIDE: usize = 8;
/// The channel counts come from the family's geometry constants, and the decoded side comes from
/// `decoded_pixel_dimensions`, so this journey checks the decoder against the same arithmetic the
/// pipeline uses to size an image rather than against restated numbers.
const LATENT_CHANNEL_COUNT: usize = QWEN_IMAGE_21_LATENT_CHANNEL_COUNT;
const OUTPUT_CHANNEL_COUNT: usize = QWEN_IMAGE_21_OUTPUT_CHANNEL_COUNT;

/// A deterministic, spatially varying latent: every position and channel differs, so a
/// spatially constant bug in the decoder cannot hide.
fn journey_latents(runtime: &MlxRuntime) -> astronomical_runtime_integration::MlxArray {
    let position_count = LATENT_SIDE * LATENT_SIDE * LATENT_CHANNEL_COUNT;
    let latent_values: Vec<f32> = (0..position_count)
        .map(|index| ((index % 1024) as f32 / 1024.0 - 0.5) * 4.0)
        .collect();
    runtime
        .array_from_f32(
            &latent_values,
            &[
                1,
                LATENT_SIDE as i32,
                LATENT_SIDE as i32,
                LATENT_CHANNEL_COUNT as i32,
            ],
        )
        .expect("the journey latents should materialize")
}

#[ignore = "requires the Qwen-Image-2.1 artifact directory; set ASTRONOMICAL_QWEN_IMAGE_21_ARTIFACT_DIRECTORY"]
#[tokio::test]
async fn should_decode_a_latent_grid_into_deterministic_clamped_rgba_pixels() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    tokio::time::timeout(VAE_DECODE_JOURNEY_TIMEOUT, async {
        let runtime = shared_journey_runtime();
        let decoder = QwenImage21VaeDecoder::load(
            &runtime,
            File::open(component_weights_path("vae")).expect("the VAE weights should open"),
        )
        .expect("the Qwen-Image-2.1 VAE decoder should load from the real artifact");

        let latents = journey_latents(&runtime);
        let (expected_height, expected_width) = decoded_pixel_dimensions(LATENT_SIDE, LATENT_SIDE);
        let decoded = decoder
            .decode_image_latents(&runtime, &latents)
            .expect("the VAE decoder should decode the journey latents");
        assert_eq!(
            decoded.shape(),
            vec![
                1,
                expected_height as i32,
                expected_width as i32,
                OUTPUT_CHANNEL_COUNT as i32,
            ],
            "the decoder must expand the latent grid by the spatial compression ratio and emit RGBA"
        );
        let pixels = decoded.to_vec_f32().expect("the pixels should materialize");
        assert_eq!(pixels.len(), decoded_pixel_count());
        assert!(
            pixels.iter().all(|pixel| pixel.is_finite()),
            "every decoded pixel must be finite"
        );
        assert!(
            pixels.iter().all(|pixel| (-1.0..=1.0).contains(pixel)),
            "the decode must clamp its output to the reference's [-1, 1] range"
        );
        assert!(
            pixels.iter().any(|pixel| *pixel != pixels[0]),
            "a spatially varying latent must not decode to one constant value"
        );

        let redecoded = decoder
            .decode_image_latents(&runtime, &latents)
            .expect("the second decode should succeed");
        let repixels = redecoded
            .to_vec_f32()
            .expect("the redecoded pixels should materialize");
        assert_eq!(pixels, repixels, "repeated decodes must be deterministic");
    })
    .await
    .expect("the VAE decode journey should finish within its 115 s budget");
}

/// `height × width × channels` for the journey's decode — the pixel vector the decoder must fill.
fn decoded_pixel_count() -> usize {
    let (height, width) = decoded_pixel_dimensions(LATENT_SIDE, LATENT_SIDE);
    height * width * OUTPUT_CHANNEL_COUNT
}

#[ignore = "requires the Qwen-Image-2.1 artifact directory; set ASTRONOMICAL_QWEN_IMAGE_21_ARTIFACT_DIRECTORY"]
#[tokio::test]
async fn should_reject_a_latent_grid_with_the_wrong_channel_count() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    tokio::time::timeout(VAE_DECODE_JOURNEY_TIMEOUT, async {
        let runtime = shared_journey_runtime();
        let decoder = QwenImage21VaeDecoder::load(
            &runtime,
            File::open(component_weights_path("vae")).expect("the VAE weights should open"),
        )
        .expect("the Qwen-Image-2.1 VAE decoder should load from the real artifact");

        // Three RGB channels is the classic VAE shape; this artifact decodes 64 latent channels,
        // and feeding it anything else must fail loudly instead of silently misreading memory.
        let malformed = runtime
            .array_from_f32(&vec![0.0; 8 * 8 * 3], &[1, 8, 8, 3])
            .expect("the malformed latents should materialize");
        let rejection = decoder.decode_image_latents(&runtime, &malformed);
        assert!(
            rejection.is_err(),
            "a latent grid with 3 channels must be rejected, not decoded"
        );
    })
    .await
    .expect("the rejection journey should finish within its 115 s budget");
}
