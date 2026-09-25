//! Direct-MLX acceptance journey: the Qwen-Image-2.1 serving engine answers a worker-shaped
//! request end to end.
//!
//! This ignored journey exercises the production engine path — the one the inference worker
//! constructs from a model-family factory — rather than the pipeline directly: `load` advertises
//! the identity and memory floor, `start_generation` admits one REST-shaped command, each
//! `advance_generation` crosses exactly one render boundary, and the completion publishes a PNG
//! with request metadata. It then serves a second request on the same engine, proving the
//! per-request weight release and reload the native owner performs, and checks the
//! post-cleanup memory observation and the request attribution log the engine writes.

use std::time::Duration;

use astronomical_ipc_protocol::{ImageGenerationCommand, ImageGenerationSettings, RequestId};
use astronomical_model_serving::{
    ImageGenerationEngine, ImageGenerationEngineStep, QWEN_IMAGE_21_LICENSE_IDENTIFIER,
    QWEN_IMAGE_21_OFFICIAL_MODEL_ID, QWEN_IMAGE_21_PROVIDER_MODEL_ID,
    QwenImage21ArtifactProvenance, QwenImage21ImageEngine,
};

use crate::common::qwen_image_21::{
    JOURNEY_ALLOCATOR_CACHE_LIMIT_BYTES, artifact_directory, journey_effective_ceiling_bytes,
};

const ENGINE_JOURNEY_TIMEOUT: Duration = Duration::from_secs(115);
/// A prompt from the repo's mandated Romeo and Juliet source text.
const ENGINE_JOURNEY_PROMPT: &str =
    "It is the east, and Juliet is the sun. Golden dawn light over a balcony garden.";
const ENGINE_JOURNEY_WIDTH: u32 = 1_024;
const ENGINE_JOURNEY_HEIGHT: u32 = 1_024;
const ENGINE_JOURNEY_STEPS: u16 = 6;
const FIRST_REQUEST_SEED: u64 = 20260924;
const SECOND_REQUEST_SEED: u64 = 20260925;
/// The reviewed MLX conversion this journey pins; the catalog entry pins the same revision.
const PINNED_ARTIFACT_REVISION: &str = "4db4e8c0c0e7a1debf0320415bec8388e888494c";
/// One render crosses conditioning, noise, one boundary per denoising step, decode, and convert.
const MAXIMUM_ADVANCES_PER_REQUEST: usize = 64;
/// Minimum neighbour luma correlation a published image must clear to count as a picture.
const MINIMUM_NEIGHBOR_CORRELATION: f64 = 0.5;

#[ignore = "requires the Qwen-Image-2.1 artifact directory; set ASTRONOMICAL_QWEN_IMAGE_21_ARTIFACT_DIRECTORY"]
#[tokio::test]
async fn should_serve_two_image_generation_requests_through_the_engine() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    tokio::time::timeout(ENGINE_JOURNEY_TIMEOUT, async {
        let attribution_directory =
            tempfile::tempdir().expect("an attribution directory should be available");
        let attribution_log_path = attribution_directory
            .path()
            .join("qwen-image-21-engine-attribution.jsonl");
        let mut engine = QwenImage21ImageEngine::from_model_family_factory(
            artifact_directory(),
            QwenImage21ArtifactProvenance::new(
                QWEN_IMAGE_21_PROVIDER_MODEL_ID,
                PINNED_ARTIFACT_REVISION,
                QWEN_IMAGE_21_LICENSE_IDENTIFIER,
            ),
            journey_effective_ceiling_bytes(),
            JOURNEY_ALLOCATOR_CACHE_LIMIT_BYTES,
            true,
            attribution_log_path.clone(),
        );

        let loaded = engine
            .load()
            .expect("the engine should validate and advertise the real artifact");
        assert_eq!(loaded.model_id(), QWEN_IMAGE_21_OFFICIAL_MODEL_ID);
        assert_eq!(loaded.capabilities().maximum_steps, 40);
        assert_eq!(loaded.capabilities().dimension_multiple_pixels, 32);
        assert!(
            loaded.minimum_mlx_memory_ceiling_bytes() > 0,
            "the load must record the artifact's memory floor"
        );

        let first_completed = advance_until_completed(
            &mut engine,
            image_generation_command(11, FIRST_REQUEST_SEED),
            RequestId::new(11),
        );
        let first_png = &first_completed.generated_image.encoded_bytes;
        assert_eq!(first_completed.generated_image.mime_type, "image/png");
        assert!(
            first_png.starts_with(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]),
            "the published image must carry the PNG magic header"
        );
        assert!(
            first_png.len() > 1024,
            "a real PNG payload should be kilobytes"
        );
        assert_eq!(first_completed.result_metadata.seed, FIRST_REQUEST_SEED);
        assert_eq!(
            (
                first_completed.result_metadata.width_pixels,
                first_completed.result_metadata.height_pixels
            ),
            (ENGINE_JOURNEY_WIDTH, ENGINE_JOURNEY_HEIGHT)
        );
        assert_eq!(first_completed.result_metadata.steps, ENGINE_JOURNEY_STEPS);
        assert_eq!(first_completed.result_metadata.guidance_thousandths, 1_000);
        let first_decoded_png = image::load_from_memory(first_png)
            .expect("the published bytes should be one decodable PNG");
        assert_eq!(
            (first_decoded_png.width(), first_decoded_png.height()),
            (
                u32::from(ENGINE_JOURNEY_WIDTH),
                u32::from(ENGINE_JOURNEY_HEIGHT)
            )
        );
        let first_neighbor_correlation = horizontal_luma_correlation(
            first_decoded_png.to_rgb8().as_raw(),
            ENGINE_JOURNEY_WIDTH as usize,
            ENGINE_JOURNEY_HEIGHT as usize,
        );
        assert!(
            first_neighbor_correlation >= MINIMUM_NEIGHBOR_CORRELATION,
            "the published image must be spatially coherent, not noise: neighbour luma \
             correlation {first_neighbor_correlation:.4} is below the \
             {MINIMUM_NEIGHBOR_CORRELATION} floor"
        );

        // The native owner releases every component after a request and reloads for the next
        // one; serving a second distinct-seed request on the same engine proves that cycle.
        let second_completed = advance_until_completed(
            &mut engine,
            image_generation_command(12, SECOND_REQUEST_SEED),
            RequestId::new(12),
        );
        assert_eq!(second_completed.result_metadata.seed, SECOND_REQUEST_SEED);
        assert_eq!(second_completed.result_metadata.guidance_thousandths, 1_000);

        let post_cleanup_telemetry = engine
            .take_post_cleanup_memory_telemetry()
            .expect("each finalized request should expose its post-cleanup MLX observation");
        assert_eq!(
            post_cleanup_telemetry.allocator_cache_memory_bytes, 0,
            "request finalization must clear reclaimable allocator storage"
        );
        assert!(
            post_cleanup_telemetry.active_memory_bytes <= post_cleanup_telemetry.peak_memory_bytes
        );

        let attribution_text = std::fs::read_to_string(&attribution_log_path)
            .expect("the enabled attribution log should be written");
        assert!(
            attribution_text.lines().count() >= 2,
            "one load plus two requests should each write an attribution record"
        );
        assert!(
            attribution_text.contains("image_pipeline_construction")
                || attribution_text.contains("image_render_boundary"),
            "the attribution log must carry the engine's image-lane operations: {attribution_text}"
        );
    })
    .await
    .expect("the engine journey should finish within its 115 s budget");
}

struct CompletedRequest {
    generated_image: astronomical_ipc_protocol::GeneratedImage,
    result_metadata: astronomical_ipc_protocol::ImageGenerationResultMetadata,
}

fn image_generation_command(request_id: u64, seed: u64) -> ImageGenerationCommand {
    ImageGenerationCommand {
        request_id: RequestId::new(request_id),
        model: QWEN_IMAGE_21_OFFICIAL_MODEL_ID.to_owned(),
        prompt: ENGINE_JOURNEY_PROMPT.to_owned(),
        settings: ImageGenerationSettings {
            width_pixels: ENGINE_JOURNEY_WIDTH,
            height_pixels: ENGINE_JOURNEY_HEIGHT,
            steps: ENGINE_JOURNEY_STEPS,
            guidance_thousandths: 1_000,
            seed,
        },
    }
}

/// Starts one request and advances it one boundary at a time until it publishes.
fn advance_until_completed(
    engine: &mut QwenImage21ImageEngine,
    command: ImageGenerationCommand,
    request_id: RequestId,
) -> CompletedRequest {
    engine
        .start_generation(command)
        .expect("the official-envelope request should start");
    for _advance_index in 0..MAXIMUM_ADVANCES_PER_REQUEST {
        match engine
            .advance_generation(request_id)
            .expect("each render boundary should advance")
        {
            ImageGenerationEngineStep::Progress {
                phase: _,
                completed_steps,
                total_steps,
                ..
            } => {
                assert!(
                    completed_steps <= total_steps,
                    "progress must stay within the request's step envelope"
                );
            }
            ImageGenerationEngineStep::Completed {
                generated_image,
                result_metadata,
            } => {
                return CompletedRequest {
                    generated_image,
                    result_metadata,
                };
            }
        }
    }
    panic!("the request should complete within {MAXIMUM_ADVANCES_PER_REQUEST} boundaries");
}

/// Pearson correlation between neighbouring pixels' luma — the cheapest separator between a
/// picture and noise. A real render scores well above 0.8 at these dimensions; uncorrelated
/// noise scores near 0, so the floor sits far from both.
fn horizontal_luma_correlation(rgb_bytes: &[u8], width: usize, height: usize) -> f64 {
    let mut left_pixels = Vec::with_capacity(width * height);
    let mut right_pixels = Vec::with_capacity(width * height);
    for row_index in 0..height {
        let row_start = row_index * width * 3;
        for column_index in 0..width - 1 {
            left_pixels.push(luma_of(&rgb_bytes[row_start + column_index * 3..]));
            right_pixels.push(luma_of(&rgb_bytes[row_start + (column_index + 1) * 3..]));
        }
    }
    let left_mean = left_pixels.iter().sum::<f64>() / left_pixels.len() as f64;
    let right_mean = right_pixels.iter().sum::<f64>() / right_pixels.len() as f64;
    let mut covariance = 0.0;
    let mut left_variance = 0.0;
    let mut right_variance = 0.0;
    for (left, right) in left_pixels.iter().zip(right_pixels.iter()) {
        let left_offset = left - left_mean;
        let right_offset = right - right_mean;
        covariance += left_offset * right_offset;
        left_variance += left_offset * left_offset;
        right_variance += right_offset * right_offset;
    }
    covariance / (left_variance * right_variance).sqrt()
}

fn luma_of(rgb_triple: &[u8]) -> f64 {
    0.299 * f64::from(rgb_triple[0])
        + 0.587 * f64::from(rgb_triple[1])
        + 0.114 * f64::from(rgb_triple[2])
}
