//! Direct-MLX acceptance journey: a complete Qwen-Image-2.1 text-to-image render.
//!
//! This ignored journey loads the whole pipeline — artifact tokenizer, Qwen3-VL text encoder,
//! block-causal denoising transformer, flow-matching schedule, and VAE decoder — from the
//! reviewed artifact and renders one image end to end from a real prompt. It proves the full
//! chain executes and produces a structurally valid image: correct dimensions, a real PNG
//! header, and non-uniform pixel content.
//!
//! The journey runs at production image dimensions (1024×1024) with six denoising steps and one
//! fixed seed so it fits the bounded 120-second budget; the render example binary drives the same
//! pipeline with more denoising steps. Numeric equality with the diffusers reference remains the end-to-end
//! frontier — there is no torch on this machine to produce a reference image.
//!
//! Because no reference image is available, the journey pins spatial coherence as the structural
//! proxy for "a real picture": a render whose neighbouring pixels are uncorrelated is noise
//! whatever its dimensions are. That check is what catches sampling bugs the size-only assertions
//! wave through — pairing the transformer with the training-timestep scale instead of the sigma
//! renders noise while still producing a full-size, non-uniform, valid PNG.
//!
//! Setting `ASTRONOMICAL_QWEN_IMAGE_21_RENDER_OUTPUT` additionally writes the rendered PNG to
//! that path, so the same journey can hand a real image to a human.

use std::path::PathBuf;
use std::time::Duration;

use astronomical_model_serving::{QwenImage21Pipeline, QwenImage21RenderRequest};
use astronomical_runtime_integration::MlxRuntime;

use crate::common::qwen_image_21::{artifact_directory, shared_journey_runtime};

const RENDER_JOURNEY_TIMEOUT: Duration = Duration::from_secs(115);
/// A prompt from the repo's mandated Romeo and Juliet source text.
const RENDER_PROMPT: &str =
    "It is the east, and Juliet is the sun. Golden dawn light over a balcony garden.";
const RENDER_WIDTH: usize = 1024;
const RENDER_HEIGHT: usize = 1024;
const RENDER_STEPS: usize = 6;
const RENDER_SEED: u64 = 20260924;
const RENDER_OUTPUT_ENV: &str = "ASTRONOMICAL_QWEN_IMAGE_21_RENDER_OUTPUT";
/// Minimum neighbour luma correlation a render must clear to count as a picture rather than noise.
const MINIMUM_NEIGHBOR_CORRELATION: f64 = 0.5;

#[ignore = "requires the Qwen-Image-2.1 artifact directory; set ASTRONOMICAL_QWEN_IMAGE_21_ARTIFACT_DIRECTORY"]
#[tokio::test]
async fn should_render_a_text_to_image_png_from_the_full_pipeline() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    tokio::time::timeout(RENDER_JOURNEY_TIMEOUT, async {
        let runtime: MlxRuntime = shared_journey_runtime();
        let pipeline = QwenImage21Pipeline::load(&runtime, &artifact_directory())
            .expect("the full Qwen-Image-2.1 pipeline should load from the real artifact");

        let mut observed_steps = 0_usize;
        let mut progress_reporter = |step_index: usize, total_steps: usize| {
            observed_steps += 1;
            assert!(
                step_index < total_steps,
                "the progress observer must report in-range steps"
            );
        };

        let request = QwenImage21RenderRequest {
            prompt: RENDER_PROMPT.to_owned(),
            width: RENDER_WIDTH,
            height: RENDER_HEIGHT,
            num_inference_steps: RENDER_STEPS,
            seed: RENDER_SEED,
        };
        let rendered = pipeline
            .render(&runtime, &request, &mut progress_reporter)
            .expect("the pipeline should render the prompt end to end");
        assert_eq!(
            (rendered.width, rendered.height),
            (RENDER_WIDTH, RENDER_HEIGHT)
        );
        assert_eq!(
            rendered.rgb_bytes.len(),
            RENDER_WIDTH * RENDER_HEIGHT * 3,
            "the render must produce one RGB byte triple per pixel"
        );
        assert_eq!(
            observed_steps, RENDER_STEPS,
            "every denoising step must report progress"
        );

        let png_bytes = rendered
            .to_png_bytes()
            .expect("the rendered pixels should encode as PNG");
        assert!(
            png_bytes.starts_with(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]),
            "the encoded image must carry the PNG magic header"
        );
        assert!(
            png_bytes.len() > 1024,
            "a real PNG payload should be kilobytes, not a stub"
        );
        assert!(
            rendered
                .rgb_bytes
                .iter()
                .any(|&byte| byte != rendered.rgb_bytes[0]),
            "a rendered image must not be one uniform color"
        );
        let neighbor_correlation =
            horizontal_luma_correlation(&rendered.rgb_bytes, RENDER_WIDTH, RENDER_HEIGHT);
        assert!(
            neighbor_correlation >= MINIMUM_NEIGHBOR_CORRELATION,
            "a rendered image must be spatially coherent, not noise: neighbour luma correlation \
             {neighbor_correlation:.4} is below the {MINIMUM_NEIGHBOR_CORRELATION} floor"
        );

        if let Some(output_path) = std::env::var_os(RENDER_OUTPUT_ENV) {
            let output_path = PathBuf::from(output_path);
            if let Some(parent) = output_path.parent() {
                std::fs::create_dir_all(parent).expect("the render output parent should create");
            }
            std::fs::write(&output_path, &png_bytes)
                .unwrap_or_else(|error| panic!("the render output should write: {error}"));
        }
    })
    .await
    .expect("the render journey should finish within its 115 s budget");
}

/// Pearson correlation between neighbouring pixels' luma — the cheapest separator between a
/// picture and noise. A real render scores well above 0.8 at these dimensions; uncorrelated noise
/// scores near 0, so the floor sits far from both.
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
