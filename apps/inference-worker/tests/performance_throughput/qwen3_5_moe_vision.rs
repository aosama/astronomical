//! Measures the serving throughput of the resident 35B sparse-MoE model's
//! VISION path over the supervisor IPC boundary: prompt-processing (prefill)
//! tokens per second and decode tokens per second, both server-attributed
//! from the worker's generation performance log (never client wall clock).
//!
//! The measured completion carries a real image whose grid yields 9,216
//! visual tokens (3,072 x 3,072 pixels, patches of 16, spatial merge of 2),
//! so the journey exercises the full vision tower plus the language-model
//! prefill of the visual embeddings — not just text with a token image. With
//! the continuation instruction and the Romeo and Juliet opening the full
//! input is ~10,300 tokens, clearing the >=10,000-token input requirement;
//! each measured completion acquires ~1,000 output tokens.
//!
//! The image is built in-test as a solid-color PNG, so the fixture stays
//! source-controlled and tiny while the patch grid stays large; visual token
//! counts depend only on the resized grid, not the pixel content.
//!
//! The test case follows AGENTS.md "Instructions for Performance Throughput
//! Tests": the ~1,000-token warmup completion (small image, ~100 output
//! tokens) is discarded so first-use JIT kernel compilation — including the
//! vision tower's — never inflates the measured prefill, the SSD (persistent
//! prompt) cache is disabled, and the measured completion's server-attributed
//! rates persist to the durable historical record under the "vision" journey
//! kind, which delineates them from the text journey's lines in the same
//! shared history log. The journey asserts nothing about the measured rates.
//!
//! This is a laptop-only journey: it loads real weights into wired GPU
//! memory, so it is `#[ignore]`d and not wired into CI. Invoke it through
//! scripts/run-performance-throughput.sh (invocation only) or directly with
//! `cargo test --release -p astronomical-inference-worker --features
//! astronomical-inference-worker/performance_throughput --test
//! performance_throughput_tests -- --ignored --exact`.

use std::io::Cursor;

use astronomical_ipc_protocol::ChatImageInput;
use image::{DynamicImage, ImageFormat, Rgb, RgbImage};

use crate::performance_throughput::historical_record::ThroughputJourneyKind;
use crate::performance_throughput::support::{self as throughput_support, ThroughputJourney};
use crate::support;

/// The small warmup image: 448 x 448 pixels (a multiple of the 32-pixel
/// patch-merge block) yielding 49 visual tokens, enough to spin up the vision
/// tower's first-use kernels before the measured run.
const WARMUP_IMAGE_SIDE_PIXELS: u32 = 448;

/// The measured image: 3,072 x 3,072 pixels (a multiple of the 32-pixel
/// patch-merge block) yielding a 192 x 192 patch grid that spatial-merges
/// into 9,216 visual tokens.
const MEASURED_IMAGE_SIDE_PIXELS: u32 = 3_072;

/// The short warmup: a ~1,000-token Romeo and Juliet opening, continued for a
/// short passage, that spins up first-use JIT kernels before the measured run.
const WARMUP_INPUT_INSTRUCTION: &str =
    "Describe the supplied image, then continue the supplied Romeo and Juliet story above.";

/// The ~1,000-token Romeo and Juliet opening used as the warmup and measured
/// continuation source.
const ROMEO_AND_JULIET_SOURCE: &str =
    include_str!("../fixtures/model_metrics_warmup_romeo_and_juliet.txt");

/// The warmup output cap: ~100 output tokens.
const WARMUP_MAXIMUM_OUTPUT_TOKENS: u16 = 100;

/// The continuation instruction prepended to the measured image and the Romeo
/// and Juliet source.
const MEASURED_INPUT_INSTRUCTION: &str = "Describe the supplied image, then continue the supplied Romeo and Juliet story above for approximately 1000 words.";

/// The measured output cap: ~1,000 output tokens (±10% acceptable).
const MEASURED_MAXIMUM_OUTPUT_TOKENS: u16 = 1_000;

/// The sampling temperature in thousandths (1_000 = 1.0, unclamped).
const TEMPERATURE_THOUSANDTHS: u16 = 1_000;

/// Measures the resident sparse-MoE model's VISION serving prompt-processing
/// and decode throughput: after a short small-image warmup, a 9,216-visual-
/// token image plus the Romeo and Juliet continuation drives ~1,000 output
/// tokens with the SSD (persistent prompt) cache disabled, and the measured
/// completion's server-attributed rates are persisted to the durable
/// historical record under the "vision" journey kind.
#[test]
#[ignore = "loads the resident 35B sparse-MoE model and measures its vision serving throughput over IPC"]
fn should_measure_resident_sparse_moe_vision_prompt_processing_and_decode_throughput() {
    let journey = ThroughputJourney {
        journey_kind: ThroughputJourneyKind::Vision,
        warmup_input_prompt: WARMUP_INPUT_INSTRUCTION.to_owned() + "\n\n" + ROMEO_AND_JULIET_SOURCE,
        warmup_images: vec![solid_color_png(WARMUP_IMAGE_SIDE_PIXELS)],
        warmup_output_tokens: WARMUP_MAXIMUM_OUTPUT_TOKENS,
        measured_input_prompt: MEASURED_INPUT_INSTRUCTION.to_owned()
            + "\n\n"
            + ROMEO_AND_JULIET_SOURCE,
        measured_images: vec![solid_color_png(MEASURED_IMAGE_SIDE_PIXELS)],
        measured_output_tokens: MEASURED_MAXIMUM_OUTPUT_TOKENS,
        temperature_thousandths: TEMPERATURE_THOUSANDTHS,
    };
    throughput_support::run_journey_with_timeout(support::resident_sparse_moe_model_id(), journey);
}

/// Encodes one solid-color PNG of the requested side length. The pixel content
/// does not affect the visual token count; a solid color keeps the encoded
/// fixture bytes tiny.
fn solid_color_png(side_pixels: u32) -> ChatImageInput {
    let source_image = RgbImage::from_pixel(side_pixels, side_pixels, Rgb([128, 64, 32]));
    let mut encoded_image_bytes = Cursor::new(Vec::new());
    DynamicImage::ImageRgb8(source_image)
        .write_to(&mut encoded_image_bytes, ImageFormat::Png)
        .expect("the solid-color PNG should encode");
    ChatImageInput {
        mime_type: "image/png".to_owned(),
        decoded_bytes: encoded_image_bytes.into_inner(),
    }
}
