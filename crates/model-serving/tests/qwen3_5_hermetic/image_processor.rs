use std::io::Cursor;

use astronomical_model_serving::{Qwen3_5ImageDimensions, Qwen3_5ImageGrid, Qwen3_5ImageProcessor};
use image::{DynamicImage, ImageBuffer, ImageFormat, Rgb};
use sha2::{Digest, Sha256};

use crate::common::SYNTHETIC_RED_PNG_BYTES;

#[test]
fn should_preprocess_the_synthetic_red_fixture_into_vision_patches() {
    let image_processor = Qwen3_5ImageProcessor::qwen3_5_moe_35b_optiq();
    let decoded_red_image = image::load_from_memory(SYNTHETIC_RED_PNG_BYTES)
        .expect("the synthetic red fixture should decode")
        .to_rgba8();

    assert_eq!(decoded_red_image.get_pixel(0, 0).0, [255, 0, 0, 255]);

    let processed_image = image_processor
        .process_image_bytes(SYNTHETIC_RED_PNG_BYTES)
        .expect("the synthetic red fixture should decode and preprocess");

    assert_eq!(
        processed_image.image_grid,
        Qwen3_5ImageGrid {
            temporal_patch_count: 1,
            height_patch_count: 16,
            width_patch_count: 16,
        }
    );
    assert_eq!(processed_image.resized_height_pixels, 256);
    assert_eq!(processed_image.resized_width_pixels, 256);
    assert_eq!(processed_image.pixel_values_row_count, 256);
    assert_eq!(processed_image.pixel_values_column_count, 1_536);
    assert_eq!(processed_image.image_token_count_after_spatial_merge, 64);
    assert_eq!(
        processed_image.pixel_values.len(),
        processed_image.pixel_values_row_count * processed_image.pixel_values_column_count
    );
}

#[test]
fn should_plan_ornith_image_resize_dimensions_from_pixel_budget_and_patch_factor() {
    let image_processor = Qwen3_5ImageProcessor::qwen3_5_moe_35b_optiq();

    assert_eq!(
        image_processor
            .resized_dimensions_for_image(8, 8)
            .expect("small image dimensions should expand to the minimum pixel budget"),
        Qwen3_5ImageDimensions {
            height_pixels: 256,
            width_pixels: 256,
        }
    );
    assert_eq!(
        image_processor
            .resized_dimensions_for_image(5_000, 5_000)
            .expect("large image dimensions should shrink to the maximum pixel budget"),
        Qwen3_5ImageDimensions {
            height_pixels: 4_096,
            width_pixels: 4_096,
        }
    );
}

#[test]
fn should_preprocess_the_openai_data_uri_one_pixel_png_fixture() {
    let image_processor = Qwen3_5ImageProcessor::qwen3_5_moe_35b_optiq();

    let processed_image = image_processor
        .process_image_bytes(SYNTHETIC_RED_PNG_BYTES)
        .expect("the accepted OpenAI data URI fixture should preprocess");

    assert_eq!(processed_image.resized_height_pixels, 256);
    assert_eq!(processed_image.resized_width_pixels, 256);
    assert_eq!(processed_image.image_token_count_after_spatial_merge, 64);
}

#[test]
fn should_normalize_white_image_pixels_to_one() {
    let image_processor = Qwen3_5ImageProcessor::qwen3_5_moe_35b_optiq();
    let encoded_white_png_bytes = encode_solid_rgb_png([255, 255, 255], 256, 256);

    let processed_image = image_processor
        .process_image_bytes(&encoded_white_png_bytes)
        .expect("solid white PNG should decode and preprocess");

    assert_eq!(processed_image.image_grid.height_patch_count, 16);
    assert_eq!(processed_image.image_grid.width_patch_count, 16);
    assert!(
        processed_image
            .pixel_values
            .iter()
            .all(|normalized_channel| (*normalized_channel - 1.0).abs() < f32::EPSILON),
        "all white channels should normalize from 255 to 1.0"
    );
}

#[test]
fn should_hash_the_exact_encoded_image_bytes() {
    let image_processor = Qwen3_5ImageProcessor::qwen3_5_moe_35b_optiq();
    let encoded_png_bytes = encode_solid_rgb_png([128, 64, 32], 32, 32);
    let mut byte_distinct_png_bytes = encoded_png_bytes.clone();
    byte_distinct_png_bytes.push(0);

    let processed_image = image_processor
        .process_image_bytes(&encoded_png_bytes)
        .expect("the original PNG should decode and preprocess");
    let byte_distinct_processed_image = image_processor
        .process_image_bytes(&byte_distinct_png_bytes)
        .expect("a PNG with a trailing byte should remain a valid byte-distinct encoding");
    let expected_encoded_image_sha256: [u8; 32] = Sha256::digest(&encoded_png_bytes).into();

    assert_eq!(
        processed_image.encoded_image_sha256,
        expected_encoded_image_sha256
    );
    assert_ne!(
        processed_image.encoded_image_sha256, byte_distinct_processed_image.encoded_image_sha256,
        "byte-distinct encodings must never share persistent visual state"
    );
    assert_eq!(
        processed_image.pixel_values, byte_distinct_processed_image.pixel_values,
        "the exact-byte identity must remain distinct even when decoded pixels match"
    );
}

fn encode_solid_rgb_png(rgb_pixel: [u8; 3], width_pixels: u32, height_pixels: u32) -> Vec<u8> {
    let solid_rgb_image = ImageBuffer::from_pixel(width_pixels, height_pixels, Rgb(rgb_pixel));
    let mut encoded_png_bytes = Vec::new();
    DynamicImage::ImageRgb8(solid_rgb_image)
        .write_to(&mut Cursor::new(&mut encoded_png_bytes), ImageFormat::Png)
        .expect("test PNG encoding should succeed");
    encoded_png_bytes
}
