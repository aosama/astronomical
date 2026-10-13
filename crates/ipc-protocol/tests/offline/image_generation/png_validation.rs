//! Adversarial PNG fixtures for the image-completion wire boundary.

use std::{io::Cursor, ops::Range};

use astronomical_ipc_protocol::{
    GeneratedImage, ImageGenerationCompletionValidationError, ImageGenerationResultMetadata,
    ProtocolError, RequestId, WorkerEvent, decode_event, encode_event,
};
use image::{ExtendedColorType, ImageEncoder, codecs::png::PngEncoder};

#[test]
fn should_reject_empty_idat_data_at_the_wire_boundary() {
    assert_invalid_png_encoding(png_bytes_from_chunks(1_024, 768, 8, 2, &[]));
}

#[test]
fn should_reject_decoded_rgb_bomb_before_allocation() {
    let worker_event = image_completion_event(
        png_bytes_from_chunks(16_384, 16_384, 8, 2, &[0]),
        ImageGenerationResultMetadata {
            width_pixels: 16_384,
            height_pixels: 16_384,
            ..valid_image_result_metadata()
        },
    );

    assert!(matches!(
        decode_event(&encode_event(&worker_event).expect("event should serialize")),
        Err(ProtocolError::InvalidImageGenerationCompletion(
            ImageGenerationCompletionValidationError::PngDecodeResourceLimit
        ))
    ));
}

#[test]
fn should_reject_source_palette_png_even_when_the_decoder_expands_it_to_rgb8() {
    let mut palette_png = encode_solid_luma_png(1_024, 768);
    let ihdr_range = png_chunk_range(&palette_png, b"IHDR");
    palette_png[ihdr_range.start + 17] = 3;
    rewrite_png_chunk_crc(&mut palette_png, ihdr_range.clone());
    let idat_start = png_chunk_range(&palette_png, b"IDAT").start;
    palette_png.splice(idat_start..idat_start, png_chunk_bytes(b"PLTE", &[0, 0, 0]));

    let worker_event = image_completion_event(palette_png, valid_image_result_metadata());
    assert!(matches!(
        decode_event(&encode_event(&worker_event).expect("event should serialize")),
        Err(ProtocolError::InvalidImageGenerationCompletion(
            ImageGenerationCompletionValidationError::NonRgb8Png
        ))
    ));
}

#[test]
fn should_reject_corrupt_missing_and_trailing_iend_at_the_wire_boundary() {
    let valid_png = valid_rgb_png(1_024, 768);
    let iend_range = png_chunk_range(&valid_png, b"IEND");

    let mut corrupt_iend_png = valid_png.clone();
    corrupt_iend_png[iend_range.end - 1] ^= 0xff;
    assert_invalid_png_encoding(corrupt_iend_png);

    let mut missing_iend_png = valid_png.clone();
    missing_iend_png.drain(iend_range.clone());
    assert_invalid_png_encoding(missing_iend_png);

    let mut trailing_iend_png = valid_png;
    trailing_iend_png.push(0);
    assert_invalid_png_encoding(trailing_iend_png);
}

#[test]
fn should_reject_nonzero_length_iend_at_the_wire_boundary() {
    let mut png = png_bytes_from_chunks(1_024, 768, 8, 2, &[0]);
    let iend_range = png_chunk_range(&png, b"IEND");
    png.splice(iend_range, png_chunk_bytes(b"IEND", &[0]));

    assert_invalid_png_encoding(png);
}

#[test]
fn should_reject_malformed_deflate_with_valid_chunk_crcs() {
    assert_invalid_png_encoding(png_bytes_from_chunks(
        1_024,
        768,
        8,
        2,
        &[0xde, 0xad, 0xbe, 0xef],
    ));
}

fn encode_solid_luma_png(width_pixels: u32, height_pixels: u32) -> Vec<u8> {
    let pixel_bytes = vec![0; width_pixels as usize * height_pixels as usize];
    encode_png(
        &pixel_bytes,
        width_pixels,
        height_pixels,
        ExtendedColorType::L8,
    )
}

fn valid_rgb_png(width_pixels: u32, height_pixels: u32) -> Vec<u8> {
    let pixel_bytes = vec![0; width_pixels as usize * height_pixels as usize * 3];
    encode_png(
        &pixel_bytes,
        width_pixels,
        height_pixels,
        ExtendedColorType::Rgb8,
    )
}

fn encode_png(
    pixel_bytes: &[u8],
    width_pixels: u32,
    height_pixels: u32,
    color_type: ExtendedColorType,
) -> Vec<u8> {
    let mut png_bytes = Vec::new();
    PngEncoder::new(Cursor::new(&mut png_bytes))
        .write_image(pixel_bytes, width_pixels, height_pixels, color_type)
        .expect("the test image should encode as PNG");
    png_bytes
}

fn png_bytes_from_chunks(
    width_pixels: u32,
    height_pixels: u32,
    bit_depth: u8,
    color_type: u8,
    idat_bytes: &[u8],
) -> Vec<u8> {
    let mut ihdr_data = Vec::with_capacity(13);
    ihdr_data.extend_from_slice(&width_pixels.to_be_bytes());
    ihdr_data.extend_from_slice(&height_pixels.to_be_bytes());
    ihdr_data.extend_from_slice(&[bit_depth, color_type, 0, 0, 0]);

    let mut png_bytes = b"\x89PNG\r\n\x1a\n".to_vec();
    png_bytes.extend(png_chunk_bytes(b"IHDR", &ihdr_data));
    png_bytes.extend(png_chunk_bytes(b"IDAT", idat_bytes));
    png_bytes.extend(png_chunk_bytes(b"IEND", &[]));
    png_bytes
}

fn png_chunk_bytes(chunk_type: &[u8; 4], chunk_data: &[u8]) -> Vec<u8> {
    let mut chunk_bytes = Vec::with_capacity(chunk_data.len() + 12);
    chunk_bytes.extend_from_slice(
        &u32::try_from(chunk_data.len())
            .expect("the test chunk should fit in a PNG length")
            .to_be_bytes(),
    );
    chunk_bytes.extend_from_slice(chunk_type);
    chunk_bytes.extend_from_slice(chunk_data);
    let mut crc_hasher = crc32fast::Hasher::new();
    crc_hasher.update(chunk_type);
    crc_hasher.update(chunk_data);
    chunk_bytes.extend_from_slice(&crc_hasher.finalize().to_be_bytes());
    chunk_bytes
}

fn png_chunk_range(png_bytes: &[u8], expected_chunk_type: &[u8; 4]) -> Range<usize> {
    let mut chunk_start = 8;
    while chunk_start < png_bytes.len() {
        let chunk_data_bytes = u32::from_be_bytes(
            png_bytes[chunk_start..chunk_start + 4]
                .try_into()
                .expect("the encoded PNG should contain a chunk length"),
        ) as usize;
        let chunk_end = chunk_start + 12 + chunk_data_bytes;
        if &png_bytes[chunk_start + 4..chunk_start + 8] == expected_chunk_type {
            return chunk_start..chunk_end;
        }
        chunk_start = chunk_end;
    }
    panic!("the encoded PNG should contain the requested chunk")
}

fn rewrite_png_chunk_crc(png_bytes: &mut [u8], chunk_range: Range<usize>) {
    let chunk_type_start = chunk_range.start + 4;
    let crc_start = chunk_range.end - 4;
    let mut crc_hasher = crc32fast::Hasher::new();
    crc_hasher.update(&png_bytes[chunk_type_start..crc_start]);
    png_bytes[crc_start..chunk_range.end].copy_from_slice(&crc_hasher.finalize().to_be_bytes());
}

fn image_completion_event(
    encoded_bytes: Vec<u8>,
    result_metadata: ImageGenerationResultMetadata,
) -> WorkerEvent {
    WorkerEvent::ImageGenerationCompleted {
        request_id: RequestId::new(501),
        generated_image: GeneratedImage {
            mime_type: "image/png".to_owned(),
            encoded_bytes,
        },
        result_metadata,
    }
}

fn valid_image_result_metadata() -> ImageGenerationResultMetadata {
    ImageGenerationResultMetadata {
        width_pixels: 1_024,
        height_pixels: 768,
        steps: 4,
        guidance_thousandths: 1_000,
        seed: 42,
        elapsed_millis: 1_800,
    }
}

fn assert_invalid_png_encoding(encoded_bytes: Vec<u8>) {
    let worker_event = image_completion_event(encoded_bytes, valid_image_result_metadata());
    assert!(matches!(
        decode_event(&encode_event(&worker_event).expect("event should serialize")),
        Err(ProtocolError::InvalidImageGenerationCompletion(
            ImageGenerationCompletionValidationError::InvalidPngEncoding
        ))
    ));
}
