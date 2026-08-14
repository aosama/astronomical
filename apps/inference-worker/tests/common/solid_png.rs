//! Builds a solid-color PNG in memory for multimodal REST journeys.

#![allow(dead_code)]

use base64::Engine as _;

/// Returns a `data:image/png;base64,...` URL for a solid RGB image.
pub(crate) fn solid_rgb_png_data_url(
    width: u32,
    height: u32,
    red: u8,
    green: u8,
    blue: u8,
) -> String {
    let png_bytes = encode_solid_rgb_png(width, height, red, green, blue);
    format!(
        "data:image/png;base64,{}",
        base64::engine::general_purpose::STANDARD.encode(png_bytes)
    )
}

fn encode_solid_rgb_png(width: u32, height: u32, red: u8, green: u8, blue: u8) -> Vec<u8> {
    let mut png_bytes = Vec::from(b"\x89PNG\r\n\x1a\n".as_slice());
    let mut ihdr_payload = Vec::new();
    ihdr_payload.extend_from_slice(&width.to_be_bytes());
    ihdr_payload.extend_from_slice(&height.to_be_bytes());
    ihdr_payload.extend_from_slice(&[8, 2, 0, 0, 0]);
    write_png_chunk(&mut png_bytes, b"IHDR", &ihdr_payload);

    let row_stride = 1_usize
        .checked_add(
            usize::try_from(width)
                .expect("PNG width should fit usize")
                .saturating_mul(3),
        )
        .expect("PNG row stride should fit");
    let raw_scanline_byte_count = row_stride
        .checked_mul(usize::try_from(height).expect("PNG height should fit usize"))
        .expect("PNG payload should fit");
    let mut raw_scanlines = vec![0_u8; raw_scanline_byte_count];
    for row_index in 0..height as usize {
        let row_start = row_index.saturating_mul(row_stride);
        raw_scanlines[row_start] = 0;
        for pixel_index in 0..width as usize {
            let pixel_start = row_start
                .saturating_add(1)
                .saturating_add(pixel_index.saturating_mul(3));
            raw_scanlines[pixel_start] = red;
            raw_scanlines[pixel_start.saturating_add(1)] = green;
            raw_scanlines[pixel_start.saturating_add(2)] = blue;
        }
    }
    write_png_chunk(&mut png_bytes, b"IDAT", &zlib_store(&raw_scanlines));
    write_png_chunk(&mut png_bytes, b"IEND", &[]);
    png_bytes
}

fn write_png_chunk(png_bytes: &mut Vec<u8>, chunk_type: &[u8; 4], chunk_payload: &[u8]) {
    png_bytes.extend_from_slice(&(chunk_payload.len() as u32).to_be_bytes());
    png_bytes.extend_from_slice(chunk_type);
    png_bytes.extend_from_slice(chunk_payload);
    let mut crc_input = Vec::from(chunk_type.as_slice());
    crc_input.extend_from_slice(chunk_payload);
    png_bytes.extend_from_slice(&png_crc32(&crc_input).to_be_bytes());
}

fn zlib_store(raw_bytes: &[u8]) -> Vec<u8> {
    let mut zlib_bytes = vec![0x78, 0x01];
    let mut remaining = raw_bytes;
    while !remaining.is_empty() {
        let block_length = remaining.len().min(65_535);
        let is_final_block = block_length == remaining.len();
        zlib_bytes.push(u8::from(is_final_block));
        let block_length_u16 = block_length as u16;
        zlib_bytes.extend_from_slice(&block_length_u16.to_le_bytes());
        zlib_bytes.extend_from_slice((!block_length_u16).to_le_bytes().as_ref());
        zlib_bytes.extend_from_slice(&remaining[..block_length]);
        remaining = &remaining[block_length..];
    }
    zlib_bytes.extend_from_slice(&adler32(raw_bytes).to_be_bytes());
    zlib_bytes
}

fn adler32(raw_bytes: &[u8]) -> u32 {
    let mut first_sum = 1_u32;
    let mut second_sum = 0_u32;
    for byte in raw_bytes {
        first_sum = (first_sum + u32::from(*byte)) % 65_521;
        second_sum = (second_sum + first_sum) % 65_521;
    }
    (second_sum << 16) | first_sum
}

fn png_crc32(bytes: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for byte in bytes {
        crc ^= u32::from(*byte);
        for _bit in 0..8 {
            let mask = 0u32.wrapping_sub(crc & 1);
            crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
        }
    }
    !crc
}
