use super::{WeightBlobFile, pack_uint4_low_nibble_first};

fn little_endian_u32(file_bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(
        file_bytes[offset..offset + 4]
            .try_into()
            .expect("four bytes"),
    )
}

fn little_endian_u64(file_bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(
        file_bytes[offset..offset + 8]
            .try_into()
            .expect("eight bytes"),
    )
}

#[test]
fn should_write_an_empty_version_two_header() {
    let file_bytes = WeightBlobFile::new().into_bytes();

    assert_eq!(file_bytes.len(), 64);
    assert_eq!(little_endian_u32(&file_bytes, 0), 0);
    assert_eq!(little_endian_u32(&file_bytes, 4), 2);
}

#[test]
fn should_describe_an_fp16_blob_with_a_sentinel_metadata_record() {
    let mut blob_file = WeightBlobFile::new();

    let metadata_offset = blob_file.append_fp16(&[0x3c00, 0xc100]);
    let file_bytes = blob_file.into_bytes();

    assert_eq!(metadata_offset, 64);
    assert_eq!(little_endian_u32(&file_bytes, 0), 1);
    assert_eq!(little_endian_u32(&file_bytes, 64), 0xDEAD_BEEF);
    assert_eq!(little_endian_u32(&file_bytes, 68), 1, "Float16 type code");
    assert_eq!(little_endian_u64(&file_bytes, 72), 4, "size in bytes");
    assert_eq!(little_endian_u64(&file_bytes, 80), 128, "data offset");
    assert_eq!(&file_bytes[128..132], &[0x00, 0x3c, 0x00, 0xc1]);
}

#[test]
fn should_start_every_blob_on_a_sixty_four_byte_boundary() {
    let mut blob_file = WeightBlobFile::new();

    let first_offset = blob_file.append_fp16(&[0; 3]);
    let second_offset = blob_file.append_uint4(&[1, 2, 3]);
    let file_bytes = blob_file.into_bytes();

    assert_eq!(first_offset, 64);
    assert_eq!(second_offset, 192);
    assert_eq!(file_bytes.len() % 64, 0);
    assert_eq!(little_endian_u32(&file_bytes, 0), 2);
}

#[test]
fn should_record_the_unused_high_nibble_of_an_odd_uint4_blob() {
    let mut blob_file = WeightBlobFile::new();

    blob_file.append_uint4(&[1, 2, 3]);
    let file_bytes = blob_file.into_bytes();

    assert_eq!(little_endian_u32(&file_bytes, 68), 11, "UInt4 type code");
    assert_eq!(little_endian_u64(&file_bytes, 72), 2, "size in bytes");
    assert_eq!(little_endian_u64(&file_bytes, 88), 4, "padding bits");
}

#[test]
fn should_record_no_padding_for_an_even_uint4_blob() {
    let mut blob_file = WeightBlobFile::new();

    blob_file.append_uint4(&[1, 2]);
    let file_bytes = blob_file.into_bytes();

    assert_eq!(little_endian_u64(&file_bytes, 88), 0);
}

#[test]
fn should_store_uint8_codes_one_per_byte() {
    let mut blob_file = WeightBlobFile::new();

    blob_file.append_uint8(&[7, 200, 255]);
    let file_bytes = blob_file.into_bytes();

    assert_eq!(little_endian_u32(&file_bytes, 68), 3, "UInt8 type code");
    assert_eq!(little_endian_u64(&file_bytes, 72), 3, "size in bytes");
    assert_eq!(&file_bytes[128..131], &[7, 200, 255]);
}

#[test]
fn should_pack_the_first_code_into_the_low_nibble() {
    assert_eq!(
        pack_uint4_low_nibble_first(&[0x1, 0xf, 0x7]),
        vec![0xf1, 0x07]
    );
    assert_eq!(pack_uint4_low_nibble_first(&[]), Vec::<u8>::new());
}
