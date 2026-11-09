use std::path::Path;

use super::parse_persistent_prompt_cache_file_hash_from_path;

const LOWERCASE_HASH_HEX: &str = "00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff";

#[test]
fn should_parse_a_lowercase_hash_name_with_or_without_an_extension() {
    let expected_hash: [u8; 32] = [
        0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee,
        0xff, 0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd,
        0xee, 0xff,
    ];
    let block_directory = format!("/cache/blocks/{LOWERCASE_HASH_HEX}");
    let state_file = format!("/cache/sequence/{LOWERCASE_HASH_HEX}.safetensors");
    assert_eq!(
        parse_persistent_prompt_cache_file_hash_from_path(Path::new(&block_directory)),
        Some(expected_hash)
    );
    assert_eq!(
        parse_persistent_prompt_cache_file_hash_from_path(Path::new(&state_file)),
        Some(expected_hash)
    );
}

#[test]
fn should_refuse_a_name_one_character_short_or_long() {
    let short_name = &LOWERCASE_HASH_HEX[..63];
    let long_name = format!("{LOWERCASE_HASH_HEX}0");
    assert_eq!(
        parse_persistent_prompt_cache_file_hash_from_path(Path::new(short_name)),
        None
    );
    assert_eq!(
        parse_persistent_prompt_cache_file_hash_from_path(Path::new(&long_name)),
        None
    );
}

#[test]
fn should_refuse_a_64_byte_name_with_a_multibyte_character_instead_of_panicking() {
    // One ASCII byte, then a two-byte "é", then 61 more: 64 bytes, but every
    // two-byte hex slice after the first character cuts "é" in half.
    let multibyte_name = format!("a\u{e9}{}", "a".repeat(61));
    assert_eq!(multibyte_name.len(), 64);
    assert_eq!(
        parse_persistent_prompt_cache_file_hash_from_path(Path::new(&multibyte_name)),
        None
    );
}

#[test]
fn should_refuse_names_that_only_number_parsing_accepts() {
    // from_str_radix accepts a leading sign and uppercase digits; neither is
    // a name the cache ever writes, and accepting them lets two directories
    // claim one hash on a case-sensitive volume.
    let signed_name = format!("+f{}", &LOWERCASE_HASH_HEX[2..]);
    let uppercase_name = LOWERCASE_HASH_HEX.to_ascii_uppercase();
    for foreign_name in [signed_name, uppercase_name] {
        assert_eq!(
            parse_persistent_prompt_cache_file_hash_from_path(Path::new(&foreign_name)),
            None,
            "{foreign_name}"
        );
    }
}
