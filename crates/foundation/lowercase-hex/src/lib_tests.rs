use super::{decode_lowercase_hex, encode_lowercase_hex, is_lowercase_hex_of_length};

const ASCENDING_DIGEST_HEX: &str =
    "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f";

fn ascending_digest() -> [u8; 32] {
    core::array::from_fn(byte_value_from_index)
}

fn byte_value_from_index(byte_index: usize) -> u8 {
    u8::try_from(byte_index).expect("a digest index fits in a byte")
}

#[test]
fn should_encode_every_byte_as_two_lowercase_digits() {
    assert_eq!(
        encode_lowercase_hex(&ascending_digest()),
        ASCENDING_DIGEST_HEX
    );
    assert_eq!(encode_lowercase_hex(&[0xab, 0xcd, 0xef]), "abcdef");
}

#[test]
fn should_encode_all_zero_and_all_ff_bytes_at_the_digit_boundaries() {
    assert_eq!(encode_lowercase_hex(&[0x00; 32]), "0".repeat(64));
    assert_eq!(encode_lowercase_hex(&[0xff; 32]), "f".repeat(64));
}

#[test]
fn should_encode_no_bytes_as_empty_text() {
    assert_eq!(encode_lowercase_hex(&[]), "");
}

#[test]
fn should_decode_every_lowercase_digit_into_its_byte() {
    assert_eq!(
        decode_lowercase_hex::<32>(ASCENDING_DIGEST_HEX),
        Some(ascending_digest())
    );
}

#[test]
fn should_decode_all_zero_and_all_ff_digits_at_the_digit_boundaries() {
    assert_eq!(
        decode_lowercase_hex::<32>(&"0".repeat(64)),
        Some([0x00; 32])
    );
    assert_eq!(
        decode_lowercase_hex::<32>(&"f".repeat(64)),
        Some([0xff; 32])
    );
}

#[test]
fn should_decode_at_a_width_other_than_a_digest() {
    assert_eq!(
        decode_lowercase_hex::<8>("0001020304050607"),
        Some([0, 1, 2, 3, 4, 5, 6, 7])
    );
}

#[test]
fn should_decode_back_what_it_encoded() {
    let digest = ascending_digest();
    assert_eq!(
        decode_lowercase_hex::<32>(&encode_lowercase_hex(&digest)),
        Some(digest)
    );
}

#[test]
fn should_decode_empty_text_only_for_a_zero_byte_width() {
    assert_eq!(decode_lowercase_hex::<0>(""), Some([]));
    assert_eq!(decode_lowercase_hex::<32>(""), None);
}

#[test]
fn should_refuse_to_decode_text_one_digit_short_or_long() {
    assert_eq!(
        decode_lowercase_hex::<32>(&ASCENDING_DIGEST_HEX[..63]),
        None
    );
    assert_eq!(
        decode_lowercase_hex::<32>(&format!("{ASCENDING_DIGEST_HEX}0")),
        None
    );
}

#[test]
fn should_refuse_to_decode_an_odd_number_of_digits() {
    assert_eq!(decode_lowercase_hex::<1>("a"), None);
    assert_eq!(decode_lowercase_hex::<1>("abc"), None);
}

#[test]
fn should_refuse_to_decode_uppercase_signed_and_non_hex_digits() {
    for foreign_digest in [
        ASCENDING_DIGEST_HEX.to_ascii_uppercase(),
        format!("+f{}", &ASCENDING_DIGEST_HEX[2..]),
        format!("g0{}", &ASCENDING_DIGEST_HEX[2..]),
        ASCENDING_DIGEST_HEX.replacen('a', "A", 1),
    ] {
        assert_eq!(
            decode_lowercase_hex::<32>(&foreign_digest),
            None,
            "{foreign_digest}"
        );
    }
}

#[test]
fn should_refuse_a_multibyte_character_without_panicking() {
    let multibyte_digest = format!("a\u{e9}{}", "a".repeat(61));
    assert_eq!(multibyte_digest.len(), 64);
    assert_eq!(decode_lowercase_hex::<32>(&multibyte_digest), None);
    assert!(!is_lowercase_hex_of_length(&multibyte_digest, 64));
}

#[test]
fn should_accept_lowercase_digits_of_the_expected_length() {
    assert!(is_lowercase_hex_of_length(ASCENDING_DIGEST_HEX, 64));
    assert!(is_lowercase_hex_of_length("0123456789abcdef", 16));
}

#[test]
fn should_accept_all_zero_and_all_f_digits_and_the_empty_text_of_length_zero() {
    assert!(is_lowercase_hex_of_length(&"0".repeat(40), 40));
    assert!(is_lowercase_hex_of_length(&"f".repeat(40), 40));
    assert!(is_lowercase_hex_of_length("", 0));
}

#[test]
fn should_reject_uppercase_non_hex_and_wrongly_sized_text() {
    assert!(!is_lowercase_hex_of_length("0123456789ABCDEF", 16));
    assert!(!is_lowercase_hex_of_length("0123456789abcdeg", 16));
    assert!(!is_lowercase_hex_of_length("+123456789abcdef", 16));
    assert!(!is_lowercase_hex_of_length("0123456789abcde", 16));
    assert!(!is_lowercase_hex_of_length("0123456789abcdef0", 16));
    assert!(!is_lowercase_hex_of_length("abcd", 3));
    assert!(!is_lowercase_hex_of_length("", 16));
}
