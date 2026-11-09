use super::decode_lowercase_hex_digest;

const ASCENDING_DIGEST_HEX: &str =
    "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f";

#[test]
fn should_decode_every_lowercase_hex_digit_into_its_byte() {
    let ascending_digest: [u8; 32] = core::array::from_fn(byte_value_from_index);
    assert_eq!(
        decode_lowercase_hex_digest(ASCENDING_DIGEST_HEX),
        Some(ascending_digest)
    );
    assert_eq!(
        decode_lowercase_hex_digest(&"f".repeat(64)),
        Some([0xff; 32])
    );
}

fn byte_value_from_index(byte_index: usize) -> u8 {
    u8::try_from(byte_index).expect("a digest index fits in a byte")
}

#[test]
fn should_refuse_a_digest_one_character_short_or_long() {
    assert_eq!(
        decode_lowercase_hex_digest(&ASCENDING_DIGEST_HEX[..63]),
        None
    );
    assert_eq!(
        decode_lowercase_hex_digest(&format!("{ASCENDING_DIGEST_HEX}0")),
        None
    );
    assert_eq!(decode_lowercase_hex_digest(""), None);
}

#[test]
fn should_refuse_uppercase_signed_and_non_hex_characters() {
    for foreign_digest in [
        ASCENDING_DIGEST_HEX.to_ascii_uppercase(),
        format!("+f{}", &ASCENDING_DIGEST_HEX[2..]),
        format!("g0{}", &ASCENDING_DIGEST_HEX[2..]),
    ] {
        assert_eq!(
            decode_lowercase_hex_digest(&foreign_digest),
            None,
            "{foreign_digest}"
        );
    }
}

#[test]
fn should_refuse_a_64_byte_text_holding_a_multibyte_character_without_panicking() {
    let multibyte_digest = format!("a\u{e9}{}", "a".repeat(61));
    assert_eq!(multibyte_digest.len(), 64);
    assert_eq!(decode_lowercase_hex_digest(&multibyte_digest), None);
}

#[test]
fn should_encode_every_byte_as_two_lowercase_digits_and_decode_back() {
    let ascending_digest: [u8; 32] = core::array::from_fn(byte_value_from_index);
    assert_eq!(
        super::encode_lowercase_hex_digest(ascending_digest),
        ASCENDING_DIGEST_HEX
    );
    assert_eq!(
        super::encode_lowercase_hex_digest([0xff; 32]),
        "f".repeat(64)
    );
    assert_eq!(super::encode_lowercase_hex_digest([0; 32]), "0".repeat(64));
}
