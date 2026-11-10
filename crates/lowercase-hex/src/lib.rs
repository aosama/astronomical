//! Lowercase hexadecimal text for the digests, revisions and cache
//! identities the product crates write and read.
//!
//! Those identities are written as lowercase hex, so the decoder and
//! validator reject uppercase digits: accepting both cases would let two
//! different texts name one digest. That is also why this crate exists
//! instead of the `hex` crate, which decodes uppercase digits.
//!
//! `astronomical-native-identity` and `astronomical-native-build-orchestration`
//! keep their own encoders on purpose: their sources are inputs to the native
//! build identity, so editing them would force a cold native rebuild for no
//! change in output.
//!
//! The text is walked as bytes. Slicing a `str` at two-byte offsets panics
//! when a corrupt name holds a multibyte character, and `u8::from_str_radix`
//! accepts a leading `+` as well as uppercase digits.

const LOWERCASE_HEX_DIGITS: &[u8; 16] = b"0123456789abcdef";

/// Encodes each byte as two lowercase hexadecimal digits, high nibble first.
#[must_use]
pub fn encode_lowercase_hex(bytes: &[u8]) -> String {
    let mut encoded_text = String::with_capacity(bytes.len() * 2);
    for &source_byte in bytes {
        encoded_text.push(char::from(
            LOWERCASE_HEX_DIGITS[usize::from(source_byte >> 4)],
        ));
        encoded_text.push(char::from(
            LOWERCASE_HEX_DIGITS[usize::from(source_byte & 0x0f)],
        ));
    }
    encoded_text
}

/// Whether `text` is exactly `expected_digit_count` lowercase hexadecimal digits.
///
/// The length is part of the check because every caller validates a fixed
/// identity width (a 40-digit commit, a 64-digit SHA-256, a 16-digit hash
/// prefix); a bare digit check would let a truncated identity through.
#[must_use]
pub fn is_lowercase_hex_of_length(text: &str, expected_digit_count: usize) -> bool {
    text.len() == expected_digit_count && text.bytes().all(is_lowercase_hex_digit)
}

fn is_lowercase_hex_digit(hex_digit: u8) -> bool {
    lowercase_hex_nibble(hex_digit).is_some()
}

/// Decodes exactly `2 * BYTE_COUNT` lowercase hexadecimal digits into
/// `BYTE_COUNT` bytes; any other length or any other character is `None`.
#[must_use]
pub fn decode_lowercase_hex<const BYTE_COUNT: usize>(text: &str) -> Option<[u8; BYTE_COUNT]> {
    let text_bytes = text.as_bytes();
    if text_bytes.len() != BYTE_COUNT * 2 {
        return None;
    }
    let mut decoded_bytes = [0_u8; BYTE_COUNT];
    let (hex_digit_pairs, _) = text_bytes.as_chunks::<2>();
    for (decoded_byte, [high_hex_digit, low_hex_digit]) in
        decoded_bytes.iter_mut().zip(hex_digit_pairs)
    {
        let high_nibble = lowercase_hex_nibble(*high_hex_digit)?;
        let low_nibble = lowercase_hex_nibble(*low_hex_digit)?;
        *decoded_byte = (high_nibble << 4) | low_nibble;
    }
    Some(decoded_bytes)
}

fn lowercase_hex_nibble(hex_digit: u8) -> Option<u8> {
    match hex_digit {
        b'0'..=b'9' => Some(hex_digit - b'0'),
        b'a'..=b'f' => Some(hex_digit - b'a' + 10),
        _ => None,
    }
}

#[cfg(test)]
#[path = "lib_tests.rs"]
mod lib_tests;
