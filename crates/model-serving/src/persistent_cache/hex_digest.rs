//! Encoding and decoding of the 64-character lowercase hex names and fields that identify
//! cache blocks, state files and visual embeddings.
//!
//! The cache only ever writes lowercase hex, so anything else is foreign and
//! decodes to `None`. The text is walked as bytes: slicing it as `str` at
//! two-byte offsets panics when a corrupt name or manifest holds a multibyte
//! character, and `u8::from_str_radix` would also accept a leading `+` and
//! uppercase digits, letting two different names claim one digest.

/// Digest bytes in a SHA-256-sized cache identity.
pub(crate) const DIGEST_BYTE_COUNT: usize = 32;

/// Decodes exactly 64 lowercase hex characters into 32 bytes.
pub(crate) fn decode_lowercase_hex_digest(digest_text: &str) -> Option<[u8; DIGEST_BYTE_COUNT]> {
    let digest_text_bytes = digest_text.as_bytes();
    if digest_text_bytes.len() != DIGEST_BYTE_COUNT * 2 {
        return None;
    }
    let mut decoded_digest = [0_u8; DIGEST_BYTE_COUNT];
    let (hex_pairs, _) = digest_text_bytes.as_chunks::<2>();
    for (decoded_digest_byte, [high_hex_character, low_hex_character]) in
        decoded_digest.iter_mut().zip(hex_pairs)
    {
        let high_nibble = lowercase_hex_nibble(*high_hex_character)?;
        let low_nibble = lowercase_hex_nibble(*low_hex_character)?;
        *decoded_digest_byte = (high_nibble << 4) | low_nibble;
    }
    Some(decoded_digest)
}

/// Encodes 32 digest bytes as the 64 lowercase hex characters the cache
/// writes for names and manifest fields.
pub(crate) fn encode_lowercase_hex_digest(digest_bytes: [u8; DIGEST_BYTE_COUNT]) -> String {
    const LOWERCASE_HEX_DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut encoded_digest = String::with_capacity(DIGEST_BYTE_COUNT * 2);
    for digest_byte in digest_bytes {
        encoded_digest.push(char::from(
            LOWERCASE_HEX_DIGITS[usize::from(digest_byte >> 4)],
        ));
        encoded_digest.push(char::from(
            LOWERCASE_HEX_DIGITS[usize::from(digest_byte & 0x0f)],
        ));
    }
    encoded_digest
}

fn lowercase_hex_nibble(hex_character: u8) -> Option<u8> {
    match hex_character {
        b'0'..=b'9' => Some(hex_character - b'0'),
        b'a'..=b'f' => Some(hex_character - b'a' + 10),
        _ => None,
    }
}

#[cfg(test)]
#[path = "hex_digest_tests.rs"]
mod hex_digest_tests;
