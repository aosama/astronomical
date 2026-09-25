//! Shared fixtures for the Qwen-Image-2.1 hermetic tests.

use serde_json::Value;

/// The oracle fixtures store every float as exact decimal text rather than as a JSON number. The
/// stored values must round-trip bit-for-bit, and a JSON number read back through the shared JSON
/// parser can land one unit in the last place off, so the fixtures carry the decimal text and the
/// standard library's correctly-rounded float parsing reconstructs the exact stored value.
pub fn oracle_document(json: &str) -> Value {
    serde_json::from_str(json).expect("the oracle fixture must be a valid JSON document")
}

/// Parse a JSON array of exact decimal text into `f32` values.
pub fn oracle_f32_array(values: &Value) -> Vec<f32> {
    values
        .as_array()
        .expect("an oracle float field must be an array")
        .iter()
        .map(|entry| {
            entry
                .as_str()
                .expect("an oracle float must be stored as decimal text")
                .parse::<f32>()
                .expect("an oracle float must parse as f32")
        })
        .collect()
}

/// Parse a JSON array of exact decimal text into `f64` values.
pub fn oracle_f64_array(values: &Value) -> Vec<f64> {
    values
        .as_array()
        .expect("an oracle float field must be an array")
        .iter()
        .map(|entry| {
            entry
                .as_str()
                .expect("an oracle float must be stored as decimal text")
                .parse::<f64>()
                .expect("an oracle float must parse as f64")
        })
        .collect()
}

pub fn oracle_f32_values(document: &Value, field: &str) -> Vec<f32> {
    oracle_f32_array(&document[field])
}

pub fn oracle_f64_values(document: &Value, field: &str) -> Vec<f64> {
    oracle_f64_array(&document[field])
}

/// Parse a JSON array of whole-number axis indices.
pub fn oracle_index_values(document: &Value, field: &str) -> Vec<isize> {
    document[field]
        .as_array()
        .expect("an oracle index field must be an array")
        .iter()
        .map(|entry| {
            entry
                .as_i64()
                .expect("an oracle axis index must be a whole number") as isize
        })
        .collect()
}

/// Tolerated absolute difference per f32 element when comparing against the diffusers oracle.
///
/// Both compute cosine/sine in f64 and cast to f32, so the real gap is sub-ULP; the slack covers
/// platform trig differences and is deliberately not tight enough to mask a wrong index.
pub const ORACLE_TOLERANCE: f32 = 1e-4;

/// The fixed joint sequence used by every test: 2 text tokens, a 2x2 condition block, 3 text
/// tokens, a 2x3 condition block, 1 trailing text token (16 tokens total).
///
/// `img_shapes` are `(height, width)` latent-token pairs; block boundaries come from these counts,
/// not from the `true` runs in the mask.
pub fn test_img_shapes() -> [(usize, usize); 2] {
    [(2, 2), (2, 3)]
}

/// The fixed joint sequence used by every test, aligned with [`test_img_shapes`]: 2 text
/// tokens, a 2x2 condition block, 3 text tokens, a 2x3 condition block, 1 trailing text token
/// (16 tokens total).
pub fn test_image_pad_mask() -> [bool; 16] {
    [
        false, false, // text
        true, true, true, true, // 2x2 block
        false, false, false, // text
        true, true, true, true, true, true,  // 2x3 block
        false, // text
    ]
}
