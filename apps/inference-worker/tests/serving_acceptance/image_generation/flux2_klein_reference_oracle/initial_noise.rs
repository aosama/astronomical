//! Strict provenance contract for the independently generated initial latent noise.

use serde_json::{Map, Value};

use super::{
    ExpectedFluxReference, require_equal_string, require_exact_fields, require_object,
    require_string,
};

pub(super) fn parse_initial_noise(
    bundle_object: &Map<String, Value>,
    expected: &ExpectedFluxReference<'_>,
) -> Result<String, String> {
    let initial_noise = require_object(bundle_object, "initial_noise", "reference bundle")?;
    require_exact_fields(
        initial_noise,
        &[
            "implementation",
            "version",
            "dtype",
            "layout",
            "shape",
            "float32_sha256",
        ],
        "initial_noise",
    )?;
    require_equal_string(initial_noise, "implementation", "mlx")?;
    require_equal_string(initial_noise, "version", "0.32.1")?;
    require_equal_string(initial_noise, "dtype", "bfloat16")?;
    require_equal_string(initial_noise, "layout", "packed_batch_sequence_channels")?;

    let packed_sequence_length = u64::from(expected.width.div_ceil(16))
        .checked_mul(u64::from(expected.height.div_ceil(16)))
        .ok_or_else(|| "initial_noise shape overflowed".to_owned())?;
    let shape = initial_noise
        .get("shape")
        .and_then(Value::as_array)
        .ok_or_else(|| "initial_noise shape must be an array".to_owned())?;
    let expected_shape = [1, packed_sequence_length, 128];
    if shape.len() != expected_shape.len()
        || shape
            .iter()
            .zip(expected_shape)
            .any(|(dimension, expected_dimension)| dimension.as_u64() != Some(expected_dimension))
    {
        return Err("initial_noise shape does not match the acceptance dimensions".to_owned());
    }

    let float32_sha256 = require_string(initial_noise, "float32_sha256", "initial_noise")?;
    if float32_sha256.len() != 64
        || !float32_sha256
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err("initial_noise float32_sha256 must be lowercase 64-character hex".to_owned());
    }
    Ok(float32_sha256.to_owned())
}
