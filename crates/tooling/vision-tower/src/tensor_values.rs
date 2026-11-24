//! Decodes the float encodings found in checkpoints, so tensors stored at
//! different precisions can still be compared value by value.

/// The tensor's values as `f32`, or `None` for a type that is not a float or
/// bytes that are not a whole number of elements.
#[must_use]
pub fn decode_float_values(data_type: &str, tensor_bytes: &[u8]) -> Option<Vec<f32>> {
    match data_type {
        "BF16" => decode_with_width::<2>(tensor_bytes, bfloat16_to_f32),
        "F16" => decode_with_width::<2>(tensor_bytes, half_to_f32),
        "F32" => decode_with_width::<4>(tensor_bytes, f32::from_le_bytes),
        "F64" => decode_with_width::<8>(tensor_bytes, f64_to_f32),
        _ => None,
    }
}

#[must_use]
pub fn is_float_data_type(data_type: &str) -> bool {
    matches!(data_type, "BF16" | "F16" | "F32" | "F64")
}

fn decode_with_width<const WIDTH: usize>(
    tensor_bytes: &[u8],
    decode_element: fn([u8; WIDTH]) -> f32,
) -> Option<Vec<f32>> {
    if !tensor_bytes.len().is_multiple_of(WIDTH) {
        return None;
    }
    let mut float_values = Vec::with_capacity(tensor_bytes.len() / WIDTH);
    for element_bytes in tensor_bytes.as_chunks::<WIDTH>().0 {
        float_values.push(decode_element(*element_bytes));
    }
    Some(float_values)
}

/// bfloat16 is the top half of an `f32`, which is why it widens exactly.
fn bfloat16_to_f32(element_bytes: [u8; 2]) -> f32 {
    f32::from_bits(u32::from(u16::from_le_bytes(element_bytes)) << 16)
}

fn f64_to_f32(element_bytes: [u8; 8]) -> f32 {
    f64::from_le_bytes(element_bytes) as f32
}

fn half_to_f32(element_bytes: [u8; 2]) -> f32 {
    let half_bits = u32::from(u16::from_le_bytes(element_bytes));
    let sign_bit = (half_bits & 0x8000) << 16;
    let exponent = (half_bits >> 10) & 0x1f;
    let mantissa = half_bits & 0x3ff;
    let magnitude_bits = match exponent {
        0 if mantissa == 0 => 0,
        // Subnormal halves are normal numbers in f32: shift the leading one out.
        0 => {
            let leading_shift = mantissa.leading_zeros() - 21;
            let normalized_mantissa = (mantissa << leading_shift) & 0x3ff;
            ((113 - leading_shift) << 23) | (normalized_mantissa << 13)
        }
        0x1f => 0x7f80_0000 | (mantissa << 13),
        _ => ((exponent + 112) << 23) | (mantissa << 13),
    };
    f32::from_bits(sign_bit | magnitude_bits)
}

#[cfg(test)]
#[path = "tensor_values_tests.rs"]
mod tests;
