//! Whether one tensor, read from two artifacts, is really the same tensor.

use serde::Serialize;

use crate::tensor_values::decode_float_values;

// bfloat16 keeps 8 significant bits, so one conversion can move a value by up to
// about 0.4 percent; this allows two conversions on the way to a comparison.
const ROUNDING_RELATIVE_TOLERANCE: f32 = 0.01;
const ROUNDING_ABSOLUTE_TOLERANCE: f32 = 1e-6;
// Measured on real pairs: 4-bit norm weights minus the upstream's ran from 0.9961
// to 1.0078, the same bfloat16 rounding around the +1 storage offset (issue #1615).
const STORAGE_OFFSET: f32 = 1.0;
const STORAGE_OFFSET_TOLERANCE: f32 = 0.02;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TensorAgreement {
    /// Byte for byte the same.
    Identical,
    /// The same values at a different precision.
    EqualAfterRounding,
    /// A norm weight kept as `value − 1` on one side, a storage convention.
    OffsetByOne,
    /// Different values, so this tensor has been trained or edited separately.
    Different,
    /// Every value is the same constant. Two unrelated models can both have an
    /// all-zero tensor, so a match here proves nothing and it is not counted.
    Uninformative,
}

impl TensorAgreement {
    /// Whether the pair counts as evidence of one shared base.
    #[must_use]
    pub const fn is_shared(self) -> bool {
        matches!(
            self,
            Self::Identical | Self::EqualAfterRounding | Self::OffsetByOne
        )
    }
}

/// One side of a comparison.
#[derive(Clone, Copy, Debug)]
pub struct TensorSide<'tensor> {
    pub data_type: &'tensor str,
    pub tensor_bytes: &'tensor [u8],
}

#[must_use]
pub fn classify_tensor_pair(
    tensor_name: &str,
    first_side: TensorSide<'_>,
    second_side: TensorSide<'_>,
) -> TensorAgreement {
    let (Some(first_values), Some(second_values)) = (
        decode_float_values(first_side.data_type, first_side.tensor_bytes),
        decode_float_values(second_side.data_type, second_side.tensor_bytes),
    ) else {
        return TensorAgreement::Different;
    };
    if first_values.len() != second_values.len() {
        return TensorAgreement::Different;
    }
    if is_constant(&first_values) && is_constant(&second_values) {
        return TensorAgreement::Uninformative;
    }
    if first_side.data_type == second_side.data_type
        && first_side.tensor_bytes == second_side.tensor_bytes
    {
        return TensorAgreement::Identical;
    }
    if is_equal_after_rounding(&first_values, &second_values) {
        return TensorAgreement::EqualAfterRounding;
    }
    // Only norms use this convention. Elsewhere a uniform +1 is a real difference.
    if tensor_name.contains("norm") && is_offset_by_one(&first_values, &second_values) {
        return TensorAgreement::OffsetByOne;
    }
    TensorAgreement::Different
}

fn is_constant(float_values: &[f32]) -> bool {
    let Some(first_value) = float_values.first() else {
        return true;
    };
    for float_value in float_values {
        if float_value != first_value {
            return false;
        }
    }
    true
}

fn is_equal_after_rounding(first_values: &[f32], second_values: &[f32]) -> bool {
    for (first_value, second_value) in first_values.iter().zip(second_values) {
        let allowed_difference = ROUNDING_RELATIVE_TOLERANCE
            * first_value.abs().max(second_value.abs())
            + ROUNDING_ABSOLUTE_TOLERANCE;
        // Written as a positive test so a NaN, which compares false to everything,
        // counts as a difference rather than slipping through as a match.
        let is_close = (first_value - second_value).abs() <= allowed_difference;
        if !is_close {
            return false;
        }
    }
    true
}

/// The offset may be on either side, but must run one way across the tensor.
fn is_offset_by_one(first_values: &[f32], second_values: &[f32]) -> bool {
    let mut is_first_higher = true;
    let mut is_second_higher = true;
    for (first_value, second_value) in first_values.iter().zip(second_values) {
        let difference = first_value - second_value;
        is_first_higher &= (difference - STORAGE_OFFSET).abs() <= STORAGE_OFFSET_TOLERANCE;
        is_second_higher &= (difference + STORAGE_OFFSET).abs() <= STORAGE_OFFSET_TOLERANCE;
    }
    is_first_higher || is_second_higher
}

#[cfg(test)]
#[path = "tensor_agreement_tests.rs"]
mod tests;
