//! The input rules both gated-delta paths enforce: the ops-based recurrent
//! step and the fused sequence kernels (issue #1998).
//!
//! Each rule here is plain data over shapes and dtypes, so a fix to one rule
//! reaches both paths. Each path still owns its rank checks, its token axis,
//! the fused kernel's own limits, its error text, and the order it applies
//! the rules in. That order is part of each path's contract, because a caller
//! reports the first broken rule: with 64-wide keys the sequence path reports
//! that the key dimension must be 128, not that the recurrent state shape no
//! longer fits, because it checks kernel limits before companion shapes.

use mlx_bridge::MlxDtype;

/// The head geometry both paths read from their queries and values.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct GatedDeltaHeadGeometry {
    pub(super) batch_size: i32,
    pub(super) key_head_count: i32,
    pub(super) key_head_dimension: i32,
    pub(super) value_head_count: i32,
    pub(super) value_head_dimension: i32,
}

/// Which head-geometry rule an input breaks.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum GatedDeltaHeadGeometryViolation {
    NonPositiveDimension,
    ValueHeadsNotMultipleOfKeyHeads,
}

impl GatedDeltaHeadGeometry {
    /// Every dimension must be positive, and the value heads must split
    /// evenly over the key heads, because each key head's queries and keys
    /// are repeated across its group of value heads. Positivity is checked
    /// first so a zero key-head count never reaches the remainder.
    pub(super) fn check(&self) -> Result<(), GatedDeltaHeadGeometryViolation> {
        let head_dimensions = [
            self.batch_size,
            self.key_head_count,
            self.key_head_dimension,
            self.value_head_count,
            self.value_head_dimension,
        ];
        if head_dimensions.into_iter().min() <= Some(0) {
            return Err(GatedDeltaHeadGeometryViolation::NonPositiveDimension);
        }
        if self.value_head_count % self.key_head_count != 0 {
            return Err(GatedDeltaHeadGeometryViolation::ValueHeadsNotMultipleOfKeyHeads);
        }
        Ok(())
    }

    /// The recurrent state holds one `[value dimension, key dimension]`
    /// matrix per value head and batch row.
    pub(super) fn expected_recurrent_state_shape(&self) -> [i32; 4] {
        [
            self.batch_size,
            self.value_head_count,
            self.value_head_dimension,
            self.key_head_dimension,
        ]
    }
}

/// Which dtype rule an input breaks.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum GatedDeltaDtypeViolation {
    RecurrentStateNotFloat32,
    UnsupportedActivationDtype,
}

/// The state accumulates across tokens in float32 whatever the activation
/// dtype, and the kernels are written for exactly that state type, so any
/// other state is refused; a narrower one would also drift over a long
/// prompt. The state is checked before the activations. `activation_dtypes` holds the queries,
/// keys, values, decays and update rates.
pub(super) fn check_gated_delta_dtypes(
    recurrent_state_dtype: MlxDtype,
    activation_dtypes: [MlxDtype; 5],
) -> Result<(), GatedDeltaDtypeViolation> {
    if recurrent_state_dtype != MlxDtype::Float32 {
        return Err(GatedDeltaDtypeViolation::RecurrentStateNotFloat32);
    }
    if !activation_dtypes
        .into_iter()
        .all(is_supported_activation_dtype)
    {
        return Err(GatedDeltaDtypeViolation::UnsupportedActivationDtype);
    }
    Ok(())
}

fn is_supported_activation_dtype(activation_dtype: MlxDtype) -> bool {
    matches!(
        activation_dtype,
        MlxDtype::Float16 | MlxDtype::BFloat16 | MlxDtype::Float32
    )
}

#[cfg(test)]
#[path = "gated_delta_input_rules_tests.rs"]
mod gated_delta_input_rules_tests;
