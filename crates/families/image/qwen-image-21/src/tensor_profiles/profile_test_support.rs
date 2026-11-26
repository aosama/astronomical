//! Shared lookups for the tensor-profile builder tests: where a named tensor sits in the emitted
//! order, and what its dtype and shape are. Validation matches tensors by name, so production
//! does not depend on the order; the tests pin it only to prove that splitting the builders kept
//! the sequence they emitted before the split.

use std::collections::BTreeSet;

use super::QwenImage21TensorProfile;

pub(super) fn position_of(profiles: &[QwenImage21TensorProfile], tensor_name: &str) -> usize {
    for (position, profile) in profiles.iter().enumerate() {
        if profile.tensor_name == tensor_name {
            return position;
        }
    }
    panic!("the profile list should name {tensor_name}");
}

pub(super) fn has_tensor(profiles: &[QwenImage21TensorProfile], tensor_name: &str) -> bool {
    profiles
        .iter()
        .any(|profile| profile.tensor_name == tensor_name)
}

/// Fails when any tensor name appears twice, since an inventory keyed by name would silently
/// drop one of the duplicates.
pub(super) fn assert_every_name_is_unique(profiles: &[QwenImage21TensorProfile]) {
    let distinct_names: BTreeSet<&str> = profiles
        .iter()
        .map(|profile| profile.tensor_name.as_str())
        .collect();
    assert_eq!(distinct_names.len(), profiles.len());
}

pub(super) fn assert_profile_at(
    profiles: &[QwenImage21TensorProfile],
    position: usize,
    expected_name: &str,
    expected_dtype: &str,
    expected_shape: &[usize],
) {
    let profile = &profiles[position];
    assert_eq!(profile.tensor_name, expected_name);
    assert_eq!(profile.dtype, expected_dtype);
    assert_eq!(profile.shape, expected_shape);
}

/// Checks the three tensors `quantized_linear` emits for one projection, starting at `position`:
/// packed `U32` weight rows, then `BF16` scales and biases per group.
pub(super) fn assert_quantized_linear_at(
    profiles: &[QwenImage21TensorProfile],
    position: usize,
    prefix: &str,
    output_features: usize,
    packed_row_count: usize,
    group_count: usize,
) {
    assert_profile_at(
        profiles,
        position,
        &format!("{prefix}.weight"),
        "U32",
        &[output_features, packed_row_count],
    );
    assert_profile_at(
        profiles,
        position + 1,
        &format!("{prefix}.scales"),
        "BF16",
        &[output_features, group_count],
    );
    assert_profile_at(
        profiles,
        position + 2,
        &format!("{prefix}.biases"),
        "BF16",
        &[output_features, group_count],
    );
}

/// Fails unless every named tensor is present and the names appear in exactly the order given.
pub(super) fn assert_names_appear_in_order(
    profiles: &[QwenImage21TensorProfile],
    expected_name_order: &[&str],
) {
    let mut previous_position = None;
    for tensor_name in expected_name_order {
        let position = position_of(profiles, tensor_name);
        if let Some(earlier_position) = previous_position {
            assert!(
                earlier_position < position,
                "{tensor_name} should come after the tensor before it"
            );
        }
        previous_position = Some(position);
    }
}
