//! CPU inverse-permutation contract for stacked expert assignments.
//!
//! `sorted_order[sorted_slot] = original_slot`. The inverse is
//! `inverse_order[original_slot] = sorted_slot`. Gather with `sorted_indices=true`
//! is valid only after this permutation was applied to the expert ids.

use super::error::SparseExpertError;

/// Builds the inverse of a complete assignment permutation.
pub fn invert_assignment_order(sorted_order: &[u32]) -> Result<Vec<u32>, SparseExpertError> {
    let assignment_count = u32::try_from(sorted_order.len()).map_err(|_| {
        SparseExpertError::InvalidAssignmentGeometry {
            description: "assignment permutation length exceeds u32",
        }
    })?;
    let mut inverse_order = vec![0_u32; sorted_order.len()];
    let mut seen_original_slots = vec![false; sorted_order.len()];
    for (sorted_slot, original_slot) in sorted_order.iter().copied().enumerate() {
        if original_slot >= assignment_count {
            return Err(SparseExpertError::InvalidAssignmentGeometry {
                description: "assignment permutation contains an out-of-range slot",
            });
        }
        let original_slot_index = original_slot as usize;
        if seen_original_slots[original_slot_index] {
            return Err(SparseExpertError::InvalidAssignmentGeometry {
                description: "assignment permutation contains a duplicate slot",
            });
        }
        seen_original_slots[original_slot_index] = true;
        inverse_order[original_slot_index] = sorted_slot as u32;
    }
    if seen_original_slots
        .iter()
        .any(|seen_original_slot| !seen_original_slot)
    {
        return Err(SparseExpertError::InvalidAssignmentGeometry {
            description: "assignment permutation is missing a slot",
        });
    }
    Ok(inverse_order)
}
