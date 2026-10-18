//! Minimum ceiling policy for models that retain their complete dense weights.

/// Decode workspace reserved above the complete resident weight payload.
const DENSE_RESIDENT_DECODE_HEADROOM_BYTES: u64 = 1024 * 1024 * 1024;

/// Returns the minimum ceiling that keeps a dense resident model and its decode workspace live.
#[must_use]
pub(crate) const fn dense_resident_safe_minimum_mlx_memory_ceiling_bytes(
    complete_weight_payload_bytes: u64,
) -> u64 {
    complete_weight_payload_bytes.saturating_add(DENSE_RESIDENT_DECODE_HEADROOM_BYTES)
}

#[cfg(test)]
mod tests {
    #[test]
    fn should_reserve_decode_headroom_above_complete_dense_weights() {
        let complete_weight_payload_bytes = 8 * 1024 * 1024 * 1024;

        assert_eq!(
            super::dense_resident_safe_minimum_mlx_memory_ceiling_bytes(
                complete_weight_payload_bytes
            ),
            9 * 1024 * 1024 * 1024
        );
    }

    #[test]
    fn should_saturate_the_dense_resident_floor_on_byte_count_overflow() {
        assert_eq!(
            super::dense_resident_safe_minimum_mlx_memory_ceiling_bytes(u64::MAX),
            u64::MAX
        );
    }
}
