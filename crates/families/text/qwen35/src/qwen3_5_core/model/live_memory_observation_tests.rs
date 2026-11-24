use crate::qwen3_5_core::model::live_memory_observation::requested_mlx_memory_limits_when_changed;

const ONE_GIGABYTE_BYTES: u64 = 1_073_741_824;

#[test]
fn should_install_matching_active_and_buffer_limits_for_a_changed_ceiling() {
    let requested_limits =
        requested_mlx_memory_limits_when_changed(12 * ONE_GIGABYTE_BYTES, 36 * ONE_GIGABYTE_BYTES)
            .unwrap()
            .expect("a changed ceiling installs limits");
    assert_eq!(
        requested_limits.active_memory_limit_bytes() as u64,
        12 * ONE_GIGABYTE_BYTES
    );
    assert_eq!(
        requested_limits.mlx_buffer_cache_limit_bytes() as u64,
        12 * ONE_GIGABYTE_BYTES
    );
}

#[test]
fn should_leave_the_limits_alone_when_the_requested_ceiling_is_unchanged() {
    let requested_limits =
        requested_mlx_memory_limits_when_changed(36 * ONE_GIGABYTE_BYTES, 36 * ONE_GIGABYTE_BYTES)
            .unwrap();
    assert!(requested_limits.is_none());
}

#[test]
fn should_reject_a_zero_ceiling_that_differs_from_the_current_one() {
    let rejection = requested_mlx_memory_limits_when_changed(0, 36 * ONE_GIGABYTE_BYTES);
    assert!(rejection.is_err());
}

#[test]
fn should_accept_a_one_byte_ceiling_as_the_smallest_positive_limit() {
    let requested_limits = requested_mlx_memory_limits_when_changed(1, 36 * ONE_GIGABYTE_BYTES)
        .unwrap()
        .expect("a positive changed ceiling installs limits");
    assert_eq!(requested_limits.active_memory_limit_bytes(), 1);
}
