use super::restore_transient_workspace_bytes;

#[test]
fn should_take_the_larger_of_the_two_phased_totals() {
    assert_eq!(
        restore_transient_workspace_bytes(1_048_576, 262_144),
        1_048_576
    );
    assert_eq!(
        restore_transient_workspace_bytes(262_144, 1_048_576),
        1_048_576
    );
}

#[test]
fn should_accept_equal_totals_and_zeroes() {
    assert_eq!(restore_transient_workspace_bytes(512, 512), 512);
    assert_eq!(restore_transient_workspace_bytes(0, 0), 0);
}

#[test]
fn should_saturate_when_the_larger_total_exceeds_the_usize_range() {
    // On 64-bit targets u64::MAX exceeds usize only through the try_from
    // guard on 32-bit; on this crate's targets the value passes through.
    assert_eq!(
        restore_transient_workspace_bytes(u64::MAX, 1),
        u64::MAX as usize
    );
}
