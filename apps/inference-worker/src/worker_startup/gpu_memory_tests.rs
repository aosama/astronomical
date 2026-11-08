use astronomical_model_serving::machine_ceiling_from_physical_ram_bytes;

const BYTES_PER_MARKETED_GIGABYTE: u64 = 1_073_741_824;

#[test]
fn should_resolve_the_machine_ceiling_as_the_whole_of_physical_memory() {
    // A machine sold as 48 GB reports 51,539,607,552 bytes; all of it is the
    // ceiling, because MLX and macOS, not Astronomical, handle memory pressure.
    for sticker_gigabytes in [16_u64, 24, 48] {
        let physical_ram_bytes = sticker_gigabytes * BYTES_PER_MARKETED_GIGABYTE;
        assert_eq!(
            machine_ceiling_from_physical_ram_bytes(physical_ram_bytes),
            physical_ram_bytes
        );
    }
}

#[test]
fn should_give_an_8_gb_mac_its_whole_memory_instead_of_a_1_gb_floor() {
    // The fixed 8 GB reserve used to leave this Mac a 1 GB ceiling, which
    // refused every catalog model (#1902).
    assert_eq!(
        machine_ceiling_from_physical_ram_bytes(8 * BYTES_PER_MARKETED_GIGABYTE),
        8 * BYTES_PER_MARKETED_GIGABYTE
    );
}

#[test]
fn should_keep_the_smallest_and_largest_readings_unchanged() {
    assert_eq!(machine_ceiling_from_physical_ram_bytes(1), 1);
    assert_eq!(machine_ceiling_from_physical_ram_bytes(u64::MAX), u64::MAX);
}
