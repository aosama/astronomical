use astronomical_model_serving::machine_ceiling_from_physical_ram_bytes;
use astronomical_test_harness::offline_denied_test;

offline_denied_test!(
    should_resolve_the_machine_ceiling_as_physical_ram_minus_the_operating_system_reserve,
    {
        // A machine sold as 48 GB reports 51,539,607,552 bytes of physical
        // RAM; the ceiling keeps the 8 sticker-scale GB for the operating
        // system, leaving the 40 GB the sticker math promises.
        assert_eq!(
            machine_ceiling_from_physical_ram_bytes(51_539_607_552),
            42_949_672_960
        );
        assert_eq!(
            machine_ceiling_from_physical_ram_bytes(16 * 1_073_741_824),
            8 * 1_073_741_824
        );
    }
);

offline_denied_test!(
    should_floor_the_machine_ceiling_so_small_machines_stay_servable,
    {
        assert_eq!(
            machine_ceiling_from_physical_ram_bytes(8_589_934_592),
            1_073_741_824
        );
        assert_eq!(machine_ceiling_from_physical_ram_bytes(0), 1_073_741_824);
    }
);
