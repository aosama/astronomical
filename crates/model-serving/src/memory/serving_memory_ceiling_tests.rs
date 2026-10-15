use astronomical_test_harness::offline_denied_test;

use crate::effective_mlx_memory_ceiling_bytes;

offline_denied_test!(
    should_limit_the_effective_ceiling_by_the_configured_and_machine_bounds,
    {
        for configured_mlx_memory_ceiling_bytes in [
            23_000_000_000,
            28_000_000_000,
            32_000_000_000,
            35_000_000_000,
        ] {
            assert_eq!(
                effective_mlx_memory_ceiling_bytes(
                    Some(configured_mlx_memory_ceiling_bytes),
                    42_949_672_960
                ),
                configured_mlx_memory_ceiling_bytes
            );
        }
        for configured_mlx_memory_ceiling_bytes in [43_000_000_000, 51_000_000_000] {
            assert_eq!(
                effective_mlx_memory_ceiling_bytes(
                    Some(configured_mlx_memory_ceiling_bytes),
                    42_949_672_960
                ),
                42_949_672_960
            );
        }
    }
);

offline_denied_test!(
    should_use_the_machine_ceiling_without_a_configured_override,
    {
        assert_eq!(
            effective_mlx_memory_ceiling_bytes(None, 42_949_672_960),
            42_949_672_960
        );
        assert_eq!(effective_mlx_memory_ceiling_bytes(None, 0), 0);
    }
);
