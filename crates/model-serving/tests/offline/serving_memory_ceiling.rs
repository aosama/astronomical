use astronomical_model_serving::{
    effective_mlx_serving_memory_ceiling_bytes, recommended_gpu_serving_memory_ceiling_bytes,
};

#[test]
fn should_reserve_the_minimum_headroom_for_smaller_working_sets() {
    assert_eq!(
        recommended_gpu_serving_memory_ceiling_bytes(8_000_000_000),
        5_852_516_352
    );
}

#[test]
fn should_reserve_one_eighth_when_it_is_within_the_headroom_bounds() {
    assert_eq!(
        recommended_gpu_serving_memory_ceiling_bytes(32_000_000_000),
        28_000_000_000
    );
}

#[test]
fn should_cap_headroom_for_larger_working_sets() {
    assert_eq!(
        recommended_gpu_serving_memory_ceiling_bytes(80_000_000_000),
        71_410_065_408
    );
}

#[test]
fn should_limit_the_effective_ceiling_by_config_machine_and_serving_bounds() {
    for configured_mlx_memory_ceiling_bytes in [
        23_000_000_000,
        28_000_000_000,
        32_000_000_000,
        35_000_000_000,
    ] {
        assert_eq!(
            effective_mlx_serving_memory_ceiling_bytes(
                Some(configured_mlx_memory_ceiling_bytes),
                40_000_000_000,
                35_000_000_000
            ),
            configured_mlx_memory_ceiling_bytes
        );
    }
    for configured_mlx_memory_ceiling_bytes in [38_000_000_000, 40_000_000_000] {
        assert_eq!(
            effective_mlx_serving_memory_ceiling_bytes(
                Some(configured_mlx_memory_ceiling_bytes),
                40_000_000_000,
                35_000_000_000
            ),
            35_000_000_000
        );
    }
    assert_eq!(
        effective_mlx_serving_memory_ceiling_bytes(
            Some(40_000_000_000),
            32_000_000_000,
            35_000_000_000
        ),
        32_000_000_000
    );
}

#[test]
fn should_use_the_lower_machine_or_serving_ceiling_without_an_override() {
    assert_eq!(
        effective_mlx_serving_memory_ceiling_bytes(None, 40_000_000_000, 35_000_000_000),
        35_000_000_000
    );
}
