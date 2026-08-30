#[test]
fn should_use_the_configured_model_artifact_memory_ceiling_when_it_is_lower_than_the_machine() {
    assert_eq!(
        crate::common::resolve_serving_acceptance_mlx_memory_ceiling_bytes(
            Some(35_000_000_000),
            40_000_000_000,
        ),
        35_000_000_000,
    );
}

#[test]
fn should_never_raise_model_artifact_memory_above_the_machine_ceiling() {
    assert_eq!(
        crate::common::resolve_serving_acceptance_mlx_memory_ceiling_bytes(
            Some(45_000_000_000),
            40_000_000_000,
        ),
        40_000_000_000,
    );
}

#[test]
fn should_use_the_machine_model_artifact_memory_ceiling_when_no_user_limit_exists() {
    assert_eq!(
        crate::common::resolve_serving_acceptance_mlx_memory_ceiling_bytes(None, 40_000_000_000,),
        40_000_000_000,
    );
}
