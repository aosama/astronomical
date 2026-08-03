use astronomical_model_serving::resolve_sampling_seed;

#[test]
fn should_use_the_client_provided_seed_when_one_is_supplied() {
    let resolved_seed = resolve_sampling_seed(Some(42), || 9_999_999);

    assert_eq!(resolved_seed, 42);
}

#[test]
fn should_use_a_time_based_seed_when_the_client_omits_seed() {
    let resolved_seed = resolve_sampling_seed(None, || 1_700_000_000_000);

    assert_eq!(
        resolved_seed, 1_700_000_000_000,
        "when the client sends seed: null, Astronomical should use a time-based seed \
         matching MLX's default behavior, not the request_id which makes generation \
         deterministic per request and causes repeated wrong outputs"
    );
}

#[test]
fn should_produce_different_seeds_for_different_time_based_calls() {
    let first_seed = resolve_sampling_seed(None, || 1_000);
    let second_seed = resolve_sampling_seed(None, || 2_000);

    assert_ne!(
        first_seed, second_seed,
        "consecutive requests without an explicit seed should get different sampling \
          entropy when seed is null"
    );
}
