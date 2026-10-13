use astronomical_model_serving::{
    compute_default_rope_frequency_denominators, compute_yarn_rope_frequency_denominators,
};

#[test]
fn should_keep_high_frequency_yarn_pairs_and_scale_low_frequency_pairs() {
    let default_denominators = compute_default_rope_frequency_denominators(500_000.0, 64)
        .expect("default denominators should exist");
    let yarn_denominators =
        compute_yarn_rope_frequency_denominators(500_000.0, 64, 8_192, 32.0, 32.0, 1.0)
            .expect("valid YaRN geometry should produce denominators");
    let blended = yarn_denominators.frequency_denominators();
    assert!((blended[0] - default_denominators[0]).abs() < 1e-5);
    let last_index = blended.len() - 1;
    let expected_scaled = default_denominators[last_index] * 32.0;
    assert!((blended[last_index] - expected_scaled).abs() <= expected_scaled * 1e-5);
}

#[test]
fn should_reject_invalid_rotary_and_yarn_geometry() {
    assert!(compute_default_rope_frequency_denominators(10_000.0, 0).is_err());
    assert!(compute_default_rope_frequency_denominators(10_000.0, 7).is_err());
    assert!(compute_yarn_rope_frequency_denominators(500_000.0, 64, 0, 32.0, 32.0, 1.0).is_err());
    assert!(
        compute_yarn_rope_frequency_denominators(500_000.0, 64, 8_192, 32.0, 1.0, 32.0).is_err()
    );
}
