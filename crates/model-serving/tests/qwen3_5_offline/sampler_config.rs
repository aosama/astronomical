#[test]
fn should_use_temperature_top_p_and_top_k_from_model_generation_configuration() {
    let model_generation_configuration = br#"{
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20
    }"#;

    let model_sampler_configuration =
        astronomical_model_serving::discover_sampler_config(Some(model_generation_configuration));

    assert_eq!(model_sampler_configuration.temperature_thousandths, 600);
    assert_eq!(model_sampler_configuration.top_p_thousandths, 950);
    assert_eq!(model_sampler_configuration.model_top_k, 20);
}

#[test]
fn should_use_product_sampling_defaults_only_when_the_model_has_no_generation_configuration() {
    let model_sampler_configuration = astronomical_model_serving::discover_sampler_config(None);

    assert_eq!(model_sampler_configuration.temperature_thousandths, 1_000);
    assert_eq!(model_sampler_configuration.top_p_thousandths, 950);
    assert_eq!(model_sampler_configuration.model_top_k, 20);
}

#[test]
fn should_preserve_a_model_declared_zero_temperature() {
    // A zero value is a model policy, not a missing field. This protects real-artifact
    // acceptance from silently changing a deterministic model into randomized sampling.
    let model_generation_configuration = br#"{"temperature": 0, "top_p": 0.9, "top_k": 40}"#;

    let model_sampler_configuration =
        astronomical_model_serving::discover_sampler_config(Some(model_generation_configuration));

    assert_eq!(model_sampler_configuration.temperature_thousandths, 0);
    assert_eq!(model_sampler_configuration.top_p_thousandths, 900);
    assert_eq!(model_sampler_configuration.model_top_k, 40);
}

#[test]
fn should_use_the_model_declared_repetition_penalty() {
    let model_generation_configuration =
        br#"{"temperature": 0.6, "top_p": 0.95, "top_k": 20, "repetition_penalty": 1.05}"#;

    let model_sampler_configuration =
        astronomical_model_serving::discover_sampler_config(Some(model_generation_configuration));

    assert_eq!(
        model_sampler_configuration.repetition_penalty_thousandths,
        1_050
    );
}

#[test]
fn should_fall_back_to_the_neutral_repetition_penalty_without_an_artifact_value() {
    let model_sampler_configuration = astronomical_model_serving::discover_sampler_config(None);

    assert_eq!(
        model_sampler_configuration.repetition_penalty_thousandths, 1_000,
        "an absent artifact penalty must stay neutral so seeded sampling is unchanged"
    );
}

#[test]
fn should_fall_back_to_the_neutral_repetition_penalty_for_out_of_range_artifact_values() {
    for invalid_repetition_penalty in ["0.9", "2.5", "-1"] {
        let model_generation_configuration = format!(
            r#"{{"temperature": 0.6, "top_p": 0.95, "top_k": 20, "repetition_penalty": {invalid_repetition_penalty}}}"#
        );

        let model_sampler_configuration = astronomical_model_serving::discover_sampler_config(
            Some(model_generation_configuration.as_bytes()),
        );

        assert_eq!(
            model_sampler_configuration.repetition_penalty_thousandths, 1_000,
            "artifact penalty {invalid_repetition_penalty} must not change the sampling distribution"
        );
    }
}
