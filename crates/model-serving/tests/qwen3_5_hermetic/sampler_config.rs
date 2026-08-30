use astronomical_model_serving::discover_sampler_config;

#[test]
fn should_use_temperature_top_p_and_top_k_from_model_generation_configuration() {
    let model_generation_configuration = br#"{
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20
    }"#;

    let model_sampler_configuration = discover_sampler_config(Some(model_generation_configuration));

    assert_eq!(model_sampler_configuration.temperature_thousandths, 600);
    assert_eq!(model_sampler_configuration.top_p_thousandths, 950);
    assert_eq!(model_sampler_configuration.model_top_k, 20);
}

#[test]
fn should_use_product_sampling_defaults_only_when_the_model_has_no_generation_configuration() {
    let model_sampler_configuration = discover_sampler_config(None);

    assert_eq!(model_sampler_configuration.temperature_thousandths, 1_000);
    assert_eq!(model_sampler_configuration.top_p_thousandths, 950);
    assert_eq!(model_sampler_configuration.model_top_k, 20);
}

#[test]
fn should_preserve_a_model_declared_zero_temperature() {
    // A zero value is a model policy, not a missing field. This protects real-artifact
    // acceptance from silently changing a deterministic model into randomized sampling.
    let model_generation_configuration = br#"{"temperature": 0, "top_p": 0.9, "top_k": 40}"#;

    let model_sampler_configuration = discover_sampler_config(Some(model_generation_configuration));

    assert_eq!(model_sampler_configuration.temperature_thousandths, 0);
    assert_eq!(model_sampler_configuration.top_p_thousandths, 900);
    assert_eq!(model_sampler_configuration.model_top_k, 40);
}
