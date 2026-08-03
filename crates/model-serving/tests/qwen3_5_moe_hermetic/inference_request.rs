use astronomical_ipc_protocol::RequestId;
use astronomical_model_serving::{Qwen3_5MoEInferenceRequest, Qwen3_5MoESamplingStrategy};

#[test]
fn should_preserve_bounded_qwen3_5_moe_sampling_settings_in_the_inference_request() {
    let inference_request = Qwen3_5MoEInferenceRequest::new_sampling(
        RequestId::new(900),
        vec![248_045, 846, 198],
        512,
        600,
        950,
        Some(7),
    );

    assert_eq!(
        inference_request.sampling_strategy(),
        Qwen3_5MoESamplingStrategy::TopKTopP {
            temperature_thousandths: 600,
            top_k: 20,
            top_p_thousandths: 950,
            seed: Some(7),
        }
    );
}

#[test]
fn should_treat_zero_temperature_as_greedy_generation() {
    let inference_request = Qwen3_5MoEInferenceRequest::new_sampling(
        RequestId::new(901),
        vec![248_045, 846, 198],
        512,
        0,
        950,
        Some(7),
    );

    assert_eq!(
        inference_request.sampling_strategy(),
        Qwen3_5MoESamplingStrategy::Greedy
    );
}

#[test]
fn should_carry_validated_image_pad_token_id_in_the_inference_request() {
    let inference_request = Qwen3_5MoEInferenceRequest::new_sampling(
        RequestId::new(902),
        vec![248_056, 42],
        16,
        1_000,
        950,
        None,
    )
    .with_image_pad_token_id(248_056);

    assert_eq!(inference_request.image_pad_token_id(), Some(248_056));
}
