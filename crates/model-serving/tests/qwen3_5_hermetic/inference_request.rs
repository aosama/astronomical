use astronomical_ipc_protocol::RequestId;
use astronomical_model_serving::{Qwen3_5InferenceRequest, Qwen3_5SamplingStrategy};

#[test]
fn should_preserve_bounded_qwen3_5_moe_sampling_settings_in_the_inference_request() {
    let inference_request = Qwen3_5InferenceRequest::new_sampling(
        RequestId::new(900),
        vec![248_045, 846, 198],
        512,
        600,
        950,
        Some(7),
    );

    assert_eq!(
        inference_request.sampling_strategy(),
        Qwen3_5SamplingStrategy::TopKTopP {
            temperature_thousandths: 600,
            top_k: 20,
            top_p_thousandths: 950,
            seed: Some(7),
        }
    );
}

#[test]
fn should_use_temperature_one_when_sampling_settings_are_omitted() {
    let inference_request =
        Qwen3_5InferenceRequest::new(RequestId::new(899), vec![248_045, 846, 198], 512);

    assert_eq!(
        inference_request.sampling_strategy(),
        Qwen3_5SamplingStrategy::TopKTopP {
            temperature_thousandths: 1_000,
            top_k: 20,
            top_p_thousandths: 1_000,
            seed: None,
        }
    );
}

#[test]
fn should_select_highest_logit_when_temperature_is_zero() {
    let inference_request = Qwen3_5InferenceRequest::new_sampling(
        RequestId::new(901),
        vec![248_045, 846, 198],
        512,
        0,
        950,
        Some(7),
    );

    assert_eq!(
        inference_request.sampling_strategy(),
        Qwen3_5SamplingStrategy::HighestLogit
    );
}

#[test]
fn should_carry_validated_image_pad_token_id_in_the_inference_request() {
    let inference_request = Qwen3_5InferenceRequest::new_sampling(
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

#[test]
fn should_carry_the_ordinary_target_prefill_control_span_token_count() {
    let inference_request = Qwen3_5InferenceRequest::new_sampling(
        RequestId::new(903),
        vec![101, 102, 103, 201, 202, 301],
        16,
        0,
        950,
        None,
    )
    .with_ordinary_target_prefill_control_span_token_count(3);

    assert_eq!(
        inference_request.ordinary_target_prefill_control_span_token_count(),
        3
    );
}
