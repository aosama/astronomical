use super::{TextInferenceRequest, TextSamplingStrategy};
use crate::inference_engine::contract::PreparedInferenceRequest;
use ipc_protocol::RequestId;

#[test]
fn should_preserve_bounded_sampling_settings_in_the_inference_request() {
    let inference_request = TextInferenceRequest::new_sampling(
        RequestId::new(900),
        vec![248_045, 846, 198],
        512,
        600,
        950,
        Some(7),
    );

    assert_eq!(
        inference_request.sampling_strategy(),
        TextSamplingStrategy::TopKTopP {
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
        TextInferenceRequest::new(RequestId::new(899), vec![248_045, 846, 198], 512);

    assert_eq!(
        inference_request.sampling_strategy(),
        TextSamplingStrategy::TopKTopP {
            temperature_thousandths: 1_000,
            top_k: 20,
            top_p_thousandths: 1_000,
            seed: None,
        }
    );
}

#[test]
fn should_select_highest_logit_when_temperature_is_zero() {
    let inference_request = TextInferenceRequest::new_sampling(
        RequestId::new(901),
        vec![248_045, 846, 198],
        512,
        0,
        950,
        Some(7),
    );

    assert_eq!(
        inference_request.sampling_strategy(),
        TextSamplingStrategy::HighestLogit
    );
}

#[cfg(feature = "direct-mlx")]
#[test]
fn should_carry_validated_image_pad_token_id_in_the_inference_request() {
    let inference_request = TextInferenceRequest::new_sampling(
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
fn should_offer_a_streaming_retry_by_default() {
    let inference_request = TextInferenceRequest::new(RequestId::new(0), vec![248_045], 1);

    assert!(inference_request.clone_for_streaming_retry().is_some());
}

#[test]
fn should_refuse_a_streaming_retry_once_it_is_forbidden() {
    let inference_request = TextInferenceRequest::new(RequestId::new(0), vec![248_045], 1)
        .with_streaming_retry_allowed(false);

    assert!(
        inference_request.clone_for_streaming_retry().is_none(),
        "a family without a streaming engine must never be replayed on one"
    );
}

#[test]
fn should_refuse_a_streaming_retry_of_an_empty_prompt_once_forbidden() {
    let inference_request = TextInferenceRequest::new(RequestId::new(0), Vec::new(), 1)
        .with_streaming_retry_allowed(false);

    assert!(inference_request.clone_for_streaming_retry().is_none());
}
