use crate::request_budget::{
    MAX_IMAGE_PARTS_PER_REQUEST, MAX_INLINE_MEDIA_BYTES, MAX_OUTPUT_TOKENS,
};

use super::{
    RequestBoundsError, validate_image_part_count, validate_inline_media_total,
    validate_output_token_count, validate_sampling_parameter,
};

#[test]
fn should_accept_the_image_budget_itself_and_refuse_one_image_more() {
    assert_eq!(
        validate_image_part_count(MAX_IMAGE_PARTS_PER_REQUEST),
        Ok(())
    );
    let over_budget = validate_image_part_count(MAX_IMAGE_PARTS_PER_REQUEST + 1)
        .expect_err("one image over the budget");
    assert_eq!(
        over_budget,
        RequestBoundsError::ImagePartCountExceedsBudget {
            actual_image_part_count: MAX_IMAGE_PARTS_PER_REQUEST + 1,
            maximum_image_part_count: MAX_IMAGE_PARTS_PER_REQUEST,
            excess_image_part_count: 1,
        }
    );
    assert!(over_budget.is_request_too_large());
}

#[test]
fn should_accept_the_inline_media_budget_itself_and_refuse_one_byte_more() {
    assert_eq!(validate_inline_media_total(MAX_INLINE_MEDIA_BYTES), Ok(()));
    let over_budget = validate_inline_media_total(MAX_INLINE_MEDIA_BYTES + 1)
        .expect_err("one byte over the budget");
    assert!(over_budget.is_request_too_large());
    assert!(matches!(
        over_budget,
        RequestBoundsError::InlineMediaExceedsBudget {
            excess_inline_media_bytes: 1,
            ..
        }
    ));
}

#[test]
fn should_accept_one_to_the_maximum_output_tokens_and_refuse_zero_or_more() {
    assert_eq!(validate_output_token_count(1), Ok(()));
    assert_eq!(validate_output_token_count(MAX_OUTPUT_TOKENS), Ok(()));
    for out_of_range_tokens in [0, MAX_OUTPUT_TOKENS + 1] {
        let refusal = validate_output_token_count(out_of_range_tokens)
            .expect_err("outside the output token range");
        assert!(!refusal.is_request_too_large());
    }
}

#[test]
fn should_accept_an_absent_or_bounded_sampling_setting_and_refuse_others() {
    assert_eq!(validate_sampling_parameter("top_p", None, 0.0, 1.0), Ok(()));
    assert_eq!(
        validate_sampling_parameter("top_p", Some(0.0), 0.0, 1.0),
        Ok(())
    );
    assert_eq!(
        validate_sampling_parameter("top_p", Some(1.0), 0.0, 1.0),
        Ok(())
    );
    for refused_value in [1.01, -0.01, f32::NAN, f32::INFINITY] {
        assert_eq!(
            validate_sampling_parameter("top_p", Some(refused_value), 0.0, 1.0),
            Err(RequestBoundsError::SamplingParameterOutOfRange {
                parameter_name: "top_p",
                minimum: "0".to_owned(),
                maximum: "1".to_owned(),
            }),
            "{refused_value}"
        );
    }
}
