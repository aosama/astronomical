use super::{
    VisualEmbeddingValidationError, count_image_pad_placeholder_tokens,
    validate_precomputed_visual_embedding_buffer_size,
    validate_precomputed_visual_embedding_row_counts,
};

#[test]
fn should_accept_row_counts_that_match_the_image_pad_placeholders() {
    let validation_result = validate_precomputed_visual_embedding_row_counts(4, 4, 4 * 128);

    assert_eq!(validation_result, Ok(()));
}

#[test]
fn should_reject_empty_visual_embedding_values() {
    let validation_result = validate_precomputed_visual_embedding_row_counts(4, 4, 0);

    assert_eq!(
        validation_result,
        Err(VisualEmbeddingValidationError::EmptyVisualEmbeddings)
    );
}

#[test]
fn should_reject_a_zero_visual_embedding_row_count() {
    let validation_result = validate_precomputed_visual_embedding_row_counts(0, 0, 64);

    assert_eq!(
        validation_result,
        Err(VisualEmbeddingValidationError::EmptyVisualEmbeddings)
    );
}

#[test]
fn should_reject_an_image_pad_count_that_differs_from_the_row_count() {
    let validation_result = validate_precomputed_visual_embedding_row_counts(3, 4, 4 * 128);

    assert_eq!(
        validation_result,
        Err(VisualEmbeddingValidationError::ImagePadTokenCountMismatch)
    );
}

#[test]
fn should_accept_a_buffer_that_exactly_matches_rows_times_hidden_size() {
    let validation_result = validate_precomputed_visual_embedding_buffer_size(4, 4 * 128, 128);

    assert_eq!(validation_result, Ok(()));
}

#[test]
fn should_reject_a_buffer_shorter_than_rows_times_hidden_size() {
    let validation_result = validate_precomputed_visual_embedding_buffer_size(4, 100, 128);

    assert_eq!(
        validation_result,
        Err(VisualEmbeddingValidationError::HiddenBufferSizeMismatch)
    );
}

#[test]
fn should_reject_any_buffer_when_the_hidden_size_is_zero() {
    let validation_result = validate_precomputed_visual_embedding_buffer_size(4, 4, 0);

    assert_eq!(
        validation_result,
        Err(VisualEmbeddingValidationError::HiddenBufferSizeMismatch)
    );
}

#[test]
fn should_count_every_image_pad_placeholder_in_a_mixed_prompt() {
    let prompt_token_ids = [248_045, 151_655, 846, 151_655, 151_655];

    let image_pad_placeholder_token_count =
        count_image_pad_placeholder_tokens(&prompt_token_ids, 151_655);

    assert_eq!(image_pad_placeholder_token_count, 3);
}

#[test]
fn should_count_zero_placeholders_for_a_text_only_prompt() {
    let prompt_token_ids = [248_045, 846, 198];

    let image_pad_placeholder_token_count =
        count_image_pad_placeholder_tokens(&prompt_token_ids, 151_655);

    assert_eq!(image_pad_placeholder_token_count, 0);
}

#[test]
fn should_count_an_all_placeholder_prompt_and_an_empty_prompt() {
    assert_eq!(
        count_image_pad_placeholder_tokens(&[151_655, 151_655], 151_655),
        2
    );
    assert_eq!(count_image_pad_placeholder_tokens(&[], 151_655), 0);
}

#[test]
fn should_reject_a_buffer_when_the_saturating_row_hidden_product_exceeds_any_buffer() {
    // Saturating multiplication at extreme geometry can never equal a real
    // buffer length, so the mismatch must still classify as a buffer failure.
    let visual_embedding_row_count = usize::MAX / 2;

    let validation_result =
        validate_precomputed_visual_embedding_buffer_size(visual_embedding_row_count, 1_024, 4);

    assert_eq!(
        validation_result,
        Err(VisualEmbeddingValidationError::HiddenBufferSizeMismatch)
    );
}
