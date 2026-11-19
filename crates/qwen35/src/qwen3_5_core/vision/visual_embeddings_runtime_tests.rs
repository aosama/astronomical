//! Beside-source coverage for the pure visual-embedding resolution units:
//! the missing-row-count fold. The array-shape checks and slicing run under
//! the serial engine lane against real arrays.

use super::fold_missing_visual_embedding_row_count;
use crate::InferenceEngineError;
use crate::qwen3_5_core::vision::visual_embeddings::Qwen3_5VisualEmbeddingRequiredImage;

#[test]
fn should_fold_the_missing_images_row_counts_in_plan_order() {
    let required_images = [
        Qwen3_5VisualEmbeddingRequiredImage::new(0, 256, 0, 256),
        Qwen3_5VisualEmbeddingRequiredImage::new(1, 128, 32, 64),
    ];

    let total_missing_visual_embedding_row_count =
        fold_missing_visual_embedding_row_count(&required_images, &[0, 1])
            .expect("two small row counts must fold without overflow");

    assert_eq!(total_missing_visual_embedding_row_count, 384);
}

#[test]
fn should_fold_zero_missing_images_to_zero_rows() {
    let required_images = [Qwen3_5VisualEmbeddingRequiredImage::new(0, 256, 0, 256)];

    let total_missing_visual_embedding_row_count =
        fold_missing_visual_embedding_row_count(&required_images, &[])
            .expect("an empty missing set folds to zero");

    assert_eq!(total_missing_visual_embedding_row_count, 0);
}

#[test]
fn should_reject_a_missing_row_count_that_overflows() {
    let required_images = [
        Qwen3_5VisualEmbeddingRequiredImage::new(0, usize::MAX, 0, usize::MAX),
        Qwen3_5VisualEmbeddingRequiredImage::new(1, usize::MAX, 0, usize::MAX),
    ];

    let outcome = fold_missing_visual_embedding_row_count(&required_images, &[0, 1]);

    assert!(
        matches!(&outcome, Err(InferenceEngineError::Fatal { reason })
            if reason == "missing visual embedding row count overflowed"),
        "overflow must surface the fatal wrap, not wrap silently: {outcome:?}"
    );
}
