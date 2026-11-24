//! Unit tests for laying out questions with different option counts.

use serial_test::file_serial;

use super::*;
use model_primitives::synthetic_weights_test_support::initialize_test_runtime;

#[test]
fn should_number_each_batch_option_by_its_dataset_row_and_batch_question() {
    let layout = OptionLayout::new(vec![2, 3, 4]);
    let batch_rows = layout.batch_rows(&[2, 0]);
    assert_eq!(batch_rows.dataset_option_rows, vec![5, 6, 7, 8, 0, 1]);
    assert_eq!(batch_rows.option_question, vec![0, 0, 0, 0, 1, 1]);
    assert_eq!(batch_rows.option_index, vec![0, 1, 2, 3, 0, 1]);
    assert_eq!(batch_rows.widest_option_count, 4);
    assert_eq!(batch_rows.dataset_question_rows, vec![2, 0]);
    assert!(layout.batch_rows(&[]).dataset_option_rows.is_empty());
}

#[test]
#[file_serial]
fn should_pad_scores_far_below_any_real_score_and_unpad_them_in_order() {
    let runtime = initialize_test_runtime();
    let math = GpuMath { runtime: &runtime };
    let layout = OptionLayout::new(vec![1, 3]);
    let question_batch = QuestionBatch::upload(math, &layout.batch_rows(&[0, 1])).unwrap();
    let scores = math.upload(&[5.0, 1.0, 2.0, 3.0], &[4]).unwrap();
    let padded = math
        .download(&question_batch.padded(math, &scores).unwrap())
        .unwrap();
    assert_eq!(&padded[0..1], &[5.0]);
    assert!(padded[1] < -1e29 && padded[2] < -1e29);
    assert_eq!(&padded[3..6], &[1.0, 2.0, 3.0]);
    let padded_array = question_batch.padded(math, &scores).unwrap();
    let unpadded = math
        .download(&question_batch.unpadded(math, &padded_array).unwrap())
        .unwrap();
    assert_eq!(unpadded, vec![5.0, 1.0, 2.0, 3.0]);
}
