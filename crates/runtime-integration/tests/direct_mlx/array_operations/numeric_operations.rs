use astronomical_runtime_integration::MlxDtype;

use crate::common::runtime_test_support::{assert_f32_close, runtime};

#[test]
fn should_multiply_owned_float32_arrays_on_the_runtime_stream() {
    let runtime = runtime();
    let left = runtime
        .array_from_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2])
        .expect("the left matrix should be valid");
    let right = runtime
        .array_from_f32(&[5.0, 6.0, 7.0, 8.0], &[2, 2])
        .expect("the right matrix should be valid");

    let product = runtime
        .matmul(&left, &right)
        .expect("matrix multiplication should build a valid graph");

    assert_eq!(product.shape(), vec![2, 2]);
    assert_eq!(
        product
            .to_vec_f32()
            .expect("the product should evaluate as float32"),
        vec![19.0, 22.0, 43.0, 50.0]
    );
}

#[test]
fn should_apply_fused_addmm_on_the_runtime_stream() {
    let runtime = runtime();
    let bias = runtime
        .array_from_f32(&[10.0, 20.0], &[2])
        .expect("the addmm bias should be valid");
    let left_matrix = runtime
        .array_from_f32(&[1.0, 2.0], &[1, 2])
        .expect("the addmm left matrix should be valid");
    let right_matrix = runtime
        .array_from_f32(&[3.0, 4.0, 5.0, 6.0], &[2, 2])
        .expect("the addmm right matrix should be valid");

    let affine_projection = runtime
        .addmm(&bias, &left_matrix, &right_matrix, 1.0, 1.0)
        .expect("addmm should build a valid graph");

    assert_eq!(
        affine_projection
            .to_vec_f32()
            .expect("the affine projection should evaluate as float32"),
        vec![23.0, 36.0]
    );
}

#[test]
fn should_create_zero_filled_float32_arrays_on_the_runtime_stream() {
    let runtime = runtime();

    let zeros = runtime
        .zeros(&[2, 3], MlxDtype::Float32)
        .expect("zero creation should build a valid graph");

    assert_eq!(zeros.shape(), vec![2, 3]);
    assert_eq!(
        zeros
            .to_vec_f32()
            .expect("zeros should evaluate as float32"),
        vec![0.0; 6]
    );
}

#[test]
fn should_apply_unweighted_rms_normalization_on_the_runtime_stream() {
    let runtime = runtime();
    let hidden_states = runtime
        .array_from_f32(&[3.0, 4.0], &[1, 2])
        .expect("the hidden states should be valid");

    let normalized_states = runtime
        .rms_norm_without_weight(&hidden_states, 0.0)
        .expect("unweighted RMS normalization should build a valid graph");

    assert_f32_close(
        &normalized_states
            .to_vec_f32()
            .expect("the normalized states should evaluate as float32"),
        &[0.848_528_15, 1.131_370_9],
    );
}

#[test]
fn should_apply_subtract_and_divide_on_the_runtime_stream() {
    let runtime = runtime();
    let left = runtime
        .array_from_f32(&[8.0, 9.0], &[2])
        .expect("the left vector should be valid");
    let right = runtime
        .array_from_f32(&[2.0, 3.0], &[2])
        .expect("the right vector should be valid");

    let difference = runtime
        .subtract(&left, &right)
        .expect("subtraction should build a valid graph");
    let quotient = runtime
        .divide(&left, &right)
        .expect("division should build a valid graph");

    assert_eq!(
        difference
            .to_vec_f32()
            .expect("the difference should evaluate as float32"),
        vec![6.0, 6.0]
    );
    assert_eq!(
        quotient
            .to_vec_f32()
            .expect("the quotient should evaluate as float32"),
        vec![4.0, 3.0]
    );
}

#[test]
fn should_apply_unary_math_on_the_runtime_stream() {
    let runtime = runtime();
    let signed_values = runtime
        .array_from_f32(&[1.0, 2.0], &[2])
        .expect("the signed vector should be valid");
    let exponential_inputs = runtime
        .array_from_f32(&[0.0, std::f32::consts::LN_2], &[2])
        .expect("the exponential vector should be valid");
    let logarithm_inputs = runtime
        .array_from_f32(&[0.0, 3.0], &[2])
        .expect("the logarithm vector should be valid");

    let negated_values = runtime
        .negative(&signed_values)
        .expect("negation should build a valid graph");
    let exponential_values = runtime
        .exp(&exponential_inputs)
        .expect("exponential should build a valid graph");
    let logarithm_values = runtime
        .log1p(&logarithm_inputs)
        .expect("log1p should build a valid graph");

    assert_eq!(
        negated_values
            .to_vec_f32()
            .expect("the negated vector should evaluate as float32"),
        vec![-1.0, -2.0]
    );
    assert_f32_close(
        &exponential_values
            .to_vec_f32()
            .expect("the exponential vector should evaluate as float32"),
        &[1.0, 2.0],
    );
    assert_f32_close(
        &logarithm_values
            .to_vec_f32()
            .expect("the logarithm vector should evaluate as float32"),
        &[0.0, 4.0_f32.ln()],
    );
}

#[test]
fn should_reduce_sum_and_max_along_one_axis() {
    let runtime = runtime();
    let matrix = runtime
        .array_from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])
        .expect("the matrix should be valid");

    let row_sums = runtime
        .sum_axis(&matrix, 1, false)
        .expect("sum reduction should build a valid graph");
    let column_maxima = runtime
        .max_axis(&matrix, 0, false)
        .expect("max reduction should build a valid graph");

    assert_eq!(row_sums.shape(), vec![2]);
    assert_eq!(
        row_sums
            .to_vec_f32()
            .expect("the row sums should evaluate as float32"),
        vec![6.0, 15.0]
    );
    assert_eq!(column_maxima.shape(), vec![3]);
    assert_eq!(
        column_maxima
            .to_vec_f32()
            .expect("the column maxima should evaluate as float32"),
        vec![4.0, 5.0, 6.0]
    );
}

#[test]
fn should_take_values_along_one_axis() {
    let runtime = runtime();
    let routing_scores = runtime
        .array_from_f32(&[1.0, 5.0, 3.0, 2.0, 4.0, 0.0, 7.0, 6.0], &[2, 4])
        .expect("the routing score matrix should be valid");
    let selected_indices = runtime
        .array_from_i32(&[1, 2], &[2, 1])
        .expect("the selected index matrix should be valid");

    let selected_scores = runtime
        .take_along_axis(&routing_scores, &selected_indices, 1)
        .expect("take-along-axis should build a valid graph");

    assert_eq!(selected_scores.shape(), vec![2, 1]);
    assert_eq!(
        selected_scores
            .to_vec_f32()
            .expect("the selected scores should evaluate as float32"),
        vec![5.0, 7.0]
    );
}

#[test]
fn should_put_selected_values_and_copy_uint32_indices_contiguously() {
    let runtime = runtime();
    let destination = runtime
        .array_from_f32(&[0.0; 4], &[1, 4])
        .expect("the scatter destination should be valid");
    let selected_indices = runtime
        .array_from_u32(&[1, 3], &[1, 2])
        .expect("the selected indices should be valid");
    let selected_values = runtime
        .array_from_f32(&[5.0, 7.0], &[1, 2])
        .expect("the selected values should be valid");

    let scattered_values = runtime
        .put_along_axis(&destination, &selected_indices, &selected_values, -1)
        .expect("put-along-axis should build a valid graph");

    assert_eq!(
        scattered_values
            .to_vec_f32()
            .expect("the scattered values should evaluate as float32"),
        vec![0.0, 5.0, 0.0, 7.0]
    );
    assert_eq!(
        runtime
            .copy_u32_values(&selected_indices)
            .expect("selected indices should copy through contiguous storage"),
        vec![1, 3]
    );
}

#[test]
fn should_return_contiguous_top_values_along_one_axis() {
    let runtime = runtime();
    let routing_scores = runtime
        .array_from_f32(&[1.0, 5.0, 3.0, 2.0, 4.0, 0.0, 7.0, 6.0], &[2, 4])
        .expect("the routing score matrix should be valid");

    let top_scores = runtime
        .topk_axis(&routing_scores, 1, 1)
        .expect("top-k should build a valid graph");

    assert_eq!(top_scores.shape(), vec![2, 1]);
    assert_eq!(
        top_scores
            .to_vec_f32()
            .expect("the top scores should evaluate as float32"),
        vec![5.0, 7.0]
    );
}

#[test]
fn should_copy_strided_top_indices_and_select_scores() {
    let runtime = runtime();
    let routing_scores = runtime
        .array_from_f32(&[1.0, 5.0, 3.0, 2.0, 4.0, 0.0, 7.0, 6.0], &[2, 4])
        .expect("the routing score matrix should be valid");

    let partitioned_indices = runtime
        .argpartition_axis(&routing_scores, -1, 1)
        .expect("argpartition should build a valid graph");
    let top_index_column = runtime
        .slice(&partitioned_indices, &[0, 3], &[2, 4], &[1, 1])
        .expect("slicing the top index column should build a valid graph");
    let selected_scores = runtime
        .take_along_axis(&routing_scores, &top_index_column, 1)
        .expect("take-along-axis should select the top scores");

    assert_eq!(top_index_column.shape(), vec![2, 1]);
    assert_eq!(selected_scores.shape(), vec![2, 1]);
    let contiguous_top_indices = runtime
        .build_contiguous_row_major_copy(&top_index_column)
        .expect("the strided top-index slice should build a contiguous copy graph");
    contiguous_top_indices
        .evaluate()
        .expect("the contiguous top-index copy should evaluate");
    assert_eq!(
        contiguous_top_indices
            .copy_evaluated_u32_values()
            .expect("evaluated top indices should copy into Rust-owned memory"),
        vec![1, 2]
    );
    assert_eq!(
        selected_scores
            .to_vec_f32()
            .expect("the selected scores should evaluate as float32"),
        vec![5.0, 7.0]
    );
}

#[test]
fn should_apply_grouped_conv1d_on_the_runtime_stream() {
    let runtime = runtime();
    let input = runtime
        .array_from_f32(&[1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0], &[1, 4, 2])
        .expect("the conv1d input should be valid");
    let depthwise_weights = runtime
        .array_from_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2, 1])
        .expect("the grouped conv1d weights should be valid");

    let convolution = runtime
        .conv1d(&input, &depthwise_weights, 1, 0, 1, 2)
        .expect("grouped conv1d should build a valid graph");

    assert_eq!(convolution.shape(), vec![1, 3, 2]);
    assert_eq!(
        convolution
            .to_vec_f32()
            .expect("the grouped convolution should evaluate as float32"),
        vec![5.0, 110.0, 8.0, 180.0, 11.0, 250.0]
    );
}

#[test]
fn should_apply_conv3d_on_the_runtime_stream() {
    let runtime = runtime();
    let input = runtime
        .array_from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[1, 2, 2, 2, 1])
        .expect("the conv3d input should be valid");
    let convolution_weights = runtime
        .array_from_f32(&[1.0; 8], &[1, 2, 2, 2, 1])
        .expect("the conv3d weights should be valid");

    let convolution = runtime
        .conv3d(
            &input,
            &convolution_weights,
            [2, 2, 2],
            [0, 0, 0],
            [1, 1, 1],
            1,
        )
        .expect("conv3d should build a valid graph");

    assert_eq!(convolution.shape(), vec![1, 1, 1, 1, 1]);
    assert_eq!(
        convolution
            .to_vec_f32()
            .expect("the convolution should evaluate as float32"),
        vec![36.0]
    );
}
