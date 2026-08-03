use crate::common::runtime_test_support::runtime;

#[test]
fn should_repeat_values_along_one_axis() {
    let runtime = runtime();
    let matrix = runtime
        .array_from_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2])
        .expect("matrix should be valid");

    let repeated_columns = runtime
        .repeat_axis(&matrix, 2, 1)
        .expect("repeat-axis should build a valid graph");

    assert_eq!(repeated_columns.shape(), vec![2, 4]);
    assert_eq!(
        repeated_columns
            .to_vec_f32()
            .expect("repeated columns should evaluate as float32"),
        vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]
    );
}

#[test]
fn should_stack_arrays_along_a_new_axis() {
    let runtime = runtime();
    let first_row = runtime
        .array_from_f32(&[1.0, 2.0], &[2])
        .expect("first row should be valid");
    let second_row = runtime
        .array_from_f32(&[3.0, 4.0], &[2])
        .expect("second row should be valid");

    let matrix = runtime
        .stack_axis(&[&first_row, &second_row], 0)
        .expect("stack-axis should build a valid graph");

    assert_eq!(matrix.shape(), vec![2, 2]);
    assert_eq!(
        matrix
            .to_vec_f32()
            .expect("stacked matrix should evaluate as float32"),
        vec![1.0, 2.0, 3.0, 4.0]
    );
}

#[test]
fn should_broadcast_an_array_to_a_static_shape() {
    let runtime = runtime();
    let row = runtime
        .array_from_f32(&[1.0, 2.0], &[2])
        .expect("row should be valid");

    let broadcast_matrix = runtime
        .broadcast_to(&row, &[2, 2])
        .expect("broadcast-to should build a valid graph");

    assert_eq!(broadcast_matrix.shape(), vec![2, 2]);
    assert_eq!(
        broadcast_matrix
            .to_vec_f32()
            .expect("broadcast matrix should evaluate as float32"),
        vec![1.0, 2.0, 1.0, 2.0]
    );
}

#[test]
fn should_squeeze_one_singleton_axis() {
    let runtime = runtime();
    let singleton_tensor = runtime
        .array_from_f32(&[1.0, 2.0], &[1, 2, 1])
        .expect("singleton tensor should be valid");

    let squeezed_matrix = runtime
        .squeeze_axis(&singleton_tensor, 0)
        .expect("squeeze-axis should build a valid graph");

    assert_eq!(squeezed_matrix.shape(), vec![2, 1]);
    assert_eq!(
        squeezed_matrix
            .to_vec_f32()
            .expect("squeezed matrix should evaluate as float32"),
        vec![1.0, 2.0]
    );
}
