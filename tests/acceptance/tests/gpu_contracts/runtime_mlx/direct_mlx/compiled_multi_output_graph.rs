use astronomical_mlx_c_rust::{
    MlxCompiledMultiOutputGraph, MlxStream, array_from_vector, graph_output_array, raw,
    set_graph_output_vector,
};

use crate::common::runtime_test_support;

const COMPILE_OPERATION: &str = "compile the multi-output contract graph";

#[test]
fn should_replay_one_compiled_graph_with_two_ordered_outputs() {
    let runtime = runtime_test_support::runtime();
    let compiled_graph =
        MlxCompiledMultiOutputGraph::new(build_two_output_graph, COMPILE_OPERATION, true)
            .expect("the two-output graph should compile");

    for replay_index in 0..2 {
        let input = runtime
            .array_from_f32(&[1.0, 2.0, 4.0], &[3])
            .expect("the input should build");
        let outputs = runtime
            .apply_compiled_multi_output_graph(&compiled_graph, &[&input])
            .expect("the compiled graph should apply");
        assert_eq!(
            outputs.len(),
            2,
            "replay {replay_index} should return both ordered outputs"
        );
        let doubled_values = outputs[0]
            .to_vec_f32()
            .expect("the doubled output should read back");
        let squared_values = outputs[1]
            .to_vec_f32()
            .expect("the squared output should read back");
        runtime_test_support::assert_f32_close(&doubled_values, &[2.0, 4.0, 8.0]);
        runtime_test_support::assert_f32_close(&squared_values, &[1.0, 4.0, 16.0]);
    }
}

#[test]
fn should_reject_an_apply_that_returns_no_outputs() {
    let runtime = runtime_test_support::runtime();
    let compiled_graph =
        MlxCompiledMultiOutputGraph::new(build_no_output_graph, COMPILE_OPERATION, true)
            .expect("the empty graph should compile");
    let input = runtime
        .array_from_f32(&[1.0], &[1])
        .expect("the input should build");
    let apply_result = runtime.apply_compiled_multi_output_graph(&compiled_graph, &[&input]);
    assert!(
        apply_result.is_err(),
        "an apply that produces no outputs should fail instead of returning an empty vector"
    );
}

unsafe extern "C" fn build_two_output_graph(
    output_vector: *mut raw::mlx_vector_array,
    input_vector: raw::mlx_vector_array,
) -> i32 {
    if output_vector.is_null() || unsafe { raw::mlx_vector_array_size(input_vector) } != 1 {
        return 1;
    }
    let input = match array_from_vector(input_vector, 0) {
        Ok(input) => input,
        Err(get_status) => return get_status,
    };
    let gpu_stream = match MlxStream::default_gpu() {
        Ok(gpu_stream) => gpu_stream,
        Err(_) => return 1,
    };
    let doubled = match graph_output_array(|output_array| {
        // SAFETY: The input and stream are live, and the output is uniquely writable.
        unsafe { raw::mlx_add(output_array, input.raw(), input.raw(), gpu_stream.raw()) }
    }) {
        Ok(doubled) => doubled,
        Err(build_status) => return build_status,
    };
    let squared = match graph_output_array(|output_array| {
        // SAFETY: The input and stream are live, and the output is uniquely writable.
        unsafe { raw::mlx_multiply(output_array, input.raw(), input.raw(), gpu_stream.raw()) }
    }) {
        Ok(squared) => squared,
        Err(build_status) => return build_status,
    };
    // SAFETY: The output vector is unique and live for this single publish.
    unsafe { set_graph_output_vector(output_vector, &[&doubled, &squared]) }
}

unsafe extern "C" fn build_no_output_graph(
    _output_vector: *mut raw::mlx_vector_array,
    input_vector: raw::mlx_vector_array,
) -> i32 {
    if unsafe { raw::mlx_vector_array_size(input_vector) } != 1 {
        return 1;
    }
    0
}
