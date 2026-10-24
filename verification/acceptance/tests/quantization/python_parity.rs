//! Tensor-by-tensor parity with an artifact the Python tool built from the
//! same source with the same flags.

use serial_test::serial;

use super::support::{
    fresh_output_directory, journey_runtime, python_reference_directory, source_model_directory,
};

#[test]
#[serial]
#[ignore = "needs model_directory and reference_artifact_path in the explicit test-run configuration; pending implementation"]
fn should_match_the_python_artifact_tensor_by_tensor_under_the_same_static_allocation() {
    // Work to implement:
    // 1. Read the Python artifact's optiq/metadata.json; convert the source
    //    with the Rust pipeline using the same method (static), ladder
    //    [2, 4, 8], target 4.5, group 64, and the range search disabled.
    // 2. Assert the two `per_layer` maps are identical (same allocator on
    //    the same structural scores).
    // 3. For every quantized module: load both artifacts' packed weight,
    //    scales, and biases through MLX; at 4 and 8 bits assert the packed
    //    words are identical and scales and biases equal within bf16
    //    rounding; at 2 bits assert the Rust dequantized tensor's squared
    //    error against the Python dequantized tensor is below 1e-6 relative.
    // 4. Assert both artifacts' config.json quantization documents are
    //    equal after sorting keys.
    // 5. Remove the output directory.
    let _ = (
        journey_runtime(),
        source_model_directory(),
        python_reference_directory(),
        fresh_output_directory("parity"),
    );
    todo!("convert and compare as documented above")
}

#[test]
#[serial]
#[ignore = "needs reference_artifact_path with optiq/sensitivity.json in the explicit test-run configuration; pending implementation"]
fn should_reproduce_the_python_bit_map_from_the_python_sensitivity_document() {
    // Work to implement: read the reference artifact's optiq/sensitivity.json
    // and optiq/metadata.json; allocate with the Rust allocator at the
    // metadata's target_bpw and candidate ladder; assert the resulting bit map
    // equals the metadata's per_layer for every measured module. This is the
    // allocator parity test on a real sweep rather than the synthetic table.
    let _ = python_reference_directory();
    todo!("allocate from the Python sweep and compare bit maps")
}
