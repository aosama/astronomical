//! `static` method end to end, then served by Astronomical's own engine.

use serial_test::serial;

use super::support::{
    PREFILL_CHUNK_TOKENS, ROMEO_AND_JULIET_SOURCE, fresh_output_directory, journey_runtime,
    source_model_directory,
};

#[test]
#[serial]
#[ignore = "needs ASTRONOMICAL_QUANTIZATION_SOURCE_MODEL_DIR; pending implementation of run_quantization_pipeline"]
fn should_convert_with_structural_rules_and_generate_a_romeo_and_juliet_continuation() {
    // Work to implement:
    // 1. Build a QuantizationPipelineRequest with method StaticStructural,
    //    candidate bits [4, 8], target 5.0, group 64, into a fresh output.
    // 2. Run the pipeline (no probe model is loaded on the static path).
    // 3. Validate the artifact through model-serving's Qwen3.5 artifact
    //    validator (the same path the daemon uses), confirming the
    //    quantization config and the metadata agree.
    // 4. Load the artifact as a resident model, prefill roughly 500 tokens of
    //    Romeo and Juliet in 2,048-token chunks, decode 50 tokens greedily,
    //    and assert the continuation is non-empty text that decodes without
    //    an error and contains no replacement characters.
    // 5. Remove the output directory.
    let _ = (
        journey_runtime(),
        source_model_directory(),
        fresh_output_directory("static"),
        PREFILL_CHUNK_TOKENS,
        ROMEO_AND_JULIET_SOURCE.len(),
    );
    todo!("convert, validate, load, and generate as documented above")
}
