//! The exact KL sweep through the model-serving probe on a real base.

use serial_test::serial;

use super::support::{
    PREFILL_CHUNK_TOKENS, ROMEO_AND_JULIET_SOURCE, fresh_output_directory, journey_runtime,
    source_model_directory,
};

#[test]
#[serial]
#[ignore = "needs ASTRONOMICAL_QUANTIZATION_SOURCE_MODEL_DIR; pending implementation of the Qwen3.5 probe"]
fn should_measure_sensitivity_with_the_bfloat16_reference_and_resume_from_a_checkpoint() {
    // Work to implement:
    // 1. Load the Qwen3.5 probe model from the source (bf16 reference) with
    //    the prefill chunk fixed at 2,048 tokens.
    // 2. Build two calibration sequences of 512 tokens from Romeo and Juliet
    //    through the source tokenizer (pre-tokenized mix).
    // 3. Run the sweep with candidate bits [4, 8] and a checkpoint path,
    //    with a reporter that counts progress lines; abort the driver after
    //    three modules by returning a probe failure from a wrapping probe,
    //    then run again and assert the first three modules were not
    //    re-probed (the wrapping probe counts simulate calls per module).
    // 4. Assert every result has finite KL at both widths, that KL at 8 bits
    //    is at most KL at 4 bits for at least 95% of modules (the signal is
    //    monotone in width on a bf16 reference), and that the measured
    //    majority guard passed.
    // 5. Allocate at 4.5 and convert; validate the artifact as in the static
    //    journey. Remove the output directory.
    let _ = (
        journey_runtime(),
        source_model_directory(),
        fresh_output_directory("exact"),
        PREFILL_CHUNK_TOKENS,
        ROMEO_AND_JULIET_SOURCE.len(),
    );
    todo!("sweep, interrupt, resume, allocate, convert as documented above")
}
