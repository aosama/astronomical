//! The requantizer alone: the lowest tier, measured first.

use serial_test::serial;

use super::support::{fresh_output_directory, journey_runtime, source_model_directory};

#[test]
#[serial]
#[ignore = "needs ASTRONOMICAL_QUANTIZATION_SOURCE_MODEL_DIR; pending implementation of run_requantize"]
fn should_requantize_the_installed_source_at_uniform_four_bits_and_report_throughput() {
    // Work to implement:
    // 1. Inventory the source; build a Uniform 4-bit plan.
    // 2. Warm up by requantizing the first plan entry alone into a scratch
    //    directory (absorbs kernel compilation), then run the full plan with
    //    a reporter that records every progress line.
    // 3. Report tensors per second, bytes per second of source read, output
    //    size in gigabytes (2^30), and peak MLX memory; print them with
    //    `--nocapture` so the figure lands in the run log.
    // 4. Assert progress was reported at least every five seconds of wall
    //    time (the repository's batch-job rule) and that peak memory stayed
    //    below the plan's largest tensor float32 size times eight.
    // 5. Remove the output directory.
    let _ = (
        journey_runtime(),
        source_model_directory(),
        fresh_output_directory("throughput"),
    );
    todo!("run the warmup and the measured requantization as documented above")
}
