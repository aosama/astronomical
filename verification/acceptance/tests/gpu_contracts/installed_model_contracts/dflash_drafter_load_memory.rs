//! What loading and hosting a real DFlash drafter holds in memory: the
//! active bytes after its checkpoint is mapped, after its linears are
//! requantized for serving, and after the checkpoint is let go, with the peak
//! of the whole load. The drafter's load check (`drafter_load_byte_count`)
//! must ask for the room the load really takes, no less and no more: asking
//! for the bf16 checkpoint whole refused a drafter that fits once served, on
//! the dense 27B at an 18 GB limit (#2578).
//!
//! The configuration names the target model's directory; its drafter is the
//! drafter folder inside it (`drafter`, or `dflash` in a community pack).

use std::fs::File;

use model_serving::speculative_decoding::dflash_drafter::{
    DFlashDrafterConfiguration, bind_drafter_weights, drafter_load_peak_bytes,
    read_drafter_checkpoint_format,
};
use runtime_integration::{MlxMemoryLimits, MlxRuntime};
use serial_test::file_serial;
use test_harness::test_run::TestRunContext;

const BYTES_PER_GIGABYTE: f64 = 1_073_741_824.0;

fn gigabytes(byte_count: usize) -> f64 {
    byte_count as f64 / BYTES_PER_GIGABYTE
}

fn active_gigabytes(runtime: &MlxRuntime) -> f64 {
    gigabytes(
        runtime
            .memory_snapshot()
            .expect("memory is readable")
            .active_memory_bytes(),
    )
}

#[test]
#[ignore = "loads a real drafter on the GPU; run with --configuration naming a model with a drafter"]
#[file_serial]
fn should_report_what_holding_the_checkpoint_through_the_load_costs() {
    let drafter_directory = bundled_drafter_directory();
    let runtime = MlxRuntime::initialize(
        MlxMemoryLimits::new(16 * 1024 * 1024 * 1024, 256 * 1024 * 1024)
            .expect("memory limits are valid"),
    )
    .expect("the MLX runtime initializes");
    let config_document: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(drafter_directory.join("config.json"))
            .expect("the drafter's config reads"),
    )
    .expect("the drafter's config parses");
    let configuration = DFlashDrafterConfiguration::from_config_document(&config_document)
        .expect("the drafter's configuration is accepted");
    let checkpoint_file_bytes = std::fs::metadata(drafter_directory.join("model.safetensors"))
        .expect("the drafter's weights exist")
        .len();
    runtime.reset_peak_memory().expect("the peak resets");
    let before_gigabytes = active_gigabytes(&runtime);
    let checkpoint = runtime
        .load_safetensors(
            File::open(drafter_directory.join("model.safetensors")).expect("the weights open"),
            None,
        )
        .expect("the checkpoint maps");
    let mut drafter_weights = bind_drafter_weights(&runtime, &checkpoint, &configuration)
        .expect("every drafter tensor binds");
    let after_bind_gigabytes = active_gigabytes(&runtime);
    drafter_weights
        .quantize_linears(&runtime, true)
        .expect("the linears requantize");
    let after_quantize_gigabytes = active_gigabytes(&runtime);
    drop(checkpoint);
    let after_checkpoint_dropped_gigabytes = active_gigabytes(&runtime);
    let peak_gigabytes = gigabytes(
        runtime
            .memory_snapshot()
            .expect("memory is readable")
            .peak_memory_bytes(),
    );
    println!(
        "[dflash-drafter-load-memory] checkpoint_file_gb={:.2} before_gb={before_gigabytes:.2} after_bind_gb={after_bind_gigabytes:.2} after_quantize_gb={after_quantize_gigabytes:.2} after_checkpoint_dropped_gb={after_checkpoint_dropped_gigabytes:.2} peak_gb={peak_gigabytes:.2}",
        checkpoint_file_bytes as f64 / BYTES_PER_GIGABYTE
    );
    drop(drafter_weights);
}

/// The order the engine loads in: the checkpoint let go right after its
/// tensors are bound, so each full-precision linear is freed once it has been
/// requantized. Its peak must stay within what the load check asks for.
#[test]
#[ignore = "loads a real drafter on the GPU; run with --configuration naming a model with a drafter"]
#[file_serial]
fn should_peak_within_the_room_the_load_check_asks_for() {
    let drafter_directory = bundled_drafter_directory();
    let runtime = MlxRuntime::initialize(
        MlxMemoryLimits::new(16 * 1024 * 1024 * 1024, 256 * 1024 * 1024)
            .expect("memory limits are valid"),
    )
    .expect("the MLX runtime initializes");
    let config_document: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(drafter_directory.join("config.json"))
            .expect("the drafter's config reads"),
    )
    .expect("the drafter's config parses");
    let configuration = DFlashDrafterConfiguration::from_config_document(&config_document)
        .expect("the drafter's configuration is accepted");
    runtime.reset_peak_memory().expect("the peak resets");
    let checkpoint = runtime
        .load_safetensors(
            File::open(drafter_directory.join("model.safetensors")).expect("the weights open"),
            None,
        )
        .expect("the checkpoint maps");
    let mut drafter_weights = bind_drafter_weights(&runtime, &checkpoint, &configuration)
        .expect("every drafter tensor binds");
    drop(checkpoint);
    drafter_weights
        .quantize_linears(&runtime, true)
        .expect("the linears requantize");
    let after_quantize_gigabytes = active_gigabytes(&runtime);
    let peak_gigabytes = gigabytes(
        runtime
            .memory_snapshot()
            .expect("memory is readable")
            .peak_memory_bytes(),
    );
    let checkpoint_file_bytes = std::fs::metadata(drafter_directory.join("model.safetensors"))
        .expect("the drafter's weights exist")
        .len();
    let allowed_peak_gigabytes = gigabytes(
        usize::try_from(drafter_load_peak_bytes(
            checkpoint_file_bytes,
            read_drafter_checkpoint_format(&drafter_directory.join("model.safetensors"))
                .expect("the checkpoint's header reads"),
            &configuration,
        ))
        .unwrap_or(usize::MAX),
    );
    println!(
        "[dflash-drafter-load-memory] order=let-go-before-requantizing after_quantize_gb={after_quantize_gigabytes:.2} peak_gb={peak_gigabytes:.2} allowed_peak_gb={allowed_peak_gigabytes:.2}"
    );
    assert!(
        peak_gigabytes <= allowed_peak_gigabytes,
        "the load peaked at {peak_gigabytes:.2} GB, above the {allowed_peak_gigabytes:.2} GB the load check asks for"
    );
    drop(drafter_weights);
}

/// The configured model's own drafter folder: `drafter` in Astronomical's
/// packages, `dflash` in community packs.
pub(super) fn bundled_drafter_directory() -> std::path::PathBuf {
    let model_directory = TestRunContext::current().required_model_directory();
    ipc_protocol::BUNDLED_DRAFTER_FOLDER_NAMES
        .iter()
        .map(|bundled_folder_name| model_directory.join(bundled_folder_name))
        .find(|candidate_directory| candidate_directory.join("config.json").is_file())
        .expect("the configured model carries a drafter folder")
}
