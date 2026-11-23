use ipc_protocol::DraftModelStatus;

use super::{KERNELS_UNAVAILABLE_REASON, failed_load_status, load_outcome_status};
use crate::qwen3_5_resident::inference_execution::speculative_decode_lifecycle::engine_draft_runtime::{
    Qwen3_5DraftRuntimeLoadOutcome, Qwen3_5SpeculativeDraftLoadingError,
};

#[test]
fn should_report_no_drafter_when_none_pairs() {
    assert_eq!(
        load_outcome_status(&Qwen3_5DraftRuntimeLoadOutcome::NoPairedDrafter),
        DraftModelStatus::None
    );
}

#[test]
fn should_explain_a_gpu_without_the_drafting_kernels() {
    assert_eq!(
        load_outcome_status(&Qwen3_5DraftRuntimeLoadOutcome::KernelsUnavailable),
        DraftModelStatus::Unavailable {
            reason: KERNELS_UNAVAILABLE_REASON.to_owned()
        }
    );
}

#[test]
fn should_name_the_memory_limit_when_there_is_no_room_for_the_drafter() {
    let headroom_error = Qwen3_5SpeculativeDraftLoadingError::InsufficientHeadroom {
        free_bytes: 0,
        required_bytes: 1,
    };
    let DraftModelStatus::Unavailable { reason } = failed_load_status(&headroom_error) else {
        panic!("a drafter that cannot fit is unavailable");
    };
    assert!(reason.contains("memory limit"));
}

#[test]
fn should_never_put_a_local_path_in_a_failure_reason() {
    let unreadable_error = Qwen3_5SpeculativeDraftLoadingError::WeightsUnreadable {
        source: std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "/private/drafter/model.safetensors",
        ),
    };
    let DraftModelStatus::Unavailable { reason } = failed_load_status(&unreadable_error) else {
        panic!("a drafter that cannot be read is unavailable");
    };
    assert!(!reason.contains('/'));
}

/// The Ornith team's DFlash drafter for Ornith 35B-A3B, as published.
const ORNITH_DFLASH_DRAFTER_CONFIGURATION: &str = r#"{"architectures":["DFlashDraftModel"],"attention_bias":false,"attention_dropout":0.0,"dflash_config":{"block_size":16,"mask_token_id":248077,"target_layer_ids":[1,6,11,16,22,27,32,37]},"head_dim":128,"hidden_act":"silu","hidden_size":2048,"intermediate_size":6144,"layer_types":["sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention"],"max_position_embeddings":262144,"model_type":"qwen3","num_attention_heads":32,"num_hidden_layers":6,"num_key_value_heads":8,"num_target_layers":40,"rms_norm_eps":1e-06,"rope_parameters":{"rope_theta":10000000,"rope_type":"default"},"sliding_window":4096,"tie_word_embeddings":false,"vocab_size":248320}"#;

fn ornith_35b_configuration() -> crate::Qwen3_5Config {
    crate::Qwen3_5Config::from_json_bytes(
        &crate::qwen3_5_core::frozen_config_test_support::frozen_ornith_1_0_config_bytes(),
    )
    .expect("the frozen Ornith configuration parses")
}

fn fresh_library_model(test_name: &str) -> std::path::PathBuf {
    let model_directory = std::env::temp_dir()
        .join(format!(
            "astronomical-streaming-drafter-{test_name}-{}",
            std::process::id()
        ))
        .join("example-org")
        .join("example-ornith");
    let _ = std::fs::remove_dir_all(
        model_directory
            .parent()
            .and_then(std::path::Path::parent)
            .expect("a library root"),
    );
    std::fs::create_dir_all(&model_directory).expect("the test library is writable");
    model_directory
}

/// Removes the library `fresh_library_model` made around `model_directory`.
fn remove_test_library(model_directory: &std::path::Path) {
    if let Some(library_root) = model_directory.parent().and_then(std::path::Path::parent) {
        let _ = std::fs::remove_dir_all(library_root);
    }
}

#[test]
fn should_tell_a_streamed_model_its_installed_drafter_is_unused() {
    let model_directory = fresh_library_model("bundled");
    std::fs::create_dir_all(model_directory.join("drafter")).expect("the test library is writable");
    std::fs::write(
        model_directory.join("drafter/config.json"),
        ORNITH_DFLASH_DRAFTER_CONFIGURATION,
    )
    .expect("the test library is writable");
    assert_eq!(
        super::streaming_load_status(&model_directory, &ornith_35b_configuration()),
        DraftModelStatus::Unavailable {
            reason: super::STREAMING_REASON.to_owned()
        }
    );
    remove_test_library(&model_directory);
}

#[test]
fn should_report_no_drafter_for_a_streamed_model_without_one() {
    let model_directory = fresh_library_model("none");
    assert_eq!(
        super::streaming_load_status(&model_directory, &ornith_35b_configuration()),
        DraftModelStatus::None
    );
    remove_test_library(&model_directory);
}
