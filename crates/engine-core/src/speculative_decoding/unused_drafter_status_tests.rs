use ipc_protocol::DraftModelStatus;

use super::{STREAMING_REASON, streaming_load_status};
use crate::speculative_decoding::drafter_discovery::DrafterPairing;

/// The Ornith team's DFlash drafter for Ornith 35B-A3B, as published.
const ORNITH_DFLASH_DRAFTER_CONFIGURATION: &str = r#"{"architectures":["DFlashDraftModel"],"attention_bias":false,"attention_dropout":0.0,"dflash_config":{"block_size":16,"mask_token_id":248077,"target_layer_ids":[1,6,11,16,22,27,32,37]},"head_dim":128,"hidden_act":"silu","hidden_size":2048,"intermediate_size":6144,"layer_types":["sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention"],"max_position_embeddings":262144,"model_type":"qwen3","num_attention_heads":32,"num_hidden_layers":6,"num_key_value_heads":8,"num_target_layers":40,"rms_norm_eps":1e-06,"rope_parameters":{"rope_theta":10000000,"rope_type":"default"},"sliding_window":4096,"tie_word_embeddings":false,"vocab_size":248320}"#;

/// The shape of Ornith 35B-A3B, which the Ornith drafter names.
const ORNITH_35B_PAIRING: DrafterPairing = DrafterPairing {
    target_layer_count: 40,
    target_hidden_size: 2048,
    target_vocabulary_size: 248_320,
};

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
        streaming_load_status(&model_directory, &ORNITH_35B_PAIRING),
        DraftModelStatus::Unavailable {
            reason: STREAMING_REASON.to_owned()
        }
    );
    remove_test_library(&model_directory);
}

#[test]
fn should_report_no_drafter_for_a_streamed_model_without_one() {
    let model_directory = fresh_library_model("none");
    assert_eq!(
        streaming_load_status(&model_directory, &ORNITH_35B_PAIRING),
        DraftModelStatus::None
    );
    remove_test_library(&model_directory);
}
