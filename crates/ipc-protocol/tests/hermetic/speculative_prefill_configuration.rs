use std::path::PathBuf;

use astronomical_ipc_protocol::WorkerSpeculativePrefillConfiguration;

#[test]
fn should_enable_speculative_prefill_for_its_configured_target_model() {
    let speculative_prefill_configuration = speculative_prefill_configuration_for_tests();

    let target_model_speculative_prefill_configuration =
        speculative_prefill_configuration.for_loaded_model("astronomical/target-model");

    assert_eq!(
        target_model_speculative_prefill_configuration,
        speculative_prefill_configuration
    );
}

#[test]
fn should_disable_speculative_prefill_for_an_unconfigured_model() {
    let speculative_prefill_configuration = speculative_prefill_configuration_for_tests();

    let ordinary_model_speculative_prefill_configuration =
        speculative_prefill_configuration.for_loaded_model("astronomical/ordinary-model");

    assert!(!ordinary_model_speculative_prefill_configuration.enabled);
}

fn speculative_prefill_configuration_for_tests() -> WorkerSpeculativePrefillConfiguration {
    WorkerSpeculativePrefillConfiguration {
        enabled: true,
        target_model_id: Some("astronomical/target-model".to_owned()),
        draft_model_id: Some("astronomical/draft-model".to_owned()),
        draft_model_directory: Some(PathBuf::from("/tmp/fictional-draft-model")),
        minimum_prompt_tokens: 8_192,
        keep_percentage: 20,
        selection_chunck_token_count: 32,
        mandatory_trailing_token_count: 512,
        lookahead_token_count: 8,
        importance_pooling_kernel_token_count: 13,
    }
}
