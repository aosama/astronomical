use serde_json::{Value, json};

use super::drafter_configuration::{
    DFlash2DrafterConfiguration, DFlash2DrafterConfigurationError, DrafterLayerAttention,
};

/// The published `z-lab/Qwen3.8-27B-DFlash2` configuration's fields that
/// shape the forward pass.
fn qwen38_27b_drafter_document() -> Value {
    json!({
        "architectures": ["DFlash2DraftModel"],
        "dflash_config": {
            "block_size": 8, "conv_group_size": 16, "conv_kernel_size": 2,
            "mask_token_id": 248070, "selector_rank": 256, "selector_top_k": 16,
            "target_layer_ids": [5, 19, 33, 47, 61]
        },
        "head_dim": 128, "hidden_size": 5120, "intermediate_size": 17408,
        "is_causal": false,
        "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention",
                        "sliding_attention", "sliding_attention"],
        "num_attention_heads": 32, "num_hidden_layers": 5, "num_key_value_heads": 8,
        "num_target_layers": 64, "rms_norm_eps": 1e-6,
        "rope_parameters": {"rope_theta": 10000000, "rope_type": "default"},
        "sliding_window": 2048, "vocab_size": 248320
    })
}

#[test]
fn should_read_the_published_27b_drafter_configuration() {
    let configuration =
        DFlash2DrafterConfiguration::from_config_document(&qwen38_27b_drafter_document())
            .expect("the published configuration reads");
    assert_eq!(configuration.tap_width(), 25_600);
    assert_eq!(configuration.proposal_count(), 7);
    assert_eq!(configuration.convolution_group_count(), 320);
    assert_eq!(configuration.target_tap_layer_ids, vec![5, 19, 33, 47, 61]);
    assert_eq!(
        configuration.layer_attention,
        vec![DrafterLayerAttention::SlidingWindow; 5]
    );
    assert!(!configuration.block_is_causal);
    assert_eq!(configuration.rope_theta, 10_000_000.0);
}

#[test]
fn should_refuse_a_first_generation_dflash_drafter() {
    let mut document = qwen38_27b_drafter_document();
    document["architectures"] = json!(["DFlashDraftModel"]);
    assert_eq!(
        DFlash2DrafterConfiguration::from_config_document(&document),
        Err(DFlash2DrafterConfigurationError::Architecture)
    );
}

#[test]
fn should_refuse_logit_soft_capping_it_does_not_implement() {
    let mut document = qwen38_27b_drafter_document();
    document["dflash_config"]["final_logit_softcapping"] = json!(30.0);
    assert_eq!(
        DFlash2DrafterConfiguration::from_config_document(&document),
        Err(DFlash2DrafterConfigurationError::Unsupported {
            field: "final_logit_softcapping"
        })
    );
}

#[test]
fn should_accept_identity_values_for_the_optional_scales() {
    let mut document = qwen38_27b_drafter_document();
    document["dflash_config"]["output_multiplier"] = json!(1.0);
    document["dflash_config"]["input_embedding_scale"] = json!(1.0);
    assert!(DFlash2DrafterConfiguration::from_config_document(&document).is_ok());
}

#[test]
fn should_refuse_a_tap_layer_beyond_the_target() {
    let mut document = qwen38_27b_drafter_document();
    document["dflash_config"]["target_layer_ids"] = json!([5, 64]);
    assert!(matches!(
        DFlash2DrafterConfiguration::from_config_document(&document),
        Err(DFlash2DrafterConfigurationError::Inconsistent { .. })
    ));
}

#[test]
fn should_refuse_a_block_with_no_proposal_slot() {
    let mut document = qwen38_27b_drafter_document();
    document["dflash_config"]["block_size"] = json!(1);
    assert!(matches!(
        DFlash2DrafterConfiguration::from_config_document(&document),
        Err(DFlash2DrafterConfigurationError::Inconsistent { .. })
    ));
}

#[test]
fn should_name_a_missing_field() {
    let mut document = qwen38_27b_drafter_document();
    document
        .as_object_mut()
        .expect("an object")
        .remove("sliding_window");
    assert_eq!(
        DFlash2DrafterConfiguration::from_config_document(&document),
        Err(DFlash2DrafterConfigurationError::Field {
            field: "sliding_window"
        })
    );
}

#[test]
fn should_refuse_an_unknown_layer_type() {
    let mut document = qwen38_27b_drafter_document();
    document["layer_types"][2] = json!("linear_attention");
    assert_eq!(
        DFlash2DrafterConfiguration::from_config_document(&document),
        Err(DFlash2DrafterConfigurationError::Field {
            field: "layer_types"
        })
    );
}

#[test]
fn should_limit_context_rows_to_the_window_when_every_layer_slides() {
    let configuration =
        DFlash2DrafterConfiguration::from_config_document(&qwen38_27b_drafter_document())
            .expect("the published configuration reads");
    assert_eq!(configuration.context_row_limit(), Some(2_047));
}

#[test]
fn should_keep_every_context_row_when_any_layer_attends_to_all_of_it() {
    let mut configuration =
        DFlash2DrafterConfiguration::from_config_document(&qwen38_27b_drafter_document())
            .expect("the published configuration reads");
    configuration.layer_attention[2] = DrafterLayerAttention::Full;
    assert_eq!(configuration.context_row_limit(), None);
}
