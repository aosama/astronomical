use serde_json::{Value, json};

use super::drafter_configuration::{
    CandidateSelectorShape, DFlashDrafterConfiguration, DFlashDrafterConfigurationError,
    DrafterGeneration, DrafterLayerAttention, HEAD_ONLY_CANDIDATE_COUNT,
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
        DFlashDrafterConfiguration::from_config_document(&qwen38_27b_drafter_document())
            .expect("the published configuration reads");
    assert_eq!(configuration.tap_width(), 25_600);
    assert_eq!(configuration.proposal_count(), 7);
    assert_eq!(configuration.generation, DrafterGeneration::DFlash2);
    let convolution = configuration.convolution.expect("DFlash2 has convolutions");
    assert_eq!(convolution.group_count(), 320);
    assert_eq!(
        configuration.candidate_selector,
        Some(CandidateSelectorShape {
            rank: 256,
            candidate_count: 16
        })
    );
    assert_eq!(configuration.target_tap_layer_ids, vec![5, 19, 33, 47, 61]);
    assert_eq!(
        configuration.layer_attention,
        vec![DrafterLayerAttention::SlidingWindow; 5]
    );
    for layer_index in 0..5 {
        assert!(!configuration.layer_block_is_causal(layer_index));
    }
    assert_eq!(configuration.rope_theta, 10_000_000.0);
}

/// The published `ornith-ai/Ornith-1.5-35B-A3B-DFlash` configuration's
/// fields that shape the forward pass: no convolutions, no selector, one
/// full-attention layer and no declared causality.
fn ornith_35b_dflash_drafter_document() -> Value {
    json!({
        "architectures": ["DFlashDraftModel"],
        "dflash_config": {
            "block_size": 16, "mask_token_id": 248077,
            "target_layer_ids": [1, 6, 11, 16, 22, 27, 32, 37]
        },
        "head_dim": 128, "hidden_size": 2048, "intermediate_size": 6144,
        "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention",
                        "sliding_attention", "sliding_attention", "full_attention"],
        "num_attention_heads": 32, "num_hidden_layers": 6, "num_key_value_heads": 8,
        "num_target_layers": 40, "rms_norm_eps": 1e-6,
        "rope_parameters": {"rope_theta": 10000000, "rope_type": "default"},
        "sliding_window": 4096, "vocab_size": 248320
    })
}

#[test]
fn should_read_the_published_first_generation_dflash_drafter() {
    let configuration =
        DFlashDrafterConfiguration::from_config_document(&ornith_35b_dflash_drafter_document())
            .expect("the published configuration reads");
    assert_eq!(configuration.generation, DrafterGeneration::DFlash);
    assert_eq!(configuration.convolution, None);
    assert_eq!(configuration.candidate_selector, None);
    assert_eq!(
        configuration.lattice_candidate_count(),
        HEAD_ONLY_CANDIDATE_COUNT
    );
    assert_eq!(configuration.tap_width(), 16_384);
    assert_eq!(configuration.context_row_limit(), None);
}

#[test]
fn should_make_sliding_layers_causal_and_the_full_layer_two_way_when_causality_is_undeclared() {
    let configuration =
        DFlashDrafterConfiguration::from_config_document(&ornith_35b_dflash_drafter_document())
            .expect("the published configuration reads");
    for sliding_layer_index in 0..5 {
        assert!(configuration.layer_block_is_causal(sliding_layer_index));
    }
    assert!(!configuration.layer_block_is_causal(5));
}

#[test]
fn should_apply_a_declared_causality_to_every_layer() {
    let mut document = ornith_35b_dflash_drafter_document();
    document["is_causal"] = json!(true);
    let configuration = DFlashDrafterConfiguration::from_config_document(&document)
        .expect("the configuration reads");
    assert!(configuration.layer_block_is_causal(5));
}

#[test]
fn should_refuse_a_drafter_that_declares_neither_generation() {
    let mut document = qwen38_27b_drafter_document();
    document["architectures"] = json!(["Qwen3ForCausalLM"]);
    assert_eq!(
        DFlashDrafterConfiguration::from_config_document(&document),
        Err(DFlashDrafterConfigurationError::Architecture)
    );
}

#[test]
fn should_name_a_missing_selector_field_on_a_dflash2_drafter() {
    let mut document = qwen38_27b_drafter_document();
    document["dflash_config"]
        .as_object_mut()
        .expect("an object")
        .remove("selector_top_k");
    assert_eq!(
        DFlashDrafterConfiguration::from_config_document(&document),
        Err(DFlashDrafterConfigurationError::Field {
            field: "selector_top_k"
        })
    );
}

#[test]
fn should_refuse_logit_soft_capping_it_does_not_implement() {
    let mut document = qwen38_27b_drafter_document();
    document["dflash_config"]["final_logit_softcapping"] = json!(30.0);
    assert_eq!(
        DFlashDrafterConfiguration::from_config_document(&document),
        Err(DFlashDrafterConfigurationError::Unsupported {
            field: "final_logit_softcapping"
        })
    );
}

#[test]
fn should_accept_identity_values_for_the_optional_scales() {
    let mut document = qwen38_27b_drafter_document();
    document["dflash_config"]["output_multiplier"] = json!(1.0);
    document["dflash_config"]["input_embedding_scale"] = json!(1.0);
    assert!(DFlashDrafterConfiguration::from_config_document(&document).is_ok());
}

#[test]
fn should_refuse_a_tap_layer_beyond_the_target() {
    let mut document = qwen38_27b_drafter_document();
    document["dflash_config"]["target_layer_ids"] = json!([5, 64]);
    assert!(matches!(
        DFlashDrafterConfiguration::from_config_document(&document),
        Err(DFlashDrafterConfigurationError::Inconsistent { .. })
    ));
}

#[test]
fn should_refuse_a_block_with_no_proposal_slot() {
    let mut document = qwen38_27b_drafter_document();
    document["dflash_config"]["block_size"] = json!(1);
    assert!(matches!(
        DFlashDrafterConfiguration::from_config_document(&document),
        Err(DFlashDrafterConfigurationError::Inconsistent { .. })
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
        DFlashDrafterConfiguration::from_config_document(&document),
        Err(DFlashDrafterConfigurationError::Field {
            field: "sliding_window"
        })
    );
}

#[test]
fn should_refuse_an_unknown_layer_type() {
    let mut document = qwen38_27b_drafter_document();
    document["layer_types"][2] = json!("linear_attention");
    assert_eq!(
        DFlashDrafterConfiguration::from_config_document(&document),
        Err(DFlashDrafterConfigurationError::Field {
            field: "layer_types"
        })
    );
}

#[test]
fn should_limit_context_rows_to_the_window_when_every_layer_slides() {
    let configuration =
        DFlashDrafterConfiguration::from_config_document(&qwen38_27b_drafter_document())
            .expect("the published configuration reads");
    assert_eq!(configuration.context_row_limit(), Some(2_047));
}

#[test]
fn should_keep_every_context_row_when_any_layer_attends_to_all_of_it() {
    let mut configuration =
        DFlashDrafterConfiguration::from_config_document(&qwen38_27b_drafter_document())
            .expect("the published configuration reads");
    configuration.layer_attention[2] = DrafterLayerAttention::Full;
    assert_eq!(configuration.context_row_limit(), None);
}

#[test]
fn should_read_a_drafter_naming_both_architectures_as_dflash2() {
    let mut document = qwen38_27b_drafter_document();
    document["architectures"] = json!(["DFlashDraftModel", "DFlash2DraftModel"]);
    let configuration = DFlashDrafterConfiguration::from_config_document(&document)
        .expect("the configuration reads");
    assert_eq!(configuration.generation, DrafterGeneration::DFlash2);
}

#[test]
fn should_ignore_second_generation_fields_on_a_dflash_drafter() {
    let mut document = ornith_35b_dflash_drafter_document();
    document["dflash_config"]["conv_group_size"] = json!(16);
    document["dflash_config"]["selector_top_k"] = json!(16);
    let configuration = DFlashDrafterConfiguration::from_config_document(&document)
        .expect("the configuration reads");
    assert_eq!(configuration.convolution, None);
    assert_eq!(configuration.candidate_selector, None);
}

#[test]
fn should_refuse_a_convolution_with_no_group_width_or_no_taps() {
    for (field, value) in [("conv_group_size", 0), ("conv_kernel_size", 0)] {
        let mut document = qwen38_27b_drafter_document();
        document["dflash_config"][field] = json!(value);
        assert!(
            matches!(
                DFlashDrafterConfiguration::from_config_document(&document),
                Err(DFlashDrafterConfigurationError::Inconsistent { .. })
            ),
            "{field} of {value} must be refused"
        );
    }
}
