use astronomical_model_serving::{
    LagunaAttentionKind, LagunaFeedForwardDescriptor, LagunaTargetNormalizer,
    apply_router_logit_softcap, select_laguna_router_experts,
};

use super::support::{LagunaQualificationSize, qualification_config_value};

#[test]
fn should_use_correction_bias_only_for_selection_and_gather_original_scores() {
    // Expert 0 wins the ranking only because of bias; gathered weight stays 0.5.
    let router_logits = [0.0_f32, 2.0, -2.0, 0.5];
    let correction_bias = [2.0_f32, 0.0, 0.0, 0.0];
    let selection =
        select_laguna_router_experts(&router_logits, 1, 4, 2, Some(&correction_bias), 0.0, false)
            .expect("a valid unique ranking should select");
    assert_eq!(selection.expert_indices(), &[0, 1]);
    let expected_first = sigmoid(0.0);
    let expected_second = sigmoid(2.0);
    assert!((selection.original_scores()[0] - expected_first).abs() < 1e-6);
    assert!((selection.original_scores()[1] - expected_second).abs() < 1e-6);
}

#[test]
fn should_break_equal_selection_scores_toward_the_lower_expert_id() {
    let router_logits = [0.0_f32, 0.0, -4.0];
    let selection = select_laguna_router_experts(&router_logits, 1, 3, 1, None, 0.0, false)
        .expect("a tied top score should still select one expert");
    assert_eq!(selection.expert_indices(), &[0]);
}

#[test]
fn should_apply_positive_softcap_before_sigmoid_and_skip_a_disabled_cap() {
    let uncapped = apply_router_logit_softcap(30.0, 0.0);
    let capped = apply_router_logit_softcap(30.0, 2.0);
    assert_eq!(uncapped, 30.0);
    assert!((capped - 2.0 * (15.0_f32).tanh()).abs() < 1e-6);

    let uncapped_selection = select_laguna_router_experts(&[30.0, 0.0], 1, 2, 1, None, 0.0, false)
        .expect("disabled softcap should select");
    let capped_selection = select_laguna_router_experts(&[30.0, 0.0], 1, 2, 1, None, 2.0, false)
        .expect("positive softcap should select");
    assert!(uncapped_selection.original_scores()[0] > capped_selection.original_scores()[0]);
}

#[test]
fn should_optionally_normalize_selected_sigmoid_scores() {
    let router_logits = [1.0_f32, 0.0, -1.0];
    let raw = select_laguna_router_experts(&router_logits, 1, 3, 2, None, 0.0, false)
        .expect("raw scores should select");
    let normalized = select_laguna_router_experts(&router_logits, 1, 3, 2, None, 0.0, true)
        .expect("normalized scores should select");
    let raw_sum: f32 = raw.original_scores().iter().sum();
    let normalized_sum: f32 = normalized.original_scores().iter().sum();
    assert!((normalized_sum - 1.0).abs() < 1e-6);
    assert!((normalized.original_scores()[0] - raw.original_scores()[0] / raw_sum).abs() < 1e-6);
}

#[test]
fn should_reject_invalid_router_geometry() {
    assert!(select_laguna_router_experts(&[0.0], 1, 2, 1, None, 0.0, false).is_err());
    assert!(select_laguna_router_experts(&[0.0, 0.0], 1, 2, 3, None, 0.0, false).is_err());
    assert!(select_laguna_router_experts(&[0.0, 0.0], 1, 2, 1, Some(&[0.0]), 0.0, false).is_err());
}

#[test]
fn should_treat_xs_and_s_affine_storage_profiles_as_named_rows() {
    use astronomical_model_serving::LagunaStorageDescriptor;
    for (row_name, fixture_size, expected_bits, expected_group_size) in [
        ("xs", LagunaQualificationSize::ExtraSmall, 2_u32, 64_u32),
        ("s", LagunaQualificationSize::Small, 2, 128),
    ] {
        let contract = LagunaTargetNormalizer::normalize(
            &serde_json::to_vec(&qualification_config_value(fixture_size))
                .expect("qualification config should serialize"),
        )
        .unwrap_or_else(|_| panic!("{row_name} should normalize"));
        let LagunaStorageDescriptor::DirectAffine(affine) = contract.storage() else {
            panic!("{row_name} should declare direct affine storage");
        };
        assert_eq!(affine.default_profile().bits(), expected_bits, "{row_name}");
        assert_eq!(
            affine.default_profile().group_size(),
            expected_group_size,
            "{row_name}"
        );
        assert_eq!(affine.profile_for_module("lm_head").bits(), 8, "{row_name}");
    }
}

#[test]
fn should_treat_xs_and_s_router_geometry_as_named_rows() {
    for (row_name, fixture_size, expected_top_k, expected_softcap) in [
        ("xs", LagunaQualificationSize::ExtraSmall, 8_u32, 0.0_f64),
        ("s", LagunaQualificationSize::Small, 10, 0.0),
    ] {
        let contract = LagunaTargetNormalizer::normalize(
            &serde_json::to_vec(&qualification_config_value(fixture_size))
                .expect("qualification config should serialize"),
        )
        .unwrap_or_else(|_| panic!("{row_name} should normalize"));
        assert_eq!(
            contract.model().router_logit_softcap(),
            expected_softcap,
            "{row_name}"
        );
        let sparse_layer = contract
            .layers()
            .iter()
            .find(|layer| {
                matches!(layer.feed_forward(), LagunaFeedForwardDescriptor::Moe(_))
                    && layer.attention().kind() == LagunaAttentionKind::Sliding
            })
            .unwrap_or_else(|| panic!("{row_name} should contain a sparse sliding layer"));
        let LagunaFeedForwardDescriptor::Moe(moe) = sparse_layer.feed_forward() else {
            panic!("{row_name} sparse layer should expose a Mixture-of-Experts descriptor");
        };
        assert_eq!(moe.expert_count(), 256, "{row_name}");
        assert_eq!(moe.experts_per_token(), expected_top_k, "{row_name}");
        assert!(moe.normalizes_top_k_probabilities(), "{row_name}");
        assert!(
            (moe.routed_scaling_factor() - 2.5).abs() < 1e-12,
            "{row_name}"
        );
        assert!(!moe.applies_router_weight_on_input(), "{row_name}");
        assert_eq!(
            moe.shared_expert_intermediate_size(),
            moe.expert_intermediate_size()
        );
    }
}

fn sigmoid(logit: f32) -> f32 {
    1.0 / (1.0 + (-logit).exp())
}
