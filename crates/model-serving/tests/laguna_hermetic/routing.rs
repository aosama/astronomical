use astronomical_model_serving::{apply_router_logit_softcap, select_laguna_router_experts};

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

fn sigmoid(logit: f32) -> f32 {
    1.0 / (1.0 + (-logit).exp())
}
