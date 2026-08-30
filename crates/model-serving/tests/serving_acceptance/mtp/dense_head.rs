use std::time::Duration;

use crate::serving_acceptance::support::mtp_support::run_one_layer_mtp_head_forward_acceptance;

#[tokio::test]
#[ignore = "loads the dense MTP e2e fixture and evaluates its dense MTP head"]
async fn should_evaluate_the_dense_mtp_head_from_target_pre_normalization_hidden_states() {
    tokio::time::timeout(
        Duration::from_secs(120),
        run_one_layer_mtp_head_forward_acceptance(
            crate::serving_acceptance::support::dense_mtp_model_directory(),
            "dense-mtp-head",
        ),
    )
    .await
    .expect("the dense MTP head acceptance should finish within 120 seconds");
}
