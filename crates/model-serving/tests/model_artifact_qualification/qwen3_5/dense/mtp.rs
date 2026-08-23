use std::time::Duration;

use crate::model_artifact_qualification::qwen3_5::mtp_support::run_one_layer_mtp_head_forward_qualification;

#[tokio::test]
#[ignore = "loads the complete local Qwen3.8-27B-MTPLX-4bit artifact and evaluates its dense MTP head"]
async fn should_evaluate_the_dense_qwen3_6_mtp_head_from_target_pre_normalization_hidden_states() {
    tokio::time::timeout(
        Duration::from_secs(120),
        run_one_layer_mtp_head_forward_qualification(
            super::qwen3_8_27b_mtplx_model_directory(),
            "oq4e-dense-mtp-head",
        ),
    )
    .await
    .expect("the dense oQ4e MTP head qualification should finish within 120 seconds");
}
