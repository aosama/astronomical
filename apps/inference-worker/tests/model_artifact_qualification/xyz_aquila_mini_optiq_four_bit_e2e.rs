use tokio::time::timeout;

use super::model_artifact_rest_qualification::{
    E2E_TIMEOUT, image_chat_request_body_for_model, run_model_artifact_request_e2e_for_model,
    text_chat_request_body_for_model,
};

const XYZ_AQUILA_MINI_OPTIQ_FOUR_BIT_MODEL_ID: &str = "XYZ-Aquila-mini-OptiQ-4bit";

#[tokio::test(flavor = "multi_thread")]
#[ignore = "loads configured XYZ-Aquila-mini-OptiQ-4bit through the production REST surface"]
async fn should_stream_xyz_aquila_mini_optiq_four_bit_text_output_through_the_openai_endpoint() {
    timeout(
        E2E_TIMEOUT,
        run_model_artifact_request_e2e_for_model(
            XYZ_AQUILA_MINI_OPTIQ_FOUR_BIT_MODEL_ID,
            crate::common::configured_model_artifact_directory_by_id(
                XYZ_AQUILA_MINI_OPTIQ_FOUR_BIT_MODEL_ID,
            ),
            "text",
            text_chat_request_body_for_model(XYZ_AQUILA_MINI_OPTIQ_FOUR_BIT_MODEL_ID),
        ),
    )
    .await
    .expect("the XYZ-Aquila-mini text E2E test must finish within 115 seconds");
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "loads configured XYZ-Aquila-mini-OptiQ-4bit with one image through the production REST surface"]
async fn should_stream_xyz_aquila_mini_optiq_four_bit_image_output_through_the_openai_endpoint() {
    timeout(
        E2E_TIMEOUT,
        run_model_artifact_request_e2e_for_model(
            XYZ_AQUILA_MINI_OPTIQ_FOUR_BIT_MODEL_ID,
            crate::common::configured_model_artifact_directory_by_id(
                XYZ_AQUILA_MINI_OPTIQ_FOUR_BIT_MODEL_ID,
            ),
            "image",
            image_chat_request_body_for_model(XYZ_AQUILA_MINI_OPTIQ_FOUR_BIT_MODEL_ID),
        ),
    )
    .await
    .expect("the XYZ-Aquila-mini image E2E test must finish within 115 seconds");
}
