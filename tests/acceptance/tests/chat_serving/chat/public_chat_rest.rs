use tokio::time::timeout;

use super::openai_rest::{
    E2E_TIMEOUT, run_deployed_rest_surface_litmus, run_serving_chat_request, text_chat_request_body,
};

#[tokio::test(flavor = "multi_thread")]
#[ignore = "launches the production REST surface and real worker for a repeated-request tool-call journey"]
async fn should_complete_a_tool_call_and_reuse_the_worker() {
    timeout(E2E_TIMEOUT, run_deployed_rest_surface_litmus())
        .await
        .expect("the tool-call reuse journey must finish within 115 seconds");
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "streams Romeo and Juliet through Chat Completions"]
async fn should_stream_romeo_and_juliet_through_chat_completions() {
    timeout(
        E2E_TIMEOUT,
        run_serving_chat_request("text", text_chat_request_body()),
    )
    .await
    .expect("the Chat Completions journey must finish within 115 seconds");
}
