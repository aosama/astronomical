//! Observatory can pick the converted streaming model from the public list.

use tokio::time::timeout;

use super::support::{
    JOURNEY_TIMEOUT, STREAMING_MODEL_ID, assert_streaming_model_is_advertised,
    launch_streaming_model_rest_server, stop_streaming_model_rest_server,
};

#[tokio::test(flavor = "multi_thread")]
#[ignore = "advertises the converted Ornith-1.5-35B-A3B-OptiQ-4bit-expert-streaming model on GET /v1/models"]
async fn should_advertise_the_expert_streaming_model_as_its_own_identity() {
    timeout(JOURNEY_TIMEOUT, run_advertise_journey())
        .await
        .expect("the expert-streaming advertise journey must finish within 115 seconds");
}

async fn run_advertise_journey() {
    let rest_server = launch_streaming_model_rest_server().await;
    eprintln!("[aligned-expert-packs] phase=advertise model={STREAMING_MODEL_ID}");
    assert_streaming_model_is_advertised(rest_server.server_address).await;
    stop_streaming_model_rest_server(rest_server).await;
}
