//! Observatory can pick K2 Horizon MoVA from the public model list.

use tokio::time::timeout;

use super::support::{
    JOURNEY_TIMEOUT, assert_k2_is_advertised, launch_k2_rest_server, stop_k2_rest_server,
};

#[tokio::test(flavor = "multi_thread")]
#[ignore = "advertises a configured K2 Horizon MoVA artifact on GET /v1/models"]
async fn should_advertise_k2_horizon_mova_for_chat_and_tools() {
    timeout(JOURNEY_TIMEOUT, run_advertise_journey())
        .await
        .expect("the K2 Horizon MoVA advertise journey must finish within 115 seconds");
}

async fn run_advertise_journey() {
    let (_isolated_development_home, rest_server) = launch_k2_rest_server().await;
    eprintln!("[k2-horizon-mova] phase=advertise");
    assert_k2_is_advertised(rest_server.server_address).await;
    stop_k2_rest_server(rest_server).await;
}
