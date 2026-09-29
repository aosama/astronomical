//! Daemon IPC model-inventory journeys: discovered-model listing, release
//! catalog projection, download endpoints without a coordinator, and the
//! default-model set/reject round trip.

use std::sync::{Arc, RwLock};

use astronomical_config::leaf_model_id;
use astronomical_ipc_protocol::{
    DaemonIpcClient, DaemonRequest, DaemonResponse, DaemonWorkerStatus,
};
use astronomical_supervisor::DownloadCatalog;
use tokio::time::timeout;

use crate::common::daemon_ipc::{
    HANDSHAKE_TEST_TIMEOUT, fresh_instance_state_directory, ipc_chat_discovered_model,
    ipc_runtime_config, ready_stub_executor, start_stub_daemon_ipc_service,
    start_stub_daemon_ipc_service_with_runtime_config,
};

#[tokio::test]
async fn should_list_discovered_models_over_daemon_ipc() {
    let state_directory = fresh_instance_state_directory("models-list");
    let discovered_model = ipc_chat_discovered_model("test/local-chatter");
    let reloadable_config = Arc::new(RwLock::new(ipc_runtime_config(
        vec![discovered_model],
        &["test/local-chatter"],
    )));
    // The worker is ready with exactly this model, so the listing must mark
    // it resident.
    let stub_executor = ready_stub_executor("test/local-chatter");
    let daemon_ipc_service = start_stub_daemon_ipc_service_with_runtime_config(
        &state_directory,
        stub_executor,
        Some(reloadable_config),
    )
    .await;

    let mut daemon_client =
        DaemonIpcClient::connect(daemon_ipc_service.socket_path().to_path_buf())
            .await
            .expect("the client should connect to the running daemon IPC service");
    timeout(
        HANDSHAKE_TEST_TIMEOUT,
        daemon_client.send_request(&DaemonRequest::ModelsList),
    )
    .await
    .expect("the models list send should finish inside the test timeout")
    .expect("the models list request should transmit");
    let list_response = timeout(HANDSHAKE_TEST_TIMEOUT, daemon_client.next_response())
        .await
        .expect("the models list response should arrive inside the test timeout")
        .expect("the models list read should not fail at the transport layer")
        .expect("the daemon should answer the models list request");
    assert_eq!(
        list_response,
        DaemonResponse::ModelsList {
            models: vec![astronomical_ipc_protocol::DaemonListedModel {
                model_id: "test/local-chatter".to_owned(),
                family: "qwen3_5".to_owned(),
                context_window: Some(2_048),
                supports_embeddings: false,
                is_resident: true,
                size_bytes: 1_000_000_000,
            }],
        }
    );

    timeout(HANDSHAKE_TEST_TIMEOUT, daemon_ipc_service.shutdown())
        .await
        .expect("the daemon IPC service shutdown should finish inside the test timeout")
        .expect("the daemon IPC service should shut down cleanly");
    let _ = std::fs::remove_dir_all(&state_directory);
}

#[tokio::test]
async fn should_project_the_release_catalog_over_daemon_ipc() {
    let state_directory = fresh_instance_state_directory("catalog");
    let stub_executor = ready_stub_executor("test/local-chatter");
    let daemon_ipc_service = start_stub_daemon_ipc_service(&state_directory, stub_executor).await;

    let mut daemon_client =
        DaemonIpcClient::connect(daemon_ipc_service.socket_path().to_path_buf())
            .await
            .expect("the client should connect to the running daemon IPC service");
    timeout(
        HANDSHAKE_TEST_TIMEOUT,
        daemon_client.send_request(&DaemonRequest::Catalog),
    )
    .await
    .expect("the catalog send should finish inside the test timeout")
    .expect("the catalog request should transmit");
    let catalog_response = timeout(HANDSHAKE_TEST_TIMEOUT, daemon_client.next_response())
        .await
        .expect("the catalog response should arrive inside the test timeout")
        .expect("the catalog read should not fail at the transport layer")
        .expect("the daemon should answer the catalog request");
    let DaemonResponse::Catalog { entries } = catalog_response else {
        panic!("the daemon should answer with a catalog response");
    };
    // Structural assertions only: the bundled catalog ships with the release,
    // so couple to its invariants, not to specific entries.
    assert!(
        !entries.is_empty(),
        "the bundled catalog should project its entries"
    );
    for catalog_entry in &entries {
        assert!(
            !catalog_entry.huggingface_id.is_empty(),
            "every catalog entry must carry its hugging face id"
        );
        assert!(
            !catalog_entry.ready_on_this_mac,
            "no catalog entry can be ready without a download coordinator or discovery: {:?}",
            catalog_entry
        );
        assert!(
            catalog_entry.requestable_model_id.is_none(),
            "an entry that is not ready cannot expose a requestable model id"
        );
        assert!(
            catalog_entry.download_state.is_none(),
            "an entry with no active download cannot expose a download state"
        );
    }

    timeout(HANDSHAKE_TEST_TIMEOUT, daemon_ipc_service.shutdown())
        .await
        .expect("the daemon IPC service shutdown should finish inside the test timeout")
        .expect("the daemon IPC service should shut down cleanly");
    let _ = std::fs::remove_dir_all(&state_directory);
}

#[tokio::test]
async fn should_report_no_active_download_without_a_coordinator() {
    let state_directory = fresh_instance_state_directory("download-status");
    let stub_executor = ready_stub_executor("test/local-chatter");
    let daemon_ipc_service = start_stub_daemon_ipc_service(&state_directory, stub_executor).await;

    let mut daemon_client =
        DaemonIpcClient::connect(daemon_ipc_service.socket_path().to_path_buf())
            .await
            .expect("the client should connect to the running daemon IPC service");
    timeout(
        HANDSHAKE_TEST_TIMEOUT,
        daemon_client.send_request(&DaemonRequest::DownloadStatus),
    )
    .await
    .expect("the download status send should finish inside the test timeout")
    .expect("the download status request should transmit");
    let status_response = timeout(HANDSHAKE_TEST_TIMEOUT, daemon_client.next_response())
        .await
        .expect("the download status response should arrive inside the test timeout")
        .expect("the download status read should not fail at the transport layer")
        .expect("the daemon should answer the download status request");
    assert_eq!(
        status_response,
        DaemonResponse::DownloadStatus { job: None },
        "without a download coordinator the daemon reports no active job"
    );

    timeout(HANDSHAKE_TEST_TIMEOUT, daemon_ipc_service.shutdown())
        .await
        .expect("the daemon IPC service shutdown should finish inside the test timeout")
        .expect("the daemon IPC service should shut down cleanly");
    let _ = std::fs::remove_dir_all(&state_directory);
}

#[tokio::test]
async fn should_start_a_download_rejection_without_a_coordinator() {
    let state_directory = fresh_instance_state_directory("download-start");
    let stub_executor = ready_stub_executor("test/local-chatter");
    let daemon_ipc_service = start_stub_daemon_ipc_service(&state_directory, stub_executor).await;

    let mut daemon_client =
        DaemonIpcClient::connect(daemon_ipc_service.socket_path().to_path_buf())
            .await
            .expect("the client should connect to the running daemon IPC service");
    timeout(
        HANDSHAKE_TEST_TIMEOUT,
        daemon_client.send_request(&DaemonRequest::DownloadStart {
            model_id: "test/any-model".to_owned(),
        }),
    )
    .await
    .expect("the download start send should finish inside the test timeout")
    .expect("the download start request should transmit");
    let start_response = timeout(HANDSHAKE_TEST_TIMEOUT, daemon_client.next_response())
        .await
        .expect("the download start response should arrive inside the test timeout")
        .expect("the download start read should not fail at the transport layer")
        .expect("the daemon should answer the download start request");
    assert!(
        matches!(
            &start_response,
            DaemonResponse::RequestRejected { reason }
                if reason.contains("no Library download coordinator")
        ),
        "a download start without a coordinator must be rejected: {start_response:?}"
    );

    timeout(HANDSHAKE_TEST_TIMEOUT, daemon_ipc_service.shutdown())
        .await
        .expect("the daemon IPC service shutdown should finish inside the test timeout")
        .expect("the daemon IPC service should shut down cleanly");
    let _ = std::fs::remove_dir_all(&state_directory);
}

#[tokio::test]
async fn should_set_and_report_the_default_model_over_daemon_ipc() {
    let state_directory = fresh_instance_state_directory("default-model");
    let stub_executor = ready_stub_executor("test/local-chatter");
    let daemon_ipc_service = start_stub_daemon_ipc_service(&state_directory, stub_executor).await;

    // Derive the expected normalized id from the bundled catalog itself so the
    // test survives catalog repackaging.
    let bundled_catalog =
        DownloadCatalog::load_bundled().expect("the bundled release catalog should load in tests");
    let first_catalog_entry = &bundled_catalog.entries()[0];
    let requestable_model_id = leaf_model_id(first_catalog_entry.huggingface_id()).to_owned();

    // A requestable leaf id is accepted and normalized to the catalog entry's
    // requestable id.
    let mut set_client = DaemonIpcClient::connect(daemon_ipc_service.socket_path().to_path_buf())
        .await
        .expect("the client should connect to the running daemon IPC service");
    timeout(
        HANDSHAKE_TEST_TIMEOUT,
        set_client.send_request(&DaemonRequest::DefaultModelSet {
            model_id: requestable_model_id.clone(),
        }),
    )
    .await
    .expect("the default model set send should finish inside the test timeout")
    .expect("the default model set request should transmit");
    let set_response = timeout(HANDSHAKE_TEST_TIMEOUT, set_client.next_response())
        .await
        .expect("the default model set response should arrive inside the test timeout")
        .expect("the default model set read should not fail at the transport layer")
        .expect("the daemon should answer the default model set request");
    assert_eq!(
        set_response,
        DaemonResponse::DefaultModelSet {
            default_model_id: requestable_model_id.clone(),
        }
    );

    // The persisted default is visible to a fresh status probe, proving the
    // write round-trips through the instance config file.
    drop(set_client);
    let mut status_client =
        DaemonIpcClient::connect(daemon_ipc_service.socket_path().to_path_buf())
            .await
            .expect("the status client should connect to the running daemon IPC service");
    timeout(
        HANDSHAKE_TEST_TIMEOUT,
        status_client.send_request(&DaemonRequest::Status),
    )
    .await
    .expect("the status send should finish inside the test timeout")
    .expect("the status request should transmit");
    let status_response = timeout(HANDSHAKE_TEST_TIMEOUT, status_client.next_response())
        .await
        .expect("the status response should arrive inside the test timeout")
        .expect("the status read should not fail at the transport layer")
        .expect("the daemon should answer the status request");
    assert_eq!(
        status_response,
        DaemonResponse::Status {
            worker_status: DaemonWorkerStatus::Ready,
            ready_model_id: Some("test/local-chatter".to_owned()),
            default_model_id: Some(requestable_model_id),
        }
    );

    timeout(HANDSHAKE_TEST_TIMEOUT, daemon_ipc_service.shutdown())
        .await
        .expect("the daemon IPC service shutdown should finish inside the test timeout")
        .expect("the daemon IPC service should shut down cleanly");
    let _ = std::fs::remove_dir_all(&state_directory);
}

#[tokio::test]
async fn should_reject_an_unknown_default_model_over_daemon_ipc() {
    let state_directory = fresh_instance_state_directory("default-model-reject");
    let stub_executor = ready_stub_executor("test/local-chatter");
    let daemon_ipc_service = start_stub_daemon_ipc_service(&state_directory, stub_executor).await;

    let mut daemon_client =
        DaemonIpcClient::connect(daemon_ipc_service.socket_path().to_path_buf())
            .await
            .expect("the client should connect to the running daemon IPC service");
    timeout(
        HANDSHAKE_TEST_TIMEOUT,
        daemon_client.send_request(&DaemonRequest::DefaultModelSet {
            model_id: "nope/unknown-model".to_owned(),
        }),
    )
    .await
    .expect("the default model set send should finish inside the test timeout")
    .expect("the default model set request should transmit");
    let set_response = timeout(HANDSHAKE_TEST_TIMEOUT, daemon_client.next_response())
        .await
        .expect("the default model set response should arrive inside the test timeout")
        .expect("the default model set read should not fail at the transport layer")
        .expect("the daemon should answer the default model set request");
    assert!(
        matches!(
            &set_response,
            DaemonResponse::RequestRejected { reason }
                if reason.contains("nope/unknown-model is unknown")
        ),
        "an unknown default model must be rejected with its id named: {set_response:?}"
    );

    timeout(HANDSHAKE_TEST_TIMEOUT, daemon_ipc_service.shutdown())
        .await
        .expect("the daemon IPC service shutdown should finish inside the test timeout")
        .expect("the daemon IPC service should shut down cleanly");
    let _ = std::fs::remove_dir_all(&state_directory);
}
