//! Daemon IPC model-inventory journeys: discovered-model listing, release
//! catalog projection, download endpoints without a coordinator, and the
//! default-model set/reject round trip.

use std::sync::Arc;

use astronomical_daemon::DownloadCatalog;
use ipc_protocol::{DaemonRequest, DaemonResponse, ModelLoadInFlight};

use crate::integration::daemon_ipc_exchange::request_and_read_first_response;
use crate::integration::daemon_ipc_service_fixtures::{
    ready_idle_status_response, shut_down_and_remove_state_directory, start_ready_chat_service,
    start_service_with_installed_models,
};
use crate::support::daemon_ipc::{
    fresh_instance_state_directory, ipc_chat_discovered_model, ipc_decision_discovered_model,
    ready_stub_executor,
};

#[tokio::test]
async fn should_list_discovered_models_over_daemon_ipc() {
    let state_directory = fresh_instance_state_directory("models-list");
    // The worker is ready with exactly this model, so the listing must mark
    // it resident.
    let daemon_ipc_service = start_service_with_installed_models(
        &state_directory,
        ready_stub_executor("test/local-chatter"),
        vec![ipc_chat_discovered_model("test/local-chatter")],
        &["test/local-chatter"],
    )
    .await;

    let list_response = request_and_read_first_response(
        daemon_ipc_service.socket_path(),
        &DaemonRequest::ModelsList,
    )
    .await;
    assert_eq!(
        list_response,
        DaemonResponse::ModelsList {
            models: vec![ipc_protocol::DaemonListedModel {
                model_id: "test/local-chatter".to_owned(),
                family: "qwen3_5".to_owned(),
                context_window: Some(2_048),
                supports_embeddings: false,
                supports_decisions: false,
                supports_speech: false,
                supports_image_generation: false,
                is_resident: true,
                size_bytes: 1_000_000_000,
            }],
        }
    );

    shut_down_and_remove_state_directory(daemon_ipc_service, &state_directory).await;
}

#[tokio::test]
async fn should_list_a_decision_model_with_its_capability_and_no_context_window() {
    let state_directory = fresh_instance_state_directory("models-decider");
    let daemon_ipc_service = start_service_with_installed_models(
        &state_directory,
        ready_stub_executor("test/local-chatter"),
        vec![ipc_decision_discovered_model("test-local-decider")],
        &[],
    )
    .await;
    let list_response = request_and_read_first_response(
        daemon_ipc_service.socket_path(),
        &DaemonRequest::ModelsList,
    )
    .await;

    assert_eq!(
        list_response,
        DaemonResponse::ModelsList {
            models: vec![ipc_protocol::DaemonListedModel {
                model_id: "test-local-decider".to_owned(),
                family: "laya".to_owned(),
                context_window: None,
                supports_embeddings: false,
                supports_decisions: true,
                supports_speech: false,
                supports_image_generation: false,
                is_resident: false,
                size_bytes: 500_000_000,
            }],
        }
    );
    shut_down_and_remove_state_directory(daemon_ipc_service, &state_directory).await;
}

/// The worker has already dropped the previous model by the time a swap reports, so
/// the listing must not call it resident just because the snapshot still names it.
#[tokio::test]
async fn should_list_nothing_as_resident_during_a_model_load() {
    let state_directory = fresh_instance_state_directory("models-load");
    let mut stub_executor = ready_stub_executor("test/local-chatter");
    Arc::get_mut(&mut stub_executor)
        .expect("the stub executor should be uniquely owned here")
        .health_snapshot
        .model_load_in_flight = Some(ModelLoadInFlight::for_model("test/next"));
    let daemon_ipc_service = start_service_with_installed_models(
        &state_directory,
        stub_executor,
        vec![ipc_chat_discovered_model("test/local-chatter")],
        &["test/local-chatter"],
    )
    .await;
    let list_response = request_and_read_first_response(
        daemon_ipc_service.socket_path(),
        &DaemonRequest::ModelsList,
    )
    .await;

    match list_response {
        DaemonResponse::ModelsList { models } => {
            assert_eq!(models.len(), 1);
            assert!(!models[0].is_resident);
        }
        unexpected_response => panic!("expected a model list: {unexpected_response:?}"),
    }
    shut_down_and_remove_state_directory(daemon_ipc_service, &state_directory).await;
}

#[tokio::test]
async fn should_project_the_release_catalog_over_daemon_ipc() {
    let (daemon_ipc_service, _stub_executor, state_directory) =
        start_ready_chat_service("catalog").await;

    let catalog_response =
        request_and_read_first_response(daemon_ipc_service.socket_path(), &DaemonRequest::Catalog)
            .await;
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

    shut_down_and_remove_state_directory(daemon_ipc_service, &state_directory).await;
}

#[tokio::test]
async fn should_report_no_active_download_without_a_coordinator() {
    let (daemon_ipc_service, _stub_executor, state_directory) =
        start_ready_chat_service("download-status").await;

    let status_response = request_and_read_first_response(
        daemon_ipc_service.socket_path(),
        &DaemonRequest::DownloadStatus,
    )
    .await;
    assert_eq!(
        status_response,
        DaemonResponse::DownloadStatus {
            job: None,
            waiting_huggingface_ids: Vec::new(),
        },
        "without a download coordinator the daemon reports no active job"
    );

    shut_down_and_remove_state_directory(daemon_ipc_service, &state_directory).await;
}

#[tokio::test]
async fn should_start_a_download_rejection_without_a_coordinator() {
    let (daemon_ipc_service, _stub_executor, state_directory) =
        start_ready_chat_service("download-start").await;

    let start_response = request_and_read_first_response(
        daemon_ipc_service.socket_path(),
        &DaemonRequest::DownloadStart {
            model_id: "test/any-model".to_owned(),
        },
    )
    .await;
    assert!(
        matches!(
            &start_response,
            DaemonResponse::RequestRejected { reason }
                if reason.contains("no Library download coordinator")
        ),
        "a download start without a coordinator must be rejected: {start_response:?}"
    );

    shut_down_and_remove_state_directory(daemon_ipc_service, &state_directory).await;
}

#[tokio::test]
async fn should_set_and_report_the_default_model_over_daemon_ipc() {
    let (daemon_ipc_service, _stub_executor, state_directory) =
        start_ready_chat_service("default-model").await;

    // Derive the expected normalized id from the bundled catalog itself so the
    // test survives catalog repackaging.
    let bundled_catalog =
        DownloadCatalog::load_bundled().expect("the bundled release catalog should load in tests");
    let first_catalog_entry = &bundled_catalog.entries()[0];
    let requestable_model_id =
        config::leaf_model_id(first_catalog_entry.huggingface_id()).to_owned();

    // A requestable leaf id is accepted and normalized to the catalog entry's
    // requestable id.
    let set_response = request_and_read_first_response(
        daemon_ipc_service.socket_path(),
        &DaemonRequest::DefaultModelSet {
            model_id: requestable_model_id.clone(),
        },
    )
    .await;
    assert_eq!(
        set_response,
        DaemonResponse::DefaultModelSet {
            default_model_id: requestable_model_id.clone(),
        }
    );

    // The persisted default is visible to a fresh status probe, proving the
    // write round-trips through the instance config file.
    let status_response =
        request_and_read_first_response(daemon_ipc_service.socket_path(), &DaemonRequest::Status)
            .await;
    assert_eq!(
        status_response,
        ready_idle_status_response("test/local-chatter", &requestable_model_id)
    );

    shut_down_and_remove_state_directory(daemon_ipc_service, &state_directory).await;
}

#[tokio::test]
async fn should_reject_an_unknown_default_model_over_daemon_ipc() {
    let (daemon_ipc_service, _stub_executor, state_directory) =
        start_ready_chat_service("default-model-reject").await;

    let set_response = request_and_read_first_response(
        daemon_ipc_service.socket_path(),
        &DaemonRequest::DefaultModelSet {
            model_id: "nope/unknown-model".to_owned(),
        },
    )
    .await;
    assert!(
        matches!(
            &set_response,
            DaemonResponse::RequestRejected { reason }
                if reason.contains("nope/unknown-model is unknown")
        ),
        "an unknown default model must be rejected with its id named: {set_response:?}"
    );

    shut_down_and_remove_state_directory(daemon_ipc_service, &state_directory).await;
}
