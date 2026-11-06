use astronomical_ipc_protocol::{ModelLoadInFlight, ProtocolReader, ProtocolWriter, WorkerEvent};
use tokio::io::duplex;

use super::announce_load_started;

const LOADING_MODEL_ID: &str = "loading-model";

async fn announced_report(model_id: Option<&str>) -> ModelLoadInFlight {
    let (event_writer_transport, event_reader_transport) = duplex(64 * 1024);
    let mut event_writer = ProtocolWriter::new(event_writer_transport);
    announce_load_started(model_id, &mut event_writer)
        .await
        .expect("the daemon hears the announcement");
    drop(event_writer);
    let mut event_reader = ProtocolReader::new(event_reader_transport);
    match event_reader.next_event().await {
        Ok(Some(WorkerEvent::ModelLoadInFlight {
            model_load_in_flight,
        })) => model_load_in_flight,
        unexpected_event => panic!("unexpected event: {unexpected_event:?}"),
    }
}

#[tokio::test]
async fn should_name_the_model_being_loaded() {
    assert_eq!(
        announced_report(Some(LOADING_MODEL_ID)).await,
        ModelLoadInFlight::for_model(LOADING_MODEL_ID)
    );
}

#[tokio::test]
async fn should_leave_the_report_unnamed_when_the_model_is_not_known() {
    assert_eq!(announced_report(None).await, ModelLoadInFlight::unnamed());
}

#[tokio::test]
async fn should_end_with_an_error_when_the_daemon_can_no_longer_be_reached() {
    let (event_writer_transport, event_reader_transport) = duplex(64 * 1024);
    drop(event_reader_transport);
    let mut event_writer = ProtocolWriter::new(event_writer_transport);

    assert!(
        announce_load_started(Some(LOADING_MODEL_ID), &mut event_writer)
            .await
            .is_err()
    );
}
