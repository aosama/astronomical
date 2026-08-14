use std::time::Duration;

use astronomical_ipc_protocol::{ProtocolReader, ProtocolWriter, WorkerCommand, WorkerEvent};
use tokio::io::duplex;

const TEST_TIMEOUT: Duration = Duration::from_secs(5);
const TEST_TRANSPORT_CAPACITY_BYTES: usize = 16 * 1024;

#[tokio::test]
async fn should_round_trip_global_clear_prompt_cache_command() {
    run_bounded_test(async {
        let worker_command = WorkerCommand::ClearPromptCache { model_id: None };
        assert_command_round_trip(worker_command).await;
    })
    .await;
}

#[tokio::test]
async fn should_round_trip_scoped_clear_prompt_cache_command() {
    run_bounded_test(async {
        let worker_command = WorkerCommand::ClearPromptCache {
            model_id: Some("astronomical/requested-model".to_owned()),
        };
        assert_command_round_trip(worker_command).await;
    })
    .await;
}

#[tokio::test]
async fn should_round_trip_prompt_cache_cleared_event() {
    run_bounded_test(async {
        let worker_event = WorkerEvent::PromptCacheCleared {
            model_id: Some("astronomical/requested-model".to_owned()),
            blocks_removed: 1_247,
            bytes_freed: 8_589_934_592,
        };
        let (supervisor_transport, worker_transport) = duplex(TEST_TRANSPORT_CAPACITY_BYTES);
        let mut worker_writer = ProtocolWriter::new(worker_transport);
        let mut supervisor_reader = ProtocolReader::new(supervisor_transport);

        worker_writer
            .send_event(&worker_event)
            .await
            .expect("prompt-cache cleared event should be written");
        assert_eq!(
            supervisor_reader
                .next_event()
                .await
                .expect("prompt-cache cleared event should decode"),
            Some(worker_event)
        );
    })
    .await;
}

async fn assert_command_round_trip(worker_command: WorkerCommand) {
    let (supervisor_transport, worker_transport) = duplex(TEST_TRANSPORT_CAPACITY_BYTES);
    let mut supervisor_writer = ProtocolWriter::new(supervisor_transport);
    let mut worker_reader = ProtocolReader::new(worker_transport);

    supervisor_writer
        .send_command(&worker_command)
        .await
        .expect("prompt-cache clear command should be written");
    assert_eq!(
        worker_reader
            .next_command()
            .await
            .expect("prompt-cache clear command should decode"),
        Some(worker_command)
    );
}

async fn run_bounded_test(test_journey: impl std::future::Future<Output = ()>) {
    tokio::time::timeout(TEST_TIMEOUT, test_journey)
        .await
        .expect("prompt-cache protocol test should finish within five seconds");
}
