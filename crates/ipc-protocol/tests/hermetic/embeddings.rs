//! IPC round-trip and validation coverage for native embeddings commands.

use astronomical_ipc_protocol::{
    EmbeddingEncodingFormat, EmbeddingsCommand, EmbeddingsFailureReason, ProtocolReader,
    ProtocolWriter, RequestId, WorkerCommand, WorkerEvent,
};
use tokio::io::duplex;

const TEST_TRANSPORT_CAPACITY_BYTES: usize = 256 * 1024;

#[tokio::test]
async fn should_round_trip_one_embeddings_command() {
    let worker_command = WorkerCommand::GenerateEmbeddings(EmbeddingsCommand {
        request_id: RequestId::new(80),
        model: "nomicai-modernbert-embed-base-8bit".to_owned(),
        inputs: vec![
            "O Romeo, Romeo, wherefore art thou Romeo?".to_owned(),
            "Two households, both alike in dignity.".to_owned(),
        ],
        encoding_format: EmbeddingEncodingFormat::Base64,
        dimensions: Some(256),
    });
    let (supervisor_transport, worker_transport) = duplex(TEST_TRANSPORT_CAPACITY_BYTES);
    let mut supervisor_writer = ProtocolWriter::new(supervisor_transport);
    let mut worker_reader = ProtocolReader::new(worker_transport);

    supervisor_writer
        .send_command(&worker_command)
        .await
        .expect("a bounded embeddings command should be written");

    assert_eq!(
        worker_reader
            .next_command()
            .await
            .expect("the embeddings command frame should decode"),
        Some(worker_command)
    );
}

#[tokio::test]
async fn should_round_trip_embeddings_completed_failed_and_finalized_events() {
    let completed_event = WorkerEvent::EmbeddingsCompleted {
        request_id: RequestId::new(81),
        embeddings: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        input_token_counts: vec![8, 6],
        elapsed_millis: 42,
    };
    let failed_event = WorkerEvent::EmbeddingsFailed {
        request_id: RequestId::new(81),
        reason: EmbeddingsFailureReason::InvalidRequest {
            reason: "dimension exceeds the native vector width".to_owned(),
        },
    };
    let finalized_event = WorkerEvent::EmbeddingsFinalized {
        request_id: RequestId::new(81),
        elapsed_millis: 44,
        mlx_memory_snapshot: None,
    };

    let (supervisor_transport, worker_transport) = duplex(TEST_TRANSPORT_CAPACITY_BYTES);
    let mut worker_writer = ProtocolWriter::new(worker_transport);
    let mut supervisor_reader = ProtocolReader::new(supervisor_transport);

    worker_writer
        .send_event(&completed_event)
        .await
        .expect("a completed embeddings event should be written");
    worker_writer
        .send_event(&failed_event)
        .await
        .expect("a failed embeddings event should be written");
    worker_writer
        .send_event(&finalized_event)
        .await
        .expect("a finalized embeddings event should be written");

    assert_eq!(
        supervisor_reader
            .next_event()
            .await
            .expect("the completed embeddings frame should decode"),
        Some(completed_event)
    );
    assert_eq!(
        supervisor_reader
            .next_event()
            .await
            .expect("the failed embeddings event frame should decode"),
        Some(failed_event)
    );
    assert_eq!(
        supervisor_reader
            .next_event()
            .await
            .expect("the finalized embeddings event frame should decode"),
        Some(finalized_event)
    );
}

#[test]
fn should_reject_embeddings_input_above_the_ipc_boundary() {
    let oversized_inputs = vec!["x".repeat(8_193)];
    let embeddings_command = EmbeddingsCommand {
        request_id: RequestId::new(82),
        model: "nomicai-modernbert-embed-base-8bit".to_owned(),
        inputs: oversized_inputs,
        encoding_format: EmbeddingEncodingFormat::Float,
        dimensions: None,
    };

    let validation_error = embeddings_command
        .validate()
        .expect_err("an oversized embeddings input must fail at the IPC boundary");

    assert!(validation_error.to_string().contains("8193"));
}

#[test]
fn should_reject_zero_embedding_dimensions_at_the_ipc_boundary() {
    let embeddings_command = EmbeddingsCommand {
        request_id: RequestId::new(83),
        model: "nomicai-modernbert-embed-base-8bit".to_owned(),
        inputs: vec!["O Romeo, Romeo, wherefore art thou Romeo?".to_owned()],
        encoding_format: EmbeddingEncodingFormat::Float,
        dimensions: Some(0),
    };

    let validation_error = embeddings_command
        .validate()
        .expect_err("dimensions 0 must fail at the IPC boundary");

    assert!(validation_error.to_string().contains("positive"));
}
