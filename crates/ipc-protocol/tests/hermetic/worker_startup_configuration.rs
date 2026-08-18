//! Round-trip contract for the complete worker startup configuration boundary.

use std::path::PathBuf;

use astronomical_ipc_protocol::{
    ProtocolReader, ProtocolWriter, WorkerChunkingConfiguration, WorkerCommand, WorkerLogLevel,
    WorkerMtpPairingConfiguration, WorkerSpeculativePrefillConfiguration,
    WorkerStartupConfiguration,
};
use tokio::io::duplex;

const TEST_TRANSPORT_CAPACITY_BYTES: usize = 256 * 1024;

#[tokio::test]
async fn should_round_trip_worker_startup_configuration() {
    // Exercise every nested startup owner together so a newly added field cannot
    // silently disappear between the supervisor and worker process boundary.
    let worker_command = WorkerCommand::InitializeWorker(WorkerStartupConfiguration {
        global_prompt_cache_root_directory: PathBuf::from("/tmp/fictional-prompt-cache"),
        global_prompt_cache_maximum_size_bytes: 50_000_000_000,
        persistent_prompt_cache_enabled: false,
        chunking: WorkerChunkingConfiguration {
            fixed_prompt_processing_chunk_size_tokens: 2_048,
            fixed_ssd_streaming_prompt_processing_chunk_size_tokens: Some(256),
            full_attention_key_value_growth_tokens: 256,
            speculative_prefill_draft_forward_tokens: 2_048,
            prefill_graph_submission_layer_interval: 1,
            experimental_ssd_paging_generation_graph_submission_layer_interval: 3,
            prompt_cache_block_tokens: None,
            prompt_cache_common_prefix_stride_blocks: 4,
        },
        configured_maximum_mlx_memory_bytes: Some(8_000_000_000),
        mtp_enabled: true,
        mtp_draft_depth: Some(3),
        mtp_pairings: vec![WorkerMtpPairingConfiguration {
            target_model_id: "target-model".to_owned(),
            drafter_model_id: "target-model-mtp".to_owned(),
            drafter_model_directory: Some(PathBuf::from("/tmp/fictional-mtp-drafter")),
            discovered_drafter_revision: Some("0123456789ab".to_owned()),
        }],
        speculative_prefill: WorkerSpeculativePrefillConfiguration {
            enabled: true,
            target_model_id: Some("astronomical/target-model".to_owned()),
            draft_model_id: Some("astronomical/draft-model".to_owned()),
            draft_model_directory: None,
            minimum_prompt_tokens: 8_192,
            keep_percentage: 20,
            selection_chunck_token_count: 32,
            mandatory_trailing_token_count: 512,
            lookahead_token_count: 8,
            importance_pooling_kernel_token_count: 13,
        },
        performance_attribution_enabled: false,
        logging_directory: PathBuf::from("/tmp/fictional-logs"),
        logging_level: WorkerLogLevel::Warn,
        retained_log_file_count: 1,
    });
    let (supervisor_transport, worker_transport) = duplex(TEST_TRANSPORT_CAPACITY_BYTES);
    let mut supervisor_writer = ProtocolWriter::new(supervisor_transport);
    let mut worker_reader = ProtocolReader::new(worker_transport);

    supervisor_writer
        .send_command(&worker_command)
        .await
        .expect("worker startup configuration should be written");

    assert_eq!(
        worker_reader
            .next_command()
            .await
            .expect("worker startup configuration should decode"),
        Some(worker_command)
    );
}
