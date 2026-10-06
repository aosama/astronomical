use astronomical_ipc_protocol::{
    ProtocolReader, ProtocolWriter, WorkerEvent, WorkerRuntimeFeatureConfiguration,
};
use tokio::io::duplex;

const TEST_TRANSPORT_CAPACITY_BYTES: usize = 256 * 1024;

#[tokio::test]
async fn should_preserve_generation_and_chunking_configuration_in_acknowledgement() {
    use astronomical_ipc_protocol::{
        WorkerChunkingConfiguration, WorkerLoadedAutoregressiveModelRuntimeConfiguration,
        WorkerLoadedModelRuntimeConfiguration,
    };

    let configuration_generation = "abcdef0123456789".repeat(4);
    let worker_event = WorkerEvent::RuntimeFeatureConfigurationApplied {
        worker_runtime_feature_configuration: WorkerRuntimeFeatureConfiguration {
            configuration_generation: configuration_generation.clone(),
            persistent_prompt_cache_enabled: true,
            prompt_cache_maximum_size_bytes: 12_000_000_000,
            loaded_model: Some(WorkerLoadedModelRuntimeConfiguration::Autoregressive(
                WorkerLoadedAutoregressiveModelRuntimeConfiguration {
                    model_id: "fictional/target".to_owned(),
                    maximum_context_tokens: 32_768,
                    maximum_output_tokens: 4_096,
                    chunking: WorkerChunkingConfiguration {
                        fixed_prompt_processing_chunk_size_tokens: 2_048,
                        fixed_ssd_streaming_prompt_processing_chunk_size_tokens: 256,
                        full_attention_key_value_growth_tokens: 256,
                        prefill_graph_submission_layer_interval: 0,
                        experimental_ssd_paging_prefill_graph_submission_layer_interval: 1,
                        experimental_ssd_paging_generation_graph_submission_layer_interval: 0,
                        prompt_cache_block_tokens: Some(128),
                        prompt_cache_common_prefix_stride_blocks: 4,
                        experimental_decode_stage_attribution_enabled: false,
                        experimental_quantized_kv_cache_enabled: false,
                        experimental_fused_moe_decode_enabled: false,
                    },
                },
            )),
        },
    };
    let (supervisor_transport, worker_transport) = duplex(TEST_TRANSPORT_CAPACITY_BYTES);
    let mut worker_writer = ProtocolWriter::new(worker_transport);
    let mut supervisor_reader = ProtocolReader::new(supervisor_transport);

    worker_writer
        .send_event(&worker_event)
        .await
        .expect("acknowledgement should write");
    let decoded_event = supervisor_reader
        .next_event()
        .await
        .expect("acknowledgement should decode")
        .expect("acknowledgement should be present");

    assert_eq!(decoded_event, worker_event);
    let serialized_event = serde_json::to_string(&decoded_event).expect("event should serialize");
    assert!(serialized_event.contains(&configuration_generation));
    assert!(!serialized_event.contains("/tmp/"));
}
