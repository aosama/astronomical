use astronomical_ipc_protocol::{
    ChatGenerationCommand, ChatGenerationOutput, ChatGenerationSettings, ChatMessage,
    ChatToolChoice, ExpertMemoryMode, MAX_IPC_FRAME_BYTES, MlxMemorySnapshotSource, ProtocolReader,
    ProtocolWriter, RequestId, SpeculativePrefillRuntimeState, WorkerCommand, WorkerEvent,
    WorkerExpertResidencySnapshot, WorkerMlxMemorySnapshot, decode_command,
};
use futures_util::StreamExt;
use tokio::io::duplex;
use tokio::time::{Duration, Instant, timeout};
use tokio_util::codec::{FramedRead, LengthDelimitedCodec};

const TEST_TRANSPORT_CAPACITY_BYTES: usize = 256 * 1024;
const RETIRED_SMALL_FRAME_BYTES: usize = 64 * 1024;

#[test]
fn should_reject_a_whitespace_only_chat_model_id_before_worker_preprocessing() {
    let serialized_command = serde_json::json!({
        "kind": "generate",
        "request_id": 1,
        "model": "  ",
        "messages": [{"role": "user", "content": "Romeo and Juliet", "images": []}],
        "tools": [],
        "tool_choice": {"kind": "none"},
        "settings": {
            "max_output_tokens": 1,
            "temperature_thousandths": null,
            "top_p_thousandths": null,
            "seed": null,
            "thinking_budget": null
        }
    });

    let decoded_command =
        decode_command(&serde_json::to_vec(&serialized_command).expect("command should serialize"))
            .expect("the transport should preserve a structurally valid command");
    let WorkerCommand::Generate(chat_command) = decoded_command else {
        panic!("the chat command variant should survive transport");
    };
    assert!(chat_command.validate().is_err());
}

#[test]
fn should_serialize_speculative_prefill_runtime_state_as_snake_case() {
    assert_eq!(
        serde_json::to_string(&SpeculativePrefillRuntimeState::Unavailable)
            .expect("speculative-prefill state should serialize"),
        "\"unavailable\""
    );
}

#[tokio::test]
async fn should_round_trip_an_unversioned_chat_command() {
    let worker_command = WorkerCommand::Generate(ChatGenerationCommand {
        request_id: RequestId::new(71),
        model: "mlx-community/Ornith-1.0-35B-OptiQ-4bit".to_owned(),
        messages: vec![ChatMessage::User {
            content: "Explain this Rust function.".to_owned(),
            images: Vec::new(),
        }],
        tools: Vec::new(),
        tool_choice: ChatToolChoice::None,
        settings: ChatGenerationSettings {
            max_output_tokens: 128,
            temperature_thousandths: None,
            top_p_thousandths: None,
            seed: None,
            thinking_budget: None,
        },
        qwen_thinking_channel_seed: None,
    });
    let (supervisor_transport, worker_transport) = duplex(TEST_TRANSPORT_CAPACITY_BYTES);
    let mut supervisor_writer = ProtocolWriter::new(supervisor_transport);
    let mut worker_reader = ProtocolReader::new(worker_transport);

    supervisor_writer
        .send_command(&worker_command)
        .await
        .expect("a bounded unversioned chat command should be written");

    assert_eq!(
        worker_reader
            .next_command()
            .await
            .expect("the chat command frame should decode"),
        Some(worker_command)
    );
}

#[tokio::test]
async fn should_round_trip_expert_memory_mode_change_event() {
    let worker_event = WorkerEvent::ExpertMemoryModeChanged {
        expert_memory_mode: ExpertMemoryMode::Hybrid,
    };
    let (supervisor_transport, worker_transport) = duplex(TEST_TRANSPORT_CAPACITY_BYTES);
    let mut worker_writer = ProtocolWriter::new(worker_transport);
    let mut supervisor_reader = ProtocolReader::new(supervisor_transport);

    worker_writer
        .send_event(&worker_event)
        .await
        .expect("an expert memory mode event should be written");

    assert_eq!(
        supervisor_reader
            .next_event()
            .await
            .expect("the expert memory mode event frame should decode"),
        Some(worker_event)
    );
}

#[tokio::test]
async fn should_round_trip_generation_preparation_topology() {
    let worker_event = WorkerEvent::GenerationPreparationStarted {
        request_id: RequestId::new(72),
        total_layer_count: 40,
        resident_expert_count: 32,
        resident_expert_payload_bytes: 21_969_764_352,
    };
    let (supervisor_transport, worker_transport) = duplex(TEST_TRANSPORT_CAPACITY_BYTES);
    let mut worker_writer = ProtocolWriter::new(worker_transport);
    let mut supervisor_reader = ProtocolReader::new(supervisor_transport);

    worker_writer
        .send_event(&worker_event)
        .await
        .expect("generation-preparation topology should be written");

    assert_eq!(
        supervisor_reader
            .next_event()
            .await
            .expect("generation-preparation topology should decode"),
        Some(worker_event)
    );
}

#[tokio::test]
async fn should_round_trip_idle_worker_mlx_memory_ceiling() {
    let worker_event = WorkerEvent::Idle {
        machine_mlx_memory_ceiling_bytes: 40_000_000_000,
        effective_mlx_memory_ceiling_bytes: 32_000_000_000,
        minimum_mlx_memory_ceiling_bytes: 1,
    };
    let (supervisor_transport, worker_transport) = duplex(TEST_TRANSPORT_CAPACITY_BYTES);
    let mut worker_writer = ProtocolWriter::new(worker_transport);
    let mut supervisor_reader = ProtocolReader::new(supervisor_transport);

    worker_writer
        .send_event(&worker_event)
        .await
        .expect("the MLX memory ceiling event should be written");

    assert_eq!(
        supervisor_reader
            .next_event()
            .await
            .expect("the MLX memory ceiling event should decode"),
        Some(worker_event)
    );
}

#[tokio::test]
async fn should_round_trip_update_mlx_memory_limit_command() {
    let worker_command = WorkerCommand::UpdateMlxMemoryLimit {
        effective_mlx_memory_ceiling_bytes: 32_000_000_000,
        configuration_generation: "memory-generation".to_owned(),
    };
    let (supervisor_transport, worker_transport) = duplex(TEST_TRANSPORT_CAPACITY_BYTES);
    let mut supervisor_writer = ProtocolWriter::new(supervisor_transport);
    let mut worker_reader = ProtocolReader::new(worker_transport);

    supervisor_writer
        .send_command(&worker_command)
        .await
        .expect("the live memory-limit command should be written");
    assert_eq!(
        worker_reader
            .next_command()
            .await
            .expect("the live memory-limit command should decode"),
        Some(worker_command)
    );
}

#[tokio::test]
async fn should_round_trip_changed_and_rejected_memory_limit_events() {
    let worker_events = [
        WorkerEvent::MlxMemoryLimitChanged {
            effective_mlx_memory_ceiling_bytes: 32_000_000_000,
            minimum_mlx_memory_ceiling_bytes: 24_000_000_000,
            expert_memory_mode: ExpertMemoryMode::Paged,
            mlx_memory_snapshot: None,
            expert_residency: Some(WorkerExpertResidencySnapshot {
                total_layer_count: 24,
                resident_expert_count: 2,
                resident_expert_payload_bytes: 4_000,
            }),
        },
        WorkerEvent::MlxMemoryLimitRejected {
            requested_mlx_memory_ceiling_bytes: 20_000_000_000,
            minimum_mlx_memory_ceiling_bytes: 24_000_000_000,
            machine_mlx_memory_ceiling_bytes: 40_000_000_000,
            reason: "loaded model minimum exceeds the requested ceiling".to_owned(),
        },
    ];
    let (supervisor_transport, worker_transport) = duplex(TEST_TRANSPORT_CAPACITY_BYTES);
    let mut worker_writer = ProtocolWriter::new(worker_transport);
    let mut supervisor_reader = ProtocolReader::new(supervisor_transport);

    for worker_event in worker_events {
        worker_writer
            .send_event(&worker_event)
            .await
            .expect("the live memory-limit event should be written");
        assert_eq!(
            supervisor_reader
                .next_event()
                .await
                .expect("the live memory-limit event should decode"),
            Some(worker_event)
        );
    }
}

#[tokio::test]
async fn should_round_trip_the_idle_worker_mlx_memory_sample_command_and_response() {
    let worker_command = WorkerCommand::SampleMlxMemory;
    let worker_event = WorkerEvent::MlxMemorySample {
        mlx_memory_snapshot: Some(WorkerMlxMemorySnapshot {
            source: MlxMemorySnapshotSource::IdlePoll,
            active_memory_bytes: 28_510_000_000,
            allocator_cache_memory_bytes: 0,
            peak_memory_bytes: 29_120_000_000,
            expert_payload_bytes: 18_000_000_000,
            model_core_payload_bytes: 8_000_000_000,
            context_state_payload_bytes: 0,
            speculative_prefill_draft_memory_bytes: 0,
        }),
    };
    let (supervisor_transport, worker_transport) = duplex(TEST_TRANSPORT_CAPACITY_BYTES);
    let (supervisor_reader_transport, supervisor_writer_transport) =
        tokio::io::split(supervisor_transport);
    let (worker_reader_transport, worker_writer_transport) = tokio::io::split(worker_transport);
    let mut supervisor_writer = ProtocolWriter::new(supervisor_writer_transport);
    let mut supervisor_reader = ProtocolReader::new(supervisor_reader_transport);
    let mut worker_reader = ProtocolReader::new(worker_reader_transport);
    let mut worker_writer = ProtocolWriter::new(worker_writer_transport);

    supervisor_writer
        .send_command(&worker_command)
        .await
        .expect("the idle memory sample command should be written");
    assert_eq!(
        worker_reader
            .next_command()
            .await
            .expect("the idle memory sample command should decode"),
        Some(worker_command)
    );
    worker_writer
        .send_event(&worker_event)
        .await
        .expect("the idle memory sample event should be written");
    assert_eq!(
        supervisor_reader
            .next_event()
            .await
            .expect("the idle memory sample event should decode"),
        Some(worker_event)
    );
}

#[tokio::test]
async fn should_send_a_large_chat_command_as_one_bounded_frame() {
    let worker_command = WorkerCommand::Generate(ChatGenerationCommand {
        request_id: RequestId::new(73),
        model: "mlx-community/Ornith-1.0-35B-OptiQ-4bit".to_owned(),
        messages: vec![ChatMessage::User {
            content: "x".repeat(RETIRED_SMALL_FRAME_BYTES * 2),
            images: Vec::new(),
        }],
        tools: Vec::new(),
        tool_choice: ChatToolChoice::None,
        settings: ChatGenerationSettings {
            max_output_tokens: 128,
            temperature_thousandths: Some(600),
            top_p_thousandths: Some(950),
            seed: Some(7),
            thinking_budget: None,
        },
        qwen_thinking_channel_seed: None,
    });
    let serialized_command_bytes = serde_json::to_vec(&worker_command)
        .expect("the large typed command should serialize")
        .len();
    assert!(serialized_command_bytes > RETIRED_SMALL_FRAME_BYTES);
    assert!(serialized_command_bytes <= MAX_IPC_FRAME_BYTES);
    let (supervisor_transport, worker_transport) = duplex(TEST_TRANSPORT_CAPACITY_BYTES);
    let mut supervisor_writer = ProtocolWriter::new(supervisor_transport);
    let mut frame_codec = LengthDelimitedCodec::new();
    frame_codec.set_max_frame_length(MAX_IPC_FRAME_BYTES);
    let mut worker_reader = FramedRead::new(worker_transport, frame_codec);
    let sent_worker_command = worker_command.clone();
    let writer_task =
        tokio::spawn(async move { supervisor_writer.send_command(&sent_worker_command).await });

    let serialized_frame = worker_reader
        .next()
        .await
        .expect("the large chat command should produce one frame")
        .expect("the large chat command frame should be readable");
    assert_eq!(
        decode_command(&serialized_frame).expect("the large frame should decode"),
        worker_command
    );
    writer_task
        .await
        .expect("the writer task should not panic")
        .expect("the large chat command should fit one bounded frame");
}

#[tokio::test]
async fn should_round_trip_a_fifty_thousand_word_command_without_material_delay() {
    let worker_command = WorkerCommand::Generate(ChatGenerationCommand {
        request_id: RequestId::new(74),
        model: "mlx-community/Ornith-1.0-35B-OptiQ-4bit".to_owned(),
        messages: vec![ChatMessage::User {
            content: "public-domain-word ".repeat(50_000),
            images: Vec::new(),
        }],
        tools: Vec::new(),
        tool_choice: ChatToolChoice::None,
        settings: ChatGenerationSettings {
            max_output_tokens: 1,
            temperature_thousandths: Some(1_000),
            top_p_thousandths: Some(950),
            seed: Some(1),
            thinking_budget: None,
        },
        qwen_thinking_channel_seed: None,
    });
    let serialized_command_bytes = serde_json::to_vec(&worker_command)
        .expect("the 50K command should serialize")
        .len();
    let (supervisor_transport, worker_transport) = duplex(TEST_TRANSPORT_CAPACITY_BYTES);
    let mut supervisor_writer = ProtocolWriter::new(supervisor_transport);
    let mut worker_reader = ProtocolReader::new(worker_transport);
    let sent_worker_command = worker_command.clone();

    eprintln!("[ipc-50k] transferring {serialized_command_bytes} serialized bytes; ETA <5s");
    let transfer_started_at = Instant::now();
    let writer_task =
        tokio::spawn(async move { supervisor_writer.send_command(&sent_worker_command).await });
    let received_command = timeout(Duration::from_secs(5), worker_reader.next_command())
        .await
        .expect("the 50K IPC transfer must finish within five seconds")
        .expect("the 50K command should decode");
    writer_task
        .await
        .expect("the writer task should not panic")
        .expect("the 50K command should cross bounded frames");
    let transfer_duration = transfer_started_at.elapsed();
    eprintln!(
        "[ipc-50k] transfer_ms={:.2}",
        transfer_duration.as_secs_f64() * 1_000.0
    );

    assert_eq!(received_command, Some(worker_command));
}

#[tokio::test]
async fn should_round_trip_one_unified_chat_output_event() {
    let worker_event = WorkerEvent::Output {
        request_id: RequestId::new(72),
        sequence_number: 0,
        generated_token_count: 1,
        outputs: vec![ChatGenerationOutput::Text {
            text: "The function validates its input.".to_owned(),
        }],
        mlx_memory_snapshot: None,
    };
    let (supervisor_transport, worker_transport) = duplex(TEST_TRANSPORT_CAPACITY_BYTES);
    let mut worker_writer = ProtocolWriter::new(worker_transport);
    let mut supervisor_reader = ProtocolReader::new(supervisor_transport);

    worker_writer
        .send_event(&worker_event)
        .await
        .expect("a bounded unified output event should be written");

    assert_eq!(
        supervisor_reader
            .next_event()
            .await
            .expect("the output event frame should decode"),
        Some(worker_event)
    );
}

#[tokio::test]
async fn should_round_trip_generation_progress_event() {
    let worker_event = WorkerEvent::GenerationProgress {
        request_id: RequestId::new(82),
        generated_token_count: 7,
        maximum_output_tokens: 64,
        elapsed_millis: 900,
        mlx_memory_snapshot: None,
    };

    let (supervisor_transport, worker_transport) = duplex(TEST_TRANSPORT_CAPACITY_BYTES);
    let mut worker_writer = ProtocolWriter::new(worker_transport);
    let mut supervisor_reader = ProtocolReader::new(supervisor_transport);

    worker_writer
        .send_event(&worker_event)
        .await
        .expect("a generation progress event should be written");

    assert_eq!(
        supervisor_reader
            .next_event()
            .await
            .expect("the generation progress event frame should decode"),
        Some(worker_event)
    );
}

#[tokio::test]
async fn should_round_trip_first_decode_completion_event() {
    let worker_event = WorkerEvent::FirstDecodeCompleted {
        request_id: RequestId::new(83),
        elapsed_millis: 1_234,
    };
    let (supervisor_transport, worker_transport) = duplex(TEST_TRANSPORT_CAPACITY_BYTES);
    let mut worker_writer = ProtocolWriter::new(worker_transport);
    let mut supervisor_reader = ProtocolReader::new(supervisor_transport);

    worker_writer
        .send_event(&worker_event)
        .await
        .expect("a first decode completion event should be written");

    assert_eq!(
        supervisor_reader
            .next_event()
            .await
            .expect("the first decode completion frame should decode"),
        Some(worker_event)
    );
}

#[tokio::test]
async fn should_round_trip_one_persistent_prompt_cache_stats_event() {
    let worker_event = WorkerEvent::PersistentPromptCacheStats {
        persistent_prompt_cache_hits: 12,
        persistent_prompt_cache_misses: 3,
        persistent_prompt_cache_tokens_saved: 95_000,
        persistent_prompt_cache_block_token_count: 2_048,
        persistent_prompt_cache_sequence_state_block_count: 87,
        persistent_prompt_cache_boundary_state_snapshot_count: 1,
        persistent_prompt_cache_visual_embedding_count: 5,
        persistent_prompt_cache_total_size_bytes: 1_073_741_824,
        persistent_prompt_cache_visual_embedding_total_size_bytes: 222_222,
        persistent_prompt_cache_maximum_size_bytes: 123_456_789,
        persistent_prompt_cache_visual_embedding_hits: 4,
        persistent_prompt_cache_visual_embedding_misses: 2,
        persistent_prompt_cache_visual_embedding_rows_loaded: 256,
    };

    let (supervisor_transport, worker_transport) = duplex(TEST_TRANSPORT_CAPACITY_BYTES);
    let mut worker_writer = ProtocolWriter::new(worker_transport);
    let mut supervisor_reader = ProtocolReader::new(supervisor_transport);

    worker_writer
        .send_event(&worker_event)
        .await
        .expect("a persistent prompt-cache stats event should be written");

    assert_eq!(
        supervisor_reader
            .next_event()
            .await
            .expect("the persistent prompt-cache stats event frame should decode"),
        Some(worker_event)
    );
}
