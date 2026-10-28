use super::scripted_chat_test_doubles::{
    MalformedFinishProcessor, ScriptedChatEngine, ScriptedChatProcessor,
};
use super::support::{chat_command, close_worker_transport, next_event, ready_event};
use super::tracking_chat_engine_test_doubles::TrackingChatEngine;
use crate::{EngineBackedWorker, GeneratedToken};
use astronomical_ipc_protocol::{
    ChatGenerationCompletionReason, ChatGenerationFailureReason, ChatGenerationOutput,
    MAX_IPC_FRAME_BYTES, ProtocolReader, ProtocolWriter, RequestId, WorkerCommand, WorkerEvent,
};
use std::time::Duration;
use tokio::io::{duplex, split};
use tokio::time::timeout;
#[tokio::test]
async fn should_emit_ordered_outputs_and_reuse_capacity_after_cancellation() {
    let engine_worker =
        EngineBackedWorker::new(ScriptedChatProcessor::new(), ScriptedChatEngine::new());
    let (daemon_transport, worker_transport) = duplex(MAX_IPC_FRAME_BYTES * 2);
    let (daemon_reader_transport, daemon_writer_transport) = split(daemon_transport);
    let (worker_reader_transport, worker_writer_transport) = split(worker_transport);
    let mut daemon_reader = ProtocolReader::new(daemon_reader_transport);
    let mut daemon_writer = ProtocolWriter::new(daemon_writer_transport);
    let worker_task = tokio::spawn(async move {
        engine_worker
            .run(worker_reader_transport, worker_writer_transport)
            .await
    });

    assert_eq!(next_event(&mut daemon_reader).await, ready_event());
    daemon_writer
        .send_command(&WorkerCommand::Generate(chat_command(711, 7)))
        .await
        .expect("the worker should receive a chat request");

    assert_eq!(
        next_event(&mut daemon_reader).await,
        WorkerEvent::Output {
            request_id: RequestId::new(711),
            sequence_number: 0,
            generated_token_count: 1,
            outputs: vec![
                ChatGenerationOutput::Reasoning {
                    text: "I should inspect the source tree.".to_owned(),
                },
                ChatGenerationOutput::Text {
                    text: "I found Rust files.".to_owned(),
                },
            ],
            mlx_memory_snapshot: None,
            expert_residency: None,
        }
    );
    assert_eq!(
        next_event(&mut daemon_reader).await,
        WorkerEvent::Output {
            request_id: RequestId::new(711),
            sequence_number: 2,
            generated_token_count: 2,
            outputs: vec![ChatGenerationOutput::ToolCall {
                tool_call_index: 0,
                function_name: "glob".to_owned(),
                arguments_json: r#"{"pattern":"src/**/*.rs"}"#.to_owned(),
            }],
            mlx_memory_snapshot: None,
            expert_residency: None,
        }
    );
    assert_eq!(
        next_event(&mut daemon_reader).await,
        WorkerEvent::Completed {
            persistent_prompt_cache_diagnostics: None,
            request_id: RequestId::new(711),
            prompt_token_count: 1,
            generated_token_count: 2,
            reasoning_token_count: 0,
            cached_token_count: 0,
            reason: ChatGenerationCompletionReason::ToolCalls,
        }
    );

    daemon_writer
        .send_command(&WorkerCommand::Generate(chat_command(712, 8)))
        .await
        .expect("the worker should receive a follow-up request");
    daemon_writer
        .send_command(&WorkerCommand::Cancel {
            request_id: RequestId::new(712),
        })
        .await
        .expect("the worker should receive cancellation");
    assert_eq!(
        next_event(&mut daemon_reader).await,
        WorkerEvent::Completed {
            persistent_prompt_cache_diagnostics: None,
            request_id: RequestId::new(712),
            prompt_token_count: 1,
            generated_token_count: 0,
            reasoning_token_count: 0,
            cached_token_count: 0,
            reason: ChatGenerationCompletionReason::Cancelled,
        }
    );

    close_worker_transport(daemon_writer, worker_task).await;
}

#[tokio::test]
async fn should_emit_every_parallel_tool_call_completed_after_thinking() {
    let engine_worker = EngineBackedWorker::new(
        ScriptedChatProcessor::with_parallel_tool_calls(),
        ScriptedChatEngine::with_cached_token_count_and_generated_tokens(
            0,
            vec![
                GeneratedToken::TokenId {
                    token_id: 1,
                    is_reasoning_token: true,
                    expert_memory_mode: None,
                    mlx_memory_telemetry: None,
                    first_decode_forward_elapsed_millis: Some(123),
                    generation_finalization: None,
                },
                GeneratedToken::TokenId {
                    token_id: 2,
                    is_reasoning_token: false,
                    expert_memory_mode: None,
                    mlx_memory_telemetry: None,
                    first_decode_forward_elapsed_millis: None,
                    generation_finalization: None,
                },
                GeneratedToken::TokenId {
                    token_id: 3,
                    is_reasoning_token: false,
                    expert_memory_mode: None,
                    mlx_memory_telemetry: None,
                    first_decode_forward_elapsed_millis: None,
                    generation_finalization: None,
                },
            ],
        ),
    );
    let (daemon_transport, worker_transport) = duplex(MAX_IPC_FRAME_BYTES * 2);
    let (daemon_reader_transport, daemon_writer_transport) = split(daemon_transport);
    let (worker_reader_transport, worker_writer_transport) = split(worker_transport);
    let mut daemon_reader = ProtocolReader::new(daemon_reader_transport);
    let mut daemon_writer = ProtocolWriter::new(daemon_writer_transport);
    let worker_task = tokio::spawn(async move {
        engine_worker
            .run(worker_reader_transport, worker_writer_transport)
            .await
    });

    assert_eq!(next_event(&mut daemon_reader).await, ready_event());
    let mut parallel_tool_call_command = chat_command(713, 7);
    parallel_tool_call_command.settings.max_output_tokens = 3;
    daemon_writer
        .send_command(&WorkerCommand::Generate(parallel_tool_call_command))
        .await
        .expect("the worker should receive a chat request");
    assert_eq!(
        next_event(&mut daemon_reader).await,
        WorkerEvent::FirstDecodeCompleted {
            request_id: RequestId::new(713),
            elapsed_millis: 123,
        }
    );
    assert_eq!(
        next_event(&mut daemon_reader).await,
        WorkerEvent::Output {
            request_id: RequestId::new(713),
            sequence_number: 0,
            generated_token_count: 1,
            outputs: vec![
                ChatGenerationOutput::Reasoning {
                    text: "I should inspect the source tree.".to_owned(),
                },
                ChatGenerationOutput::Text {
                    text: "I found Rust files.".to_owned(),
                },
            ],
            mlx_memory_snapshot: None,
            expert_residency: None,
        }
    );
    assert_eq!(
        next_event(&mut daemon_reader).await,
        WorkerEvent::Output {
            request_id: RequestId::new(713),
            sequence_number: 2,
            generated_token_count: 2,
            outputs: vec![ChatGenerationOutput::ToolCall {
                tool_call_index: 0,
                function_name: "glob".to_owned(),
                arguments_json: r#"{"pattern":"src/**/*.rs"}"#.to_owned(),
            }],
            mlx_memory_snapshot: None,
            expert_residency: None,
        }
    );
    assert_eq!(
        timeout(Duration::from_millis(250), next_event(&mut daemon_reader),)
            .await
            .expect(
                "the worker should continue generation until every parallel tool call is emitted"
            ),
        WorkerEvent::Output {
            request_id: RequestId::new(713),
            sequence_number: 3,
            generated_token_count: 3,
            outputs: vec![ChatGenerationOutput::ToolCall {
                tool_call_index: 1,
                function_name: "glob".to_owned(),
                arguments_json: r#"{"pattern":"tests/**/*.rs"}"#.to_owned(),
            }],
            mlx_memory_snapshot: None,
            expert_residency: None,
        }
    );
    assert_eq!(
        next_event(&mut daemon_reader).await,
        WorkerEvent::Completed {
            persistent_prompt_cache_diagnostics: None,
            request_id: RequestId::new(713),
            prompt_token_count: 1,
            generated_token_count: 3,
            reasoning_token_count: 1,
            cached_token_count: 0,
            reason: ChatGenerationCompletionReason::ToolCalls,
        }
    );

    close_worker_transport(daemon_writer, worker_task).await;
}

#[tokio::test]
async fn should_report_a_bounded_actionable_reason_before_a_fatal_worker_exit() {
    let engine_worker = EngineBackedWorker::new(
        ScriptedChatProcessor::new(),
        ScriptedChatEngine::with_fatal_decode_reason(
            "direct MLX execution failed: [metal::malloc] Attempting to allocate 34359738368 bytes which is greater than the maximum allowed buffer size of 30150672384 bytes. at /private/build/mlx/array.cpp:352",
        ),
    );
    let (daemon_transport, worker_transport) = duplex(MAX_IPC_FRAME_BYTES * 2);
    let (daemon_reader_transport, daemon_writer_transport) = split(daemon_transport);
    let (worker_reader_transport, worker_writer_transport) = split(worker_transport);
    let mut daemon_reader = ProtocolReader::new(daemon_reader_transport);
    let mut daemon_writer = ProtocolWriter::new(daemon_writer_transport);
    let worker_task = tokio::spawn(async move {
        engine_worker
            .run(worker_reader_transport, worker_writer_transport)
            .await
    });

    assert_eq!(next_event(&mut daemon_reader).await, ready_event());
    daemon_writer
        .send_command(&WorkerCommand::Generate(chat_command(799, 7)))
        .await
        .expect("the worker should receive the fatal test request");

    assert_eq!(
        next_event(&mut daemon_reader).await,
        WorkerEvent::Failed {
            request_id: RequestId::new(799),
            reason: ChatGenerationFailureReason::FatalExecution {
                reason: "GPU allocation exceeded the platform buffer limit while evaluating the model; reduce the prompt size or configured prefill chunk size".to_owned(),
            },
        }
    );
    let worker_error = worker_task
        .await
        .expect("the worker task should join")
        .expect_err("the worker must still exit after reporting a fatal engine error");
    assert!(worker_error.to_string().contains("34359738368"));
}

#[tokio::test]
async fn should_cancel_and_remain_reusable_after_malformed_output() {
    let engine_worker =
        EngineBackedWorker::new(MalformedFinishProcessor, TrackingChatEngine::new());
    let (daemon_transport, worker_transport) = duplex(MAX_IPC_FRAME_BYTES * 2);
    let (daemon_reader_transport, daemon_writer_transport) = split(daemon_transport);
    let (worker_reader_transport, worker_writer_transport) = split(worker_transport);
    let mut daemon_reader = ProtocolReader::new(daemon_reader_transport);
    let mut daemon_writer = ProtocolWriter::new(daemon_writer_transport);
    let worker_task = tokio::spawn(async move {
        engine_worker
            .run(worker_reader_transport, worker_writer_transport)
            .await
    });

    assert!(matches!(
        next_event(&mut daemon_reader).await,
        WorkerEvent::Ready { .. }
    ));
    for request_number in [721_u64, 722_u64] {
        daemon_writer
            .send_command(&WorkerCommand::Generate(chat_command(request_number, 9)))
            .await
            .expect("the worker should receive the malformed-output request");
        let progress_event = next_event(&mut daemon_reader).await;
        match progress_event {
            WorkerEvent::GenerationProgress {
                request_id,
                generated_token_count,
                maximum_output_tokens,
                ..
            } => {
                assert_eq!(request_id, RequestId::new(request_number));
                assert_eq!(generated_token_count, 1);
                assert_eq!(maximum_output_tokens, 2);
            }
            other_event => {
                panic!("expected generation progress before malformed output, got {other_event:?}")
            }
        }
        assert_eq!(
            next_event(&mut daemon_reader).await,
            WorkerEvent::Failed {
                request_id: RequestId::new(request_number),
                reason: ChatGenerationFailureReason::MalformedModelOutput,
            }
        );
    }

    close_worker_transport(daemon_writer, worker_task).await;
}

#[tokio::test]
async fn should_ignore_a_stale_cancellation_after_request_completion() {
    let engine_worker =
        EngineBackedWorker::new(ScriptedChatProcessor::new(), ScriptedChatEngine::new());
    let (daemon_transport, worker_transport) = duplex(MAX_IPC_FRAME_BYTES * 2);
    let (daemon_reader_transport, daemon_writer_transport) = split(daemon_transport);
    let (worker_reader_transport, worker_writer_transport) = split(worker_transport);
    let mut daemon_reader = ProtocolReader::new(daemon_reader_transport);
    let mut daemon_writer = ProtocolWriter::new(daemon_writer_transport);
    let worker_task = tokio::spawn(async move {
        engine_worker
            .run(worker_reader_transport, worker_writer_transport)
            .await
    });

    assert_eq!(next_event(&mut daemon_reader).await, ready_event());
    daemon_writer
        .send_command(&WorkerCommand::Generate(chat_command(731, 10)))
        .await
        .expect("the worker should receive a chat request");
    for _expected_event_number in 0..3 {
        next_event(&mut daemon_reader).await;
    }
    daemon_writer
        .send_command(&WorkerCommand::Cancel {
            request_id: RequestId::new(731),
        })
        .await
        .expect("the worker should receive the stale cancellation");

    assert!(
        timeout(Duration::from_millis(100), daemon_reader.next_event())
            .await
            .is_err(),
        "an idle worker must not emit a duplicate cancellation completion"
    );

    close_worker_transport(daemon_writer, worker_task).await;
}

#[tokio::test]
async fn should_report_engine_cached_token_count_in_completion() {
    let engine_worker = EngineBackedWorker::new(
        ScriptedChatProcessor::new(),
        ScriptedChatEngine::with_cached_token_count(2_048),
    );
    let (daemon_transport, worker_transport) = duplex(MAX_IPC_FRAME_BYTES * 2);
    let (daemon_reader_transport, daemon_writer_transport) = split(daemon_transport);
    let (worker_reader_transport, worker_writer_transport) = split(worker_transport);
    let mut daemon_reader = ProtocolReader::new(daemon_reader_transport);
    let mut daemon_writer = ProtocolWriter::new(daemon_writer_transport);
    let worker_task = tokio::spawn(async move {
        engine_worker
            .run(worker_reader_transport, worker_writer_transport)
            .await
    });

    assert_eq!(next_event(&mut daemon_reader).await, ready_event());
    daemon_writer
        .send_command(&WorkerCommand::Generate(chat_command(741, 11)))
        .await
        .expect("the worker should receive a chat request");
    let _first_output_event = next_event(&mut daemon_reader).await;
    let _second_output_event = next_event(&mut daemon_reader).await;

    assert_eq!(
        next_event(&mut daemon_reader).await,
        WorkerEvent::Completed {
            persistent_prompt_cache_diagnostics: None,
            request_id: RequestId::new(741),
            prompt_token_count: 1,
            generated_token_count: 2,
            reasoning_token_count: 0,
            cached_token_count: 2_048,
            reason: ChatGenerationCompletionReason::ToolCalls,
        }
    );

    close_worker_transport(daemon_writer, worker_task).await;
}
