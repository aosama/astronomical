use super::*;

pub(super) fn chat_command(request_number: u64, seed: u64) -> ChatGenerationCommand {
    ChatGenerationCommand {
        request_id: RequestId::new(request_number),
        model: "example/scripted-chat".to_owned(),
        messages: vec![ChatMessage::User {
            content: "Inspect the repository.".to_owned(),
            images: Vec::new(),
        }],
        tools: vec![ChatToolDefinition {
            name: "glob".to_owned(),
            description: None,
            parameters_json: r#"{"type":"object"}"#.to_owned(),
        }],
        tool_choice: ChatToolChoice::Auto,
        settings: ChatGenerationSettings {
            max_output_tokens: 2,
            temperature_thousandths: Some(600),
            top_p_thousandths: Some(950),
            seed: Some(seed),
            thinking_budget: None,
        },
    }
}

pub(super) fn ready_event() -> WorkerEvent {
    ready_event_with_load_details(MtpRuntimeState::Disabled, None)
}

pub(super) fn ready_event_with_load_details(
    mtp_runtime_state: MtpRuntimeState,
    mtp_unavailable_reason: Option<String>,
) -> WorkerEvent {
    ready_event_with_speculative_prefill_load_details(
        mtp_runtime_state,
        mtp_unavailable_reason,
        SpeculativePrefillRuntimeState::Disabled,
        None,
        None,
        None,
    )
}

pub(super) fn ready_event_with_speculative_prefill_load_details(
    mtp_runtime_state: MtpRuntimeState,
    mtp_unavailable_reason: Option<String>,
    speculative_prefill_runtime_state: SpeculativePrefillRuntimeState,
    speculative_prefill_unavailable_reason: Option<String>,
    speculative_prefill_draft_model_id: Option<String>,
    speculative_prefill_draft_model_revision: Option<String>,
) -> WorkerEvent {
    WorkerEvent::Ready {
        model_id: "example/scripted-chat".to_owned(),
        capabilities: ChatModelCapabilities {
            supports_reasoning: true,
            supports_tool_calls: true,
            has_vision: true,
            max_input_tokens: 241_664,
            max_output_tokens: 20_480,
            context_window: 262_144,
        },
        mtp_runtime_state,
        mtp_unavailable_reason,
        speculative_prefill_runtime_state,
        speculative_prefill_unavailable_reason,
        speculative_prefill_draft_model_id,
        speculative_prefill_draft_model_revision,
    }
}

pub(super) async fn next_event<ReadTransport>(
    supervisor_reader: &mut ProtocolReader<ReadTransport>,
) -> WorkerEvent
where
    ReadTransport: tokio::io::AsyncRead + Unpin,
{
    supervisor_reader
        .next_event()
        .await
        .expect("the worker should write a valid event")
        .expect("the worker transport should remain open")
}

pub(super) async fn close_worker_transport<WriteTransport>(
    supervisor_writer: ProtocolWriter<WriteTransport>,
    worker_task: JoinHandle<Result<(), WorkerRuntimeError>>,
) where
    WriteTransport: AsyncWrite + Unpin,
{
    supervisor_writer
        .close()
        .await
        .expect("the supervisor should close the worker transport");
    assert!(
        timeout(Duration::from_secs(1), worker_task)
            .await
            .expect("the worker should stop after transport closure")
            .expect("the worker task should not panic")
            .is_ok()
    );
}
