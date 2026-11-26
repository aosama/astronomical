//! Steps of the mid-swap load failure journey, kept apart from the journey
//! itself so the test reads as the order of events.

use ipc_protocol::{
    ProtocolReader, ProtocolWriter, RequestId, WorkerCommand, WorkerEvent,
    WorkerLoadedModelRuntimeConfiguration, WorkerRuntimeFeatureConfiguration,
};
use tokio::io::{AsyncRead, AsyncWrite};

use super::support::{
    ScriptedImageEngine, ScriptedRuntime, ScriptedRuntimeFactory, completed_step,
    image_configuration, next_event, swap_image_command,
};
use crate::engine_backed_worker::tests::chat::scripted_chat_test_doubles::{
    ScriptedChatEngine, ScriptedChatProcessor,
};
use crate::engine_backed_worker::tests::chat::support as chat_support;
use crate::{EngineBackedWorker, EnginePlacement};

/// The factory hands out a chat runtime, an image runtime, a load failure and a
/// second chat runtime, in the order the journey swaps models.
pub(super) fn idle_worker_scripted_to_fail_its_third_model_load() -> EngineBackedWorker<
    ScriptedChatProcessor,
    ScriptedChatEngine,
    ScriptedRuntimeFactory,
    ScriptedImageEngine,
> {
    let factory = ScriptedRuntimeFactory::new(vec![
        Ok(ScriptedRuntime::autoregressive(
            ScriptedChatProcessor::new(),
            ScriptedChatEngine::new(),
            EnginePlacement::CompleteResidencyFits,
        )),
        Ok(ScriptedRuntime::Image(ScriptedImageEngine::new(vec![
            vec![Ok(completed_step())],
        ]))),
        Err("the replacement artifact is invalid".to_owned()),
        Ok(ScriptedRuntime::autoregressive(
            ScriptedChatProcessor::new(),
            ScriptedChatEngine::new(),
            EnginePlacement::CompleteResidencyFits,
        )),
    ]);
    EngineBackedWorker::idle_with_model_factory(factory, 8_000)
        .with_worker_runtime_feature_configuration(WorkerRuntimeFeatureConfiguration {
            configuration_generation: "generation-192".to_owned(),
            astro_ssd_prompt_cache_enabled: false,
            astro_ssd_prompt_cache_maximum_size_bytes: 0,
            loaded_model: None,
        })
}

/// Selects the chat model, then the image model, checking after each swap that
/// the applied runtime configuration names the model now loaded.
pub(super) async fn load_chat_then_image_runtime<ReadTransport, WriteTransport>(
    events: &mut ProtocolReader<ReadTransport>,
    commands: &mut ProtocolWriter<WriteTransport>,
) where
    ReadTransport: AsyncRead + Unpin,
    WriteTransport: AsyncWrite + Unpin,
{
    commands
        .send_command(&WorkerCommand::SwapModel {
            model_directory: "/models/chat".to_owned(),
            model_configuration: chat_support::worker_model_configuration("example/scripted-chat"),
        })
        .await
        .expect("chat swap");
    let _chat_swapped = next_event(events).await;
    assert!(
        matches!(next_event(events).await, WorkerEvent::RuntimeFeatureConfigurationApplied { worker_runtime_feature_configuration } if matches!(worker_runtime_feature_configuration.loaded_model, Some(WorkerLoadedModelRuntimeConfiguration::Autoregressive(_))))
    );

    commands
        .send_command(&swap_image_command())
        .await
        .expect("image swap");
    let _image_swapped = next_event(events).await;
    assert!(
        matches!(next_event(events).await, WorkerEvent::RuntimeFeatureConfigurationApplied { worker_runtime_feature_configuration } if worker_runtime_feature_configuration.loaded_model == Some(image_configuration().runtime_configuration()))
    );
    let _image_model_load_memory = next_event(events).await;
}

/// Swaps a chat model back in after the failed swap and waits until a chat
/// request sent to it completes.
pub(super) async fn replace_with_chat_model_and_complete_a_request<ReadTransport, WriteTransport>(
    events: &mut ProtocolReader<ReadTransport>,
    commands: &mut ProtocolWriter<WriteTransport>,
) where
    ReadTransport: AsyncRead + Unpin,
    WriteTransport: AsyncWrite + Unpin,
{
    commands
        .send_command(&WorkerCommand::SwapModel {
            model_directory: "/models/chat-again".to_owned(),
            model_configuration: chat_support::worker_model_configuration("example/scripted-chat"),
        })
        .await
        .expect("chat replacement");
    let _chat_swapped_again = next_event(events).await;
    let _chat_configuration = next_event(events).await;
    commands
        .send_command(&WorkerCommand::Generate(chat_support::chat_command(
            931, 19,
        )))
        .await
        .expect("chat after image");
    while !matches!(next_event(events).await, WorkerEvent::Completed { request_id, .. } if request_id == RequestId::new(931))
    {
    }
}
