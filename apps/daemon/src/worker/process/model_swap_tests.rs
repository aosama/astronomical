//! Unit contracts for the model-swap acknowledgement staging steps of the
//! swap wait.

use astronomical_ipc_protocol::{
    ChatModelCapabilities, WorkerChunkingConfiguration, WorkerEvent,
    WorkerLoadedAutoregressiveModelRuntimeConfiguration, WorkerLoadedModelRuntimeConfiguration,
    WorkerModelCapabilities, WorkerRuntimeFeatureConfiguration,
};

use super::{
    ExpectedModelSwap, StagedModelSwapAcknowledgement, stage_model_swap_acknowledgement,
    stage_runtime_policy_acknowledgement,
};
use crate::WorkerControlError;

const SAMPLE_MODEL_ID: &str = "sample-model";
const SAMPLE_CONFIGURATION_GENERATION: &str = "generation-1";

fn sample_chunking_configuration() -> WorkerChunkingConfiguration {
    WorkerChunkingConfiguration {
        fixed_prompt_processing_chunk_size_tokens: 2_048,
        fixed_ssd_streaming_prompt_processing_chunk_size_tokens: 2_048,
        full_attention_key_value_growth_tokens: 256,
        prefill_graph_submission_layer_interval: 0,
        experimental_ssd_paging_prefill_graph_submission_layer_interval: 1,
        experimental_ssd_paging_generation_graph_submission_layer_interval: 3,
        prompt_cache_block_tokens: None,
        prompt_cache_common_prefix_stride_blocks: 4,
        experimental_decode_stage_attribution_enabled: false,
        experimental_prefill_stage_attribution_enabled: false,
        experimental_in_memory_prefix_adoption_enabled: false,
        experimental_quantized_kv_cache_enabled: false,
        experimental_fused_moe_decode_enabled: false,
    }
}

fn autoregressive_runtime_configuration(model_id: &str) -> WorkerLoadedModelRuntimeConfiguration {
    WorkerLoadedModelRuntimeConfiguration::Autoregressive(
        WorkerLoadedAutoregressiveModelRuntimeConfiguration {
            model_id: model_id.to_owned(),
            maximum_context_tokens: 65_536,
            maximum_output_tokens: 8_192,
            chunking: sample_chunking_configuration(),
        },
    )
}

/// Chat capabilities that satisfy the autoregressive capability policy:
/// context window and output ceiling must match the loaded policy and the
/// input ceiling must be exactly one position below the context window.
fn matching_autoregressive_capabilities() -> WorkerModelCapabilities {
    WorkerModelCapabilities {
        chat: Some(ChatModelCapabilities {
            supports_reasoning: false,
            supports_tool_calls: false,
            has_vision: false,
            max_input_tokens: 65_535,
            max_output_tokens: 8_192,
            context_window: 65_536,
        }),
        image_generation: None,
        embeddings: None,
    }
}

fn model_swapped_event(capabilities: WorkerModelCapabilities) -> WorkerEvent {
    model_swapped_event_with_identity(SAMPLE_MODEL_ID, capabilities)
}

fn model_swapped_event_with_identity(
    model_id: &str,
    capabilities: WorkerModelCapabilities,
) -> WorkerEvent {
    WorkerEvent::ModelSwapped {
        model_id: model_id.to_owned(),
        capabilities,
        expert_memory_mode: None,
        minimum_mlx_memory_ceiling_bytes: 1,
    }
}

fn runtime_policy_acknowledgement_event(
    configuration_generation: &str,
    loaded_model: Option<WorkerLoadedModelRuntimeConfiguration>,
) -> WorkerEvent {
    WorkerEvent::RuntimeFeatureConfigurationApplied {
        worker_runtime_feature_configuration: WorkerRuntimeFeatureConfiguration {
            configuration_generation: configuration_generation.to_owned(),
            persistent_prompt_cache_enabled: true,
            prompt_cache_maximum_size_bytes: 10_000_000_000,
            loaded_model,
        },
    }
}

fn expected_model_swap<'a>(
    configuration_generation: Option<&'a str>,
    model_runtime_configuration: &'a WorkerLoadedModelRuntimeConfiguration,
) -> ExpectedModelSwap<'a> {
    ExpectedModelSwap {
        configuration_generation,
        model_runtime_configuration,
    }
}

fn assert_protocol_violation(
    stage_outcome: Result<bool, WorkerControlError>,
    expected_description: &str,
) {
    match stage_outcome {
        Err(WorkerControlError::WorkerProtocolViolation { description }) => {
            assert_eq!(description, expected_description);
        }
        unexpected_outcome => panic!(
            "expected the protocol violation {expected_description:?}, got {unexpected_outcome:?}"
        ),
    }
}

#[test]
fn should_stage_the_first_model_swapped_acknowledgement() {
    let mut staged_model_swap = StagedModelSwapAcknowledgement::new();
    let expected_configuration = autoregressive_runtime_configuration(SAMPLE_MODEL_ID);
    let expected_swap = expected_model_swap(
        Some(SAMPLE_CONFIGURATION_GENERATION),
        &expected_configuration,
    );

    let stage_outcome = stage_model_swap_acknowledgement(
        &mut staged_model_swap,
        model_swapped_event(matching_autoregressive_capabilities()),
        &expected_swap,
    );

    assert!(stage_outcome.is_ok());
    assert!(staged_model_swap.model_swap_event.is_some());
    assert!(staged_model_swap.runtime_configuration.is_none());
}

#[test]
fn should_reject_a_duplicate_model_swapped_acknowledgement() {
    let mut staged_model_swap = StagedModelSwapAcknowledgement::new();
    let expected_configuration = autoregressive_runtime_configuration(SAMPLE_MODEL_ID);
    let expected_swap = expected_model_swap(None, &expected_configuration);
    let first_stage_outcome = stage_model_swap_acknowledgement(
        &mut staged_model_swap,
        model_swapped_event(matching_autoregressive_capabilities()),
        &expected_swap,
    );
    assert!(first_stage_outcome.is_ok());

    let second_stage_outcome = stage_model_swap_acknowledgement(
        &mut staged_model_swap,
        model_swapped_event(matching_autoregressive_capabilities()),
        &expected_swap,
    );

    assert!(second_stage_outcome.is_err());
}

#[test]
fn should_reject_a_model_swapped_event_with_a_mismatched_model_identity() {
    let mut staged_model_swap = StagedModelSwapAcknowledgement::new();
    let expected_configuration = autoregressive_runtime_configuration(SAMPLE_MODEL_ID);
    let expected_swap = expected_model_swap(None, &expected_configuration);
    let mismatched_identity_event = model_swapped_event_with_identity(
        "a-different-model",
        matching_autoregressive_capabilities(),
    );

    let stage_outcome = stage_model_swap_acknowledgement(
        &mut staged_model_swap,
        mismatched_identity_event,
        &expected_swap,
    );

    assert!(stage_outcome.is_err());
    assert!(staged_model_swap.model_swap_event.is_none());
}

#[test]
fn should_reject_a_model_swapped_event_without_the_expected_chat_capabilities() {
    let mut staged_model_swap = StagedModelSwapAcknowledgement::new();
    let expected_configuration = autoregressive_runtime_configuration(SAMPLE_MODEL_ID);
    let expected_swap = expected_model_swap(None, &expected_configuration);
    let capabilities_without_chat = WorkerModelCapabilities {
        chat: None,
        image_generation: None,
        embeddings: None,
    };

    let stage_outcome = stage_model_swap_acknowledgement(
        &mut staged_model_swap,
        model_swapped_event(capabilities_without_chat),
        &expected_swap,
    );

    assert!(stage_outcome.is_err());
}

#[test]
fn should_reject_chat_capabilities_missing_the_reserved_input_position() {
    let mut staged_model_swap = StagedModelSwapAcknowledgement::new();
    let expected_configuration = autoregressive_runtime_configuration(SAMPLE_MODEL_ID);
    let expected_swap = expected_model_swap(None, &expected_configuration);
    // The autoregressive policy reserves exactly one context position for
    // generation, so an input ceiling of the full context window is invalid.
    let mut off_by_one_capabilities = matching_autoregressive_capabilities();
    if let Some(chat_capabilities) = off_by_one_capabilities.chat.as_mut() {
        chat_capabilities.max_input_tokens = 65_536;
    }

    let stage_outcome = stage_model_swap_acknowledgement(
        &mut staged_model_swap,
        model_swapped_event(off_by_one_capabilities),
        &expected_swap,
    );

    assert!(stage_outcome.is_err());
}

#[test]
fn should_stage_a_matching_runtime_policy_acknowledgement() {
    let mut staged_model_swap = StagedModelSwapAcknowledgement::new();
    let expected_configuration = autoregressive_runtime_configuration(SAMPLE_MODEL_ID);
    let expected_swap = expected_model_swap(
        Some(SAMPLE_CONFIGURATION_GENERATION),
        &expected_configuration,
    );

    let stage_outcome = stage_runtime_policy_acknowledgement(
        &mut staged_model_swap,
        &runtime_policy_acknowledgement_event(
            SAMPLE_CONFIGURATION_GENERATION,
            Some(expected_configuration.clone()),
        ),
        &expected_swap,
    );

    assert!(stage_outcome.expect("the matching acknowledgement should stage"));
    assert!(staged_model_swap.runtime_configuration.is_some());
}

#[test]
fn should_leave_unrelated_worker_events_unstaged() {
    let mut staged_model_swap = StagedModelSwapAcknowledgement::new();
    let expected_configuration = autoregressive_runtime_configuration(SAMPLE_MODEL_ID);
    let expected_swap = expected_model_swap(
        Some(SAMPLE_CONFIGURATION_GENERATION),
        &expected_configuration,
    );
    let unrelated_worker_event = WorkerEvent::Idle {
        machine_mlx_memory_ceiling_bytes: 1,
        effective_mlx_memory_ceiling_bytes: 2,
        minimum_mlx_memory_ceiling_bytes: 1,
        wired_residency_budget_bytes: None,
    };

    let stage_outcome = stage_runtime_policy_acknowledgement(
        &mut staged_model_swap,
        &unrelated_worker_event,
        &expected_swap,
    );

    assert!(!stage_outcome.expect("unrelated events never violate the protocol"));
    assert!(staged_model_swap.runtime_configuration.is_none());
}

#[test]
fn should_reject_a_runtime_policy_acknowledgement_with_a_mismatched_generation() {
    let mut staged_model_swap = StagedModelSwapAcknowledgement::new();
    let expected_configuration = autoregressive_runtime_configuration(SAMPLE_MODEL_ID);
    let expected_swap = expected_model_swap(
        Some(SAMPLE_CONFIGURATION_GENERATION),
        &expected_configuration,
    );

    let stage_outcome = stage_runtime_policy_acknowledgement(
        &mut staged_model_swap,
        &runtime_policy_acknowledgement_event(
            "a-different-generation",
            Some(expected_configuration.clone()),
        ),
        &expected_swap,
    );

    assert_protocol_violation(
        stage_outcome,
        "model swap runtime policy acknowledgement mismatch",
    );
    assert!(staged_model_swap.runtime_configuration.is_none());
}

#[test]
fn should_reject_a_runtime_policy_acknowledgement_with_a_mismatched_loaded_model() {
    let mut staged_model_swap = StagedModelSwapAcknowledgement::new();
    let expected_configuration = autoregressive_runtime_configuration(SAMPLE_MODEL_ID);
    let expected_swap = expected_model_swap(
        Some(SAMPLE_CONFIGURATION_GENERATION),
        &expected_configuration,
    );

    let stage_outcome = stage_runtime_policy_acknowledgement(
        &mut staged_model_swap,
        &runtime_policy_acknowledgement_event(
            SAMPLE_CONFIGURATION_GENERATION,
            Some(autoregressive_runtime_configuration("a-different-model")),
        ),
        &expected_swap,
    );

    assert_protocol_violation(
        stage_outcome,
        "model swap runtime policy acknowledgement mismatch",
    );
}

#[test]
fn should_leave_runtime_policy_acknowledgements_unstaged_without_a_generation() {
    let mut staged_model_swap = StagedModelSwapAcknowledgement::new();
    let expected_configuration = autoregressive_runtime_configuration(SAMPLE_MODEL_ID);
    let expected_swap = expected_model_swap(None, &expected_configuration);

    let stage_outcome = stage_runtime_policy_acknowledgement(
        &mut staged_model_swap,
        &runtime_policy_acknowledgement_event(
            SAMPLE_CONFIGURATION_GENERATION,
            Some(expected_configuration.clone()),
        ),
        &expected_swap,
    );

    assert!(!stage_outcome.expect("no expected generation never violates the protocol"));
    assert!(staged_model_swap.runtime_configuration.is_none());
}

#[test]
fn should_reject_a_duplicate_runtime_policy_acknowledgement() {
    let mut staged_model_swap = StagedModelSwapAcknowledgement::new();
    let expected_configuration = autoregressive_runtime_configuration(SAMPLE_MODEL_ID);
    let expected_swap = expected_model_swap(
        Some(SAMPLE_CONFIGURATION_GENERATION),
        &expected_configuration,
    );
    let first_stage_outcome = stage_runtime_policy_acknowledgement(
        &mut staged_model_swap,
        &runtime_policy_acknowledgement_event(
            SAMPLE_CONFIGURATION_GENERATION,
            Some(expected_configuration.clone()),
        ),
        &expected_swap,
    );
    assert!(first_stage_outcome.expect("the first acknowledgement should stage"));

    let second_stage_outcome = stage_runtime_policy_acknowledgement(
        &mut staged_model_swap,
        &runtime_policy_acknowledgement_event(
            SAMPLE_CONFIGURATION_GENERATION,
            Some(expected_configuration.clone()),
        ),
        &expected_swap,
    );

    assert_protocol_violation(
        second_stage_outcome,
        "duplicate model swap runtime policy acknowledgement",
    );
}
