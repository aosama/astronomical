use std::time::{SystemTime, UNIX_EPOCH};

use crate::application::ApplicationState;
use astronomical_config::{DiscoveredModel, ModelCapabilities, resolve_model_id};
use astronomical_ipc_protocol::WorkerModelCapabilities;
use astronomical_rest_contract::{
    OpenAiErrorResponse, OpenAiImageModelParts, OpenAiModel, OpenAiModelList, OpenAiModelParts,
    OpenAiModelValidationError,
};
use axum::{
    Json,
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Response},
};

const ASTRONOMICAL_MODEL_OWNER: &str = "astronomical";
const TEXT_INPUT_MODALITY: &str = "text";
const IMAGE_INPUT_MODALITY: &str = "image";
const TEXT_OUTPUT_MODALITY: &str = "text";
const IMAGE_OUTPUT_MODALITY: &str = "image";
const OPENAI_CHAT_REASONING_FORMAT: &str =
    "openai_chat_reasoning_content_and_responses_reasoning_summary_text";
const OPENAI_FUNCTION_CALL_FORMAT: &str = "openai_function_call";
const CHAT_COMPLETIONS_ENDPOINT_PATH: &str = "/v1/chat/completions";
const RESPONSES_ENDPOINT_PATH: &str = "/v1/responses";
const IMAGE_GENERATIONS_ENDPOINT_PATH: &str = "/v1/images/generations";

/// Lists every model the supervisor can currently advertise safely.
pub(crate) async fn list_models(State(application_state): State<ApplicationState>) -> Response {
    let model_advertisement_created_at_unix_seconds = current_unix_seconds();
    match advertised_models_for_application_state(
        &application_state,
        model_advertisement_created_at_unix_seconds,
    ) {
        Ok(advertised_models) => {
            Json(OpenAiModelList::from_models(advertised_models)).into_response()
        }
        Err(model_validation_error) => model_metadata_unavailable_response(model_validation_error),
    }
}

/// Returns one advertised model, resolving an optional provider prefix in its ID.
pub(crate) async fn retrieve_model(
    Path(requested_model_id): Path<String>,
    State(application_state): State<ApplicationState>,
) -> Response {
    let model_advertisement_created_at_unix_seconds = current_unix_seconds();
    let advertised_models = match advertised_models_for_application_state(
        &application_state,
        model_advertisement_created_at_unix_seconds,
    ) {
        Ok(advertised_models) => advertised_models,
        Err(model_validation_error) => {
            return model_metadata_unavailable_response(model_validation_error);
        }
    };
    let known_model_ids = advertised_models
        .iter()
        .map(OpenAiModel::id)
        .collect::<Vec<_>>();
    let resolved_model_id = resolve_model_id(&requested_model_id, &known_model_ids);

    match advertised_models
        .into_iter()
        .find(|advertised_model| advertised_model.id() == resolved_model_id)
    {
        Some(advertised_model) => Json(advertised_model).into_response(),
        None => model_not_found_response(),
    }
}

fn advertised_models_for_application_state(
    application_state: &ApplicationState,
    model_advertisement_created_at_unix_seconds: u64,
) -> Result<Vec<OpenAiModel>, OpenAiModelValidationError> {
    let discovered_models = application_state.discovered_models_snapshot();
    if discovered_models.is_empty() {
        let worker_health_snapshot = application_state
            .generation_executor
            .worker_health_snapshot();
        return match (
            worker_health_snapshot.status.is_ready(),
            worker_health_snapshot.ready_model_id,
            worker_health_snapshot.ready_model_capabilities,
        ) {
            (true, Some(ready_model_id), Some(ready_model_capabilities)) => {
                Ok(vec![openai_model_from_ready_worker_capabilities(
                    ready_model_id,
                    model_advertisement_created_at_unix_seconds,
                    ready_model_capabilities,
                )?])
            }
            _ => Ok(Vec::new()),
        };
    }

    discovered_models
        .iter()
        .map(|discovered_model| {
            openai_model_from_discovered_model(
                discovered_model,
                model_advertisement_created_at_unix_seconds,
            )
        })
        .collect()
}

fn openai_model_from_ready_worker_capabilities(
    ready_model_id: String,
    model_advertisement_created_at_unix_seconds: u64,
    ready_model_capabilities: WorkerModelCapabilities,
) -> Result<OpenAiModel, OpenAiModelValidationError> {
    let WorkerModelCapabilities {
        chat,
        image_generation,
    } = ready_model_capabilities;
    let Some(chat_capabilities) = chat else {
        return OpenAiModel::from_image_parts(OpenAiImageModelParts {
            model_id: ready_model_id,
            created: model_advertisement_created_at_unix_seconds,
            owned_by: ASTRONOMICAL_MODEL_OWNER.to_owned(),
            input_modalities: vec![TEXT_INPUT_MODALITY.to_owned()],
            output_modalities: vec![IMAGE_OUTPUT_MODALITY.to_owned()],
            supported_endpoints: image_generation
                .map(|_| vec![IMAGE_GENERATIONS_ENDPOINT_PATH.to_owned()])
                .unwrap_or_default(),
        });
    };
    OpenAiModel::from_parts(OpenAiModelParts {
        model_id: ready_model_id,
        created: model_advertisement_created_at_unix_seconds,
        owned_by: ASTRONOMICAL_MODEL_OWNER.to_owned(),
        context_window: chat_capabilities.context_window,
        max_input_tokens: chat_capabilities.max_input_tokens,
        max_output_tokens: chat_capabilities.max_output_tokens,
        input_modalities: input_modalities_for_model(chat_capabilities.has_vision),
        output_modalities: vec![TEXT_OUTPUT_MODALITY.to_owned()],
        supports_streaming: true,
        supports_reasoning: chat_capabilities.supports_reasoning,
        reasoning_format: reasoning_format_for_model(chat_capabilities.supports_reasoning),
        supports_tool_calls: chat_capabilities.supports_tool_calls,
        tool_call_format: tool_call_format_for_model(chat_capabilities.supports_tool_calls),
        supported_endpoints: supported_generation_endpoint_paths(),
    })
}

fn openai_model_from_discovered_model(
    discovered_model: &DiscoveredModel,
    model_advertisement_created_at_unix_seconds: u64,
) -> Result<OpenAiModel, OpenAiModelValidationError> {
    // Family discovery owns behavior; this layer owns only the common API shape.
    match &discovered_model.capabilities {
        ModelCapabilities::Chat(capabilities) => openai_model_from_discovered_chat_model(
            discovered_model,
            capabilities,
            model_advertisement_created_at_unix_seconds,
        ),
        ModelCapabilities::ImageGeneration(capabilities) => {
            OpenAiModel::from_image_parts(OpenAiImageModelParts {
                model_id: discovered_model.model_id.clone(),
                created: model_advertisement_created_at_unix_seconds,
                owned_by: ASTRONOMICAL_MODEL_OWNER.to_owned(),
                input_modalities: vec![TEXT_INPUT_MODALITY.to_owned()],
                output_modalities: vec![IMAGE_OUTPUT_MODALITY.to_owned()],
                supported_endpoints: capabilities
                    .supports_text_to_image
                    .then(|| vec![IMAGE_GENERATIONS_ENDPOINT_PATH.to_owned()])
                    .unwrap_or_default(),
            })
        }
    }
}

fn openai_model_from_discovered_chat_model(
    discovered_model: &DiscoveredModel,
    capabilities: &astronomical_config::ChatModelCapabilities,
    model_advertisement_created_at_unix_seconds: u64,
) -> Result<OpenAiModel, OpenAiModelValidationError> {
    OpenAiModel::from_parts(OpenAiModelParts {
        model_id: discovered_model.model_id.clone(),
        created: model_advertisement_created_at_unix_seconds,
        owned_by: ASTRONOMICAL_MODEL_OWNER.to_owned(),
        context_window: capabilities.context_window,
        max_input_tokens: capabilities.max_input_tokens,
        max_output_tokens: capabilities.max_output_tokens,
        input_modalities: input_modalities_for_model(capabilities.supports_vision),
        output_modalities: vec![TEXT_OUTPUT_MODALITY.to_owned()],
        supports_streaming: true,
        supports_reasoning: capabilities.supports_reasoning,
        reasoning_format: reasoning_format_for_model(capabilities.supports_reasoning),
        supports_tool_calls: capabilities.supports_tool_calls,
        tool_call_format: tool_call_format_for_model(capabilities.supports_tool_calls),
        supported_endpoints: supported_generation_endpoint_paths(),
    })
}

fn input_modalities_for_model(has_vision: bool) -> Vec<String> {
    let mut input_modalities = vec![TEXT_INPUT_MODALITY.to_owned()];
    if has_vision {
        input_modalities.push(IMAGE_INPUT_MODALITY.to_owned());
    }
    input_modalities
}

fn reasoning_format_for_model(supports_reasoning: bool) -> Option<String> {
    supports_reasoning.then(|| OPENAI_CHAT_REASONING_FORMAT.to_owned())
}

fn tool_call_format_for_model(supports_tool_calls: bool) -> Option<String> {
    supports_tool_calls.then(|| OPENAI_FUNCTION_CALL_FORMAT.to_owned())
}

fn supported_generation_endpoint_paths() -> Vec<String> {
    vec![
        CHAT_COMPLETIONS_ENDPOINT_PATH.to_owned(),
        RESPONSES_ENDPOINT_PATH.to_owned(),
    ]
}

fn current_unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration_since_epoch| duration_since_epoch.as_secs())
        .unwrap_or_default()
}

fn model_metadata_unavailable_response(
    model_validation_error: OpenAiModelValidationError,
) -> Response {
    tracing::error!(
        error = %model_validation_error,
        "refusing to advertise internally inconsistent model metadata"
    );
    (
        StatusCode::SERVICE_UNAVAILABLE,
        Json(OpenAiErrorResponse::service_unavailable(
            "model capability metadata is unavailable",
            Some("model_metadata_unavailable"),
        )),
    )
        .into_response()
}

fn model_not_found_response() -> Response {
    (
        StatusCode::NOT_FOUND,
        Json(OpenAiErrorResponse::invalid_request(
            "model is not loaded by the local worker",
            Some("model"),
            Some("model_not_found"),
        )),
    )
        .into_response()
}
