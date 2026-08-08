use astronomical_ipc_protocol::{
    ChatGenerationCommand, ChatGenerationFailureReason, ChatGenerationOutput, MtpRuntimeState,
    SpeculativePrefillRuntimeState, WorkerEvent,
};
use serde::Serialize;

use crate::PreparedInferenceRequest;

/// Model-specific request preparation and generated-token interpretation.
pub trait ModelGenerationProcessor {
    /// Prepared architecture-specific input consumed by the paired inference engine.
    type InferenceRequest: PreparedInferenceRequest + Send;
    /// Request-local state used to decode and interpret generated token IDs.
    type RequestOutput: Send;

    /// Reports the exact loaded model identity and output capabilities.
    fn ready_event(
        &self,
        mtp_runtime_state: MtpRuntimeState,
        mtp_unavailable_reason: Option<String>,
        speculative_prefill_runtime_state: SpeculativePrefillRuntimeState,
        speculative_prefill_unavailable_reason: Option<String>,
        speculative_prefill_draft_model_id: Option<String>,
        speculative_prefill_draft_model_revision: Option<String>,
    ) -> WorkerEvent;

    /// Prepares one independently validated structured-chat request.
    fn prepare_chat_generation(
        &self,
        chat_generation_command: &ChatGenerationCommand,
    ) -> Result<
        PreparedModelGeneration<Self::InferenceRequest, Self::RequestOutput>,
        ChatGenerationFailureReason,
    >;

    /// Returns whether one generated token is a model end-of-sequence marker.
    fn is_end_of_sequence_token(&self, generated_token_id: u32) -> bool;

    /// Translates one generated token through request-local output state.
    fn translate_generated_token(
        &self,
        request_output: &mut Self::RequestOutput,
        generated_token_id: u32,
    ) -> Result<ModelGeneratedTokenTranslation, ModelGenerationOutputError>;

    /// Flushes bounded state after generation stops.
    fn finish_request_output(
        &self,
        request_output: &mut Self::RequestOutput,
    ) -> Result<Vec<ChatGenerationOutput>, ModelGenerationOutputError>;
}

/// Public outputs plus optional tokenized feedback that must be injected back into the active model.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelGeneratedTokenTranslation {
    public_outputs: Vec<ChatGenerationOutput>,
    model_feedback_token_ids: Vec<u32>,
}

impl ModelGeneratedTokenTranslation {
    /// Creates a generated-token translation with client outputs and model-visible feedback tokens.
    #[must_use]
    pub fn new(
        public_outputs: Vec<ChatGenerationOutput>,
        model_feedback_token_ids: Vec<u32>,
    ) -> Self {
        Self {
            public_outputs,
            model_feedback_token_ids,
        }
    }

    /// Creates a generated-token translation with no model-visible feedback.
    #[must_use]
    pub fn from_outputs(public_outputs: Vec<ChatGenerationOutput>) -> Self {
        Self::new(public_outputs, Vec::new())
    }

    /// Splits the translation into public outputs and model-visible feedback tokens.
    #[must_use]
    pub fn into_parts(self) -> (Vec<ChatGenerationOutput>, Vec<u32>) {
        (self.public_outputs, self.model_feedback_token_ids)
    }
}

/// Failure while decoding or parsing generated model output.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ModelGenerationOutputError {
    /// Output did not satisfy the request's structured output contract.
    MalformedOutput {
        /// Private diagnostic payload for logs; not part of the public REST contract.
        diagnostic: Box<MalformedModelOutputDiagnostic>,
    },
    /// The model processor encountered a condition that invalidates worker reuse.
    Fatal { reason: String },
}

/// Private diagnostic payload emitted when model text fails the structured output contract.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct MalformedModelOutputDiagnostic {
    /// Stable parser failure code.
    pub diagnostic_code: &'static str,
    /// Human-readable parser failure.
    pub parser_error: String,
    /// Generated token IDs observed by the request-local decoder.
    pub generated_token_ids: Vec<u32>,
    /// Token IDs currently retained for byte-fallback decoding.
    pub pending_token_ids: Vec<u32>,
    /// Decoded model output text emitted by the tokenizer before the parser failed.
    pub decoded_output_text: String,
    /// Parser state at failure time.
    pub parser_state: &'static str,
    /// Parser-retained text at failure time.
    pub parser_pending_output_text: String,
}

/// Prepared inference request paired with its model-specific output state.
pub struct PreparedModelGeneration<InferenceRequest, RequestOutput> {
    pub(crate) inference_request: InferenceRequest,
    pub(crate) request_output: RequestOutput,
}

impl<InferenceRequest, RequestOutput> PreparedModelGeneration<InferenceRequest, RequestOutput> {
    /// Creates a prepared inference request and its request-local output owner.
    pub fn new(inference_request: InferenceRequest, request_output: RequestOutput) -> Self {
        Self {
            inference_request,
            request_output,
        }
    }
}
