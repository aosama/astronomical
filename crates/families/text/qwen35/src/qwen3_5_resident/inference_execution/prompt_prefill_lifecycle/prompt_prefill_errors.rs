//! The resident engine's prompt-reading errors: the shared ones, where the
//! memory governor's refusal to admit a chunk carries its typed rejection.

use crate::ResidentForwardAdmissionRejection;

use super::forward_memory::ForwardAdmissionError;

pub(super) use text_generation::generation_loop::prompt_prefill_errors::prefill_execution_error;

pub(super) type PromptPrefillChunkAttemptError =
    text_generation::generation_loop::prompt_prefill_errors::PromptPrefillChunkAttemptError<
        ResidentForwardAdmissionRejection,
    >;

impl From<ForwardAdmissionError> for PromptPrefillChunkAttemptError {
    fn from(admission_error: ForwardAdmissionError) -> Self {
        match admission_error {
            ForwardAdmissionError::Rejected { rejection, reason } => {
                Self::ForwardAdmissionRejected { rejection, reason }
            }
            ForwardAdmissionError::Engine(inference_engine_error) => {
                Self::Engine(inference_engine_error)
            }
        }
    }
}
