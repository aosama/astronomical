//! The identity of a model-load failure: a stable code, the stage it happened
//! in, a plain sentence and what the person can do.
//!
//! A load failure crosses from the worker to the daemon as a public sentence.
//! The sentences below are the ones the worker emits for its typed startup
//! failures, shared here as constants so the worker that writes them and the
//! daemon that recognises them cannot drift apart. A sentence this table does
//! not know is identified as `other`, never guessed at.

/// Public sentence prefixes the worker emits for typed Qwen3.5 startup
/// failures. Artifact validation appends its own detail after the prefix.
pub const ARTIFACT_VALIDATION_FAILED_PREFIX: &str = "Qwen3.5 artifact validation failed";
pub const MEMORY_GEOMETRY_FAILED_REASON: &str =
    "Qwen3.5 artifact RAM geometry could not be derived";
pub const RESIDENCY_ADMISSION_FAILED_REASON: &str = "Qwen3.5 complete-residency admission failed";
pub const PROCESSOR_INITIALIZATION_FAILED_REASON: &str = "Qwen3.5 processor initialization failed";
pub const QWEN_ENGINE_INITIALIZATION_FAILED_REASON: &str = "Qwen3.5 engine initialization failed";
/// What the worker says when a loaded engine fails to start, for any family.
pub const ENGINE_INITIALIZATION_FAILED_REASON: &str = "model engine initialization failed";

const DAMAGED_FILES_SENTENCE: &str = "This model's files are damaged or incomplete.";
const DAMAGED_FILES_NEXT_STEP: &str = "Remove the model and download it again.";
const MEMORY_SENTENCE: &str = "This model didn't fit in the memory Astronomical may use.";
const MEMORY_NEXT_STEP: &str = "Raise the memory limit in Settings, or pick a smaller model.";
const ENGINE_SENTENCE: &str = "The engine couldn't start this model.";
const TRY_AGAIN_NEXT_STEP: &str = "Try again. If it keeps happening, send a report.";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ModelLoadFailureIdentity {
    /// The stable name the developers group reports by.
    pub code: &'static str,
    /// Where in loading it happened, in plain words.
    pub stage: &'static str,
    /// What happened, for the person.
    pub plain_sentence: &'static str,
    /// What the person can do about it.
    pub next_step: &'static str,
}

const fn identity(
    code: &'static str,
    stage: &'static str,
    plain_sentence: &'static str,
    next_step: &'static str,
) -> ModelLoadFailureIdentity {
    ModelLoadFailureIdentity {
        code,
        stage,
        plain_sentence,
        next_step,
    }
}

/// Known failures, matched by the start of the worker's public sentence.
const KNOWN_FAILURES: [(&str, ModelLoadFailureIdentity); 6] = [
    (
        ARTIFACT_VALIDATION_FAILED_PREFIX,
        identity(
            "model_files_invalid",
            "checking the model's files",
            DAMAGED_FILES_SENTENCE,
            DAMAGED_FILES_NEXT_STEP,
        ),
    ),
    (
        MEMORY_GEOMETRY_FAILED_REASON,
        identity(
            "model_size_unreadable",
            "measuring the model",
            DAMAGED_FILES_SENTENCE,
            DAMAGED_FILES_NEXT_STEP,
        ),
    ),
    (
        RESIDENCY_ADMISSION_FAILED_REASON,
        identity(
            "memory_admission_failed",
            "fitting the model in memory",
            MEMORY_SENTENCE,
            MEMORY_NEXT_STEP,
        ),
    ),
    (
        PROCESSOR_INITIALIZATION_FAILED_REASON,
        identity(
            "tokenizer_invalid",
            "reading the tokenizer",
            DAMAGED_FILES_SENTENCE,
            DAMAGED_FILES_NEXT_STEP,
        ),
    ),
    (
        QWEN_ENGINE_INITIALIZATION_FAILED_REASON,
        identity(
            "engine_start_failed",
            "starting the engine",
            ENGINE_SENTENCE,
            TRY_AGAIN_NEXT_STEP,
        ),
    ),
    (
        ENGINE_INITIALIZATION_FAILED_REASON,
        identity(
            "engine_start_failed",
            "starting the engine",
            ENGINE_SENTENCE,
            TRY_AGAIN_NEXT_STEP,
        ),
    ),
];

const NOT_ENOUGH_MEMORY: ModelLoadFailureIdentity = identity(
    "not_enough_memory",
    "fitting the model in memory",
    MEMORY_SENTENCE,
    MEMORY_NEXT_STEP,
);

const OTHER_FAILURE: ModelLoadFailureIdentity = identity(
    "other",
    "loading the model",
    "This model couldn't be loaded.",
    TRY_AGAIN_NEXT_STEP,
);

/// Identifies a load failure from the worker's public sentence. Only the
/// identity leaves this function; the sentence itself is never reported.
#[must_use]
pub fn identify_model_load_failure(model_load_failure_reason: &str) -> ModelLoadFailureIdentity {
    for (known_sentence_start, known_identity) in KNOWN_FAILURES {
        if model_load_failure_reason.starts_with(known_sentence_start) {
            return known_identity;
        }
    }
    if model_load_failure_reason
        .to_ascii_lowercase()
        .contains("memory")
    {
        return NOT_ENOUGH_MEMORY;
    }
    OTHER_FAILURE
}

#[cfg(test)]
mod model_load_failure_identity_tests {
    use super::*;

    #[test]
    fn should_identify_each_typed_worker_failure_by_its_public_sentence() {
        let artifact_failure = identify_model_load_failure(
            "Qwen3.5 artifact validation failed: invalid safetensors header JSON in weight file model.safetensors",
        );
        assert_eq!(artifact_failure.code, "model_files_invalid");
        assert_eq!(artifact_failure.stage, "checking the model's files");
        assert_eq!(
            artifact_failure.next_step,
            "Remove the model and download it again."
        );
        assert_eq!(
            identify_model_load_failure(RESIDENCY_ADMISSION_FAILED_REASON).code,
            "memory_admission_failed"
        );
        assert_eq!(
            identify_model_load_failure(PROCESSOR_INITIALIZATION_FAILED_REASON).code,
            "tokenizer_invalid"
        );
        assert_eq!(
            identify_model_load_failure(ENGINE_INITIALIZATION_FAILED_REASON).code,
            "engine_start_failed"
        );
        assert_eq!(
            identify_model_load_failure(QWEN_ENGINE_INITIALIZATION_FAILED_REASON).stage,
            "starting the engine"
        );
    }

    #[test]
    fn should_fall_back_to_memory_then_other_for_sentences_it_does_not_know() {
        assert_eq!(
            identify_model_load_failure("not enough Memory for the weights").code,
            "not_enough_memory"
        );
        assert_eq!(
            identify_model_load_failure("unsupported architecture").code,
            "other"
        );
        assert_eq!(identify_model_load_failure("").code, "other");
    }

    #[test]
    fn should_give_every_known_failure_a_sentence_and_a_next_step() {
        for (_, known_identity) in KNOWN_FAILURES {
            assert!(!known_identity.plain_sentence.is_empty());
            assert!(!known_identity.next_step.is_empty());
        }
    }
}
