use std::io;

use super::error::{WorkerProcessError, WorkerStartupError};

#[test]
fn should_include_the_exact_io_cause_in_a_worker_startup_failure() {
    let worker_process_error =
        WorkerProcessError::Startup(WorkerStartupError::SamplePhysicalMemorySize(
            io::Error::other("synthetic sysctl spawn failure"),
        ));

    let worker_failure_diagnostic = worker_process_error.to_string();

    assert!(worker_failure_diagnostic.contains("failed to sample hw.memsize"));
    assert!(worker_failure_diagnostic.contains("synthetic sysctl spawn failure"));
}

#[test]
fn should_bound_a_chat_family_failure_reason_built_from_untrusted_metadata() {
    let untrusted_module_name = "untrusted_module_name_".repeat(64);
    let model_startup_error = model_serving::Qwen3_5ModelStartupError::ArtifactValidation {
        model_directory: std::path::PathBuf::from("/private/models/example-optiq-model"),
        source: model_serving::Qwen3_5ArtifactValidationError::OptiQMetadata(
            model_serving::OptiQMetadataError::UnsupportedBits {
                module_name: untrusted_module_name,
                actual_bits: 7,
            },
        ),
    };

    let public_failure_reason = super::error::bound_public_model_load_failure_reason(
        model_startup_error.public_model_load_failure_reason(),
    );

    assert!(
        public_failure_reason.chars().count() <= 512,
        "public model-load errors must remain bounded"
    );
    assert!(
        public_failure_reason.ends_with('…'),
        "a truncated public model-load error should identify truncation"
    );
}

#[test]
fn should_keep_a_short_chat_family_failure_reason_whole() {
    let public_failure_reason = super::error::bound_public_model_load_failure_reason(
        "model initialization failed".to_owned(),
    );
    assert_eq!(public_failure_reason, "model initialization failed");
}
