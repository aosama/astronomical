use std::{io, path::PathBuf};

use model_serving::AstroSsdPromptCacheDiskStoreError;
use runtime_integration::MlxRuntimeError;

#[test]
fn should_report_the_exact_active_memory_deficit_as_retryable_publication_pressure() {
    let publication_error = AstroSsdPromptCacheDiskStoreError::SaveSafetensors {
        source: MlxRuntimeError::ActiveMemoryLimitExceeded {
            active_memory_bytes: 9_500,
            attempted_allocation_bytes: 1_500,
            allowed_active_memory_bytes: 10_000,
        },
    };

    assert_eq!(publication_error.active_memory_deficit_bytes(), Some(1_000));
}

#[test]
fn should_not_classify_a_descriptor_write_failure_as_retryable_memory_pressure() {
    let publication_error = AstroSsdPromptCacheDiskStoreError::WriteSafetensorsDescriptor {
        source: io::Error::other("fictional descriptor failure"),
    };

    assert_eq!(publication_error.active_memory_deficit_bytes(), None);
}

#[test]
fn should_not_classify_a_storage_quota_failure_as_retryable_memory_pressure() {
    let publication_error =
        AstroSsdPromptCacheDiskStoreError::GlobalAstroSsdPromptCacheQuotaNotSatisfied {
            maximum_size_bytes: 1_000,
            remaining_size_bytes: 200,
        };

    assert_eq!(publication_error.active_memory_deficit_bytes(), None);
}

#[test]
fn should_not_classify_a_filesystem_failure_as_retryable_memory_pressure() {
    let publication_error = AstroSsdPromptCacheDiskStoreError::OpenTempFile {
        temp_file_path: PathBuf::from(
            "fictional-astro-ssd-prompt-cache/staging/sequence.safetensors",
        ),
        source: io::Error::other("fictional filesystem failure"),
    };

    assert_eq!(publication_error.active_memory_deficit_bytes(), None);
}
