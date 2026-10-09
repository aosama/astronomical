import Foundation;

import Testing;

import RuntimeIntegration;

import ModelServing;

@testable import ModelServing;

/// Retry-classification journeys for publication failures: only MLX
/// active-memory pressure is retryable as memory pressure, and its deficit
/// reports the exact bytes a retry must release; descriptor, storage
/// quota, and filesystem failures never are.
final class PersistentPromptCachePublicationRetryTests {

    @Test
    func should_report_the_exact_active_memory_deficit_as_retryable_publication_pressure() {
        let publicationError: PersistentPromptCacheDiskStoreError =
            .saveSafetensors(source: .activeMemoryLimitExceeded(
                activeMemoryBytes: 9_500,
                attemptedAllocationBytes: 1_500,
                allowedActiveMemoryBytes: 10_000));

        #expect(publicationError.activeMemoryDeficitBytes() == 1_000);
    }

    @Test
    func should_not_classify_a_descriptor_write_failure_as_retryable_memory_pressure() {
        let publicationError: PersistentPromptCacheDiskStoreError =
            .writeSafetensorsDescriptor(
                filePath: "fictional-cache/staging/sequence.safetensors",
                problem: "fictional descriptor failure");

        #expect(publicationError.activeMemoryDeficitBytes() == nil);
    }

    @Test
    func should_not_classify_a_storage_quota_failure_as_retryable_memory_pressure() {
        let publicationError: PersistentPromptCacheDiskStoreError =
            .globalPromptCacheQuotaNotSatisfied(
                maximumSizeBytes: 1_000, remainingSizeBytes: 200);

        #expect(publicationError.activeMemoryDeficitBytes() == nil);
    }

    @Test
    func should_not_classify_a_filesystem_failure_as_retryable_memory_pressure() {
        let publicationError: PersistentPromptCacheDiskStoreError = .openTempFile(
            tempFilePath: "fictional-cache/staging/sequence.safetensors",
            problem: "fictional filesystem failure");

        #expect(publicationError.activeMemoryDeficitBytes() == nil);
    }
}
