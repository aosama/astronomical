import Foundation

import Testing

import ModelServing

import RuntimeIntegration

/// Capacity-pressure classification parity across the direct MLX and
/// paged expert channels, port of
/// crates/model-serving/tests/hermetic/qwen3_5_execution_error.rs.
@Suite
final class Qwen35ExecutionErrorTests {

    @Test
    func should_classify_direct_and_paged_native_capacity_errors_as_the_same_recoverable_pressure()
    {
        let expectedCapacityEvidence = Qwen35ActiveMemoryLimitEvidence(
            activeMemoryBytes: 9_900_000_000,
            attemptedAllocationBytes: 300_000_000,
            allowedActiveMemoryBytes: 10_100_000_000)
        let directCapacityError = Qwen35ExecutionError.runtime(
            .activeMemoryLimitExceeded(
                activeMemoryBytes: expectedCapacityEvidence.activeMemoryBytes,
                attemptedAllocationBytes: expectedCapacityEvidence.attemptedAllocationBytes,
                allowedActiveMemoryBytes: expectedCapacityEvidence.allowedActiveMemoryBytes))
        let pagedCapacityError = Qwen35ExecutionError.expertPaging(
            .nativeRuntime(.activeMemoryLimitExceeded(
                activeMemoryBytes: expectedCapacityEvidence.activeMemoryBytes,
                attemptedAllocationBytes: expectedCapacityEvidence.attemptedAllocationBytes,
                allowedActiveMemoryBytes: expectedCapacityEvidence.allowedActiveMemoryBytes)))

        #expect(
            directCapacityError.activeMemoryLimitExceededEvidence() == expectedCapacityEvidence)
        #expect(
            pagedCapacityError.activeMemoryLimitExceededEvidence() == expectedCapacityEvidence,
            "paged expert loading must enter the same checkpoint, reclamation, and chunk-reduction recovery path as direct MLX execution")
    }

    @Test
    func should_classify_expert_budget_rejection_as_recoverable_capacity_pressure() {
        let expectedCapacityEvidence = Qwen35ActiveMemoryLimitEvidence(
            activeMemoryBytes: 29_151_197_454,
            attemptedAllocationBytes: 855_638_016,
            allowedActiveMemoryBytes: 30_000_000_000)
        let pagedBudgetError = Qwen35ExecutionError.expertPaging(
            .memoryBudget(.rejected(
                stage: "rust_streamed_expert_layer_39",
                boundary: .allocationProjection,
                shortfallBytes: 6_835_470,
                activeMemoryBytes: UInt64(expectedCapacityEvidence.activeMemoryBytes),
                pendingAllocationBytes: UInt64(expectedCapacityEvidence.attemptedAllocationBytes),
                activeMemoryCeilingBytes: UInt64(
                    expectedCapacityEvidence.allowedActiveMemoryBytes))))

        #expect(
            pagedBudgetError.activeMemoryLimitExceededEvidence() == expectedCapacityEvidence,
            "expert-budget rejection must enter checkpoint restoration and reclamation")
    }
}
