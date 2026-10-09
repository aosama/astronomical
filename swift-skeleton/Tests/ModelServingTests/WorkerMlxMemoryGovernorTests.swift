import Foundation;

import Testing;

import IpcProtocol;
import ModelServing;

/**
 * The worker-side wired-memory governor journeys: the composed budget
 * arithmetic, the context-admission verdicts with their named deficits,
 * and the structured ownership decomposition the status wire reports.
 * Pure arithmetic over recorded inputs; no MLX state.
 */
@Suite(.tags(.hermeticJourney))
final class WorkerMlxMemoryGovernorTests {

    @Test
    func should_compose_the_budget_from_the_ceiling_and_named_reserves() throws {
        let memoryGovernor: WorkerMlxMemoryGovernor = WorkerMlxMemoryGovernor(
            effectiveMlxMemoryCeilingBytes: 10_000,
            activationHeadroomBytes: 1_000,
            otherFixedBytes: 500);
        let composedBudget: MlxRamBudgetSnapshot = memoryGovernor.composedBudget(
            modelCorePayloadBytes: 4_000,
            contextWindowReserveBytes: 1_500,
            completeLayerStreamSlotBytes: 300);
        #expect(composedBudget.mlxActiveMemoryCeilingBytes == 10_000);
        #expect(composedBudget.modelCorePayloadBytes == 4_000);
        #expect(composedBudget.contextWindowReserveBytes == 1_500);
        #expect(composedBudget.activationHeadroomBytes == 1_000);
        #expect(composedBudget.completeLayerStreamSlotBytes == 300);
        #expect(composedBudget.otherFixedBytes == 500);
        #expect(composedBudget.retainedExpertBudgetBytes == 2_700);
    }

    @Test
    func should_saturate_the_leftover_budget_at_zero_when_charges_exceed_the_ceiling() throws {
        let memoryGovernor: WorkerMlxMemoryGovernor = WorkerMlxMemoryGovernor(
            effectiveMlxMemoryCeilingBytes: 1_000,
            activationHeadroomBytes: 900,
            otherFixedBytes: 0);
        let composedBudget: MlxRamBudgetSnapshot = memoryGovernor.composedBudget(
            modelCorePayloadBytes: 800,
            contextWindowReserveBytes: 500,
            completeLayerStreamSlotBytes: 0);
        #expect(composedBudget.retainedExpertBudgetBytes == 0);
    }

    @Test
    func should_admit_a_context_that_fits_the_budget() throws {
        let memoryGovernor: WorkerMlxMemoryGovernor = WorkerMlxMemoryGovernor(
            effectiveMlxMemoryCeilingBytes: 10_000,
            activationHeadroomBytes: 1_000);
        let admissionVerdict: WorkerMlxMemoryAdmissionVerdict = memoryGovernor
            .validateContextAdmission(
                modelCorePayloadBytes: 4_000, contextWindowReserveBytes: 4_000);
        guard case let .admitted(admittedBudget) = admissionVerdict else {
            Issue.record("expected an admission, got \(admissionVerdict)");
            return;
        }
        #expect(admittedBudget.retainedExpertBudgetBytes == 1_000);
    }

    @Test
    func should_reject_a_context_with_the_named_deficit() throws {
        let memoryGovernor: WorkerMlxMemoryGovernor = WorkerMlxMemoryGovernor(
            effectiveMlxMemoryCeilingBytes: 5_000,
            activationHeadroomBytes: 1_000);
        let admissionVerdict: WorkerMlxMemoryAdmissionVerdict = memoryGovernor
            .validateContextAdmission(
                modelCorePayloadBytes: 3_000, contextWindowReserveBytes: 4_000);
        #expect(admissionVerdict == .rejected(deficitBytes: 3_000));
    }
}
