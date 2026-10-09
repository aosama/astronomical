import Foundation;

import IpcProtocol;


/// The worker-side wired-memory governor, port of the Rust memory
/// control-surface integration (#983 E5): it composes the wired budget
/// from the effective ceiling and the named reserves using the ported
/// `MlxRamBudgetSnapshot` arithmetic, validates one request's context
/// admission against that budget before the engine starts, and derives the
/// ceiling utilization the status wire reports.
///
/// The context workspace is charged from the attached persistent
/// prompt-cache contract's per-token sequence-state bytes when a cache is
/// attached; engines without a contract report no context workspace and
/// are never rejected by this surface.
public struct WorkerMlxMemoryGovernor: Sendable {

    /// The effective live MLX active-memory ceiling.
    public let effectiveMlxMemoryCeilingBytes: UInt64;

    /// Reserved bytes for temporary activations and transient workspaces.
    public let activationHeadroomBytes: UInt64;

    /// Any additional fixed non-expert owners charged before the budget.
    public let otherFixedBytes: UInt64;

    public init(
        effectiveMlxMemoryCeilingBytes: UInt64,
        activationHeadroomBytes: UInt64 = 0,
        otherFixedBytes: UInt64 = 0
    ) {
        self.effectiveMlxMemoryCeilingBytes = effectiveMlxMemoryCeilingBytes;
        self.activationHeadroomBytes = activationHeadroomBytes;
        self.otherFixedBytes = otherFixedBytes;
    }

    /// Composes the wired budget for one observed engine state: the model
    /// core payload comes from the live snapshot, the context reserve from
    /// the request's context workspace charge, and the leftover is the
    /// retained-expert entitlement.
    public func composedBudget(
        modelCorePayloadBytes: UInt64,
        contextWindowReserveBytes: UInt64,
        completeLayerStreamSlotBytes: UInt64
    ) -> MlxRamBudgetSnapshot {
        let chargedBytes: UInt64 = modelCorePayloadBytes
            .addingReportingOverflow(contextWindowReserveBytes).partialValue
            .addingReportingOverflow(self.activationHeadroomBytes).partialValue
            .addingReportingOverflow(completeLayerStreamSlotBytes).partialValue
            .addingReportingOverflow(self.otherFixedBytes).partialValue;
        let retainedExpertBudgetBytes: UInt64;
        if chargedBytes >= self.effectiveMlxMemoryCeilingBytes {
            retainedExpertBudgetBytes = 0;
        } else {
            retainedExpertBudgetBytes = self.effectiveMlxMemoryCeilingBytes - chargedBytes;
        }
        return MlxRamBudgetSnapshot(
            mlxActiveMemoryCeilingBytes: self.effectiveMlxMemoryCeilingBytes,
            modelCorePayloadBytes: modelCorePayloadBytes,
            contextWindowReserveBytes: contextWindowReserveBytes,
            activationHeadroomBytes: self.activationHeadroomBytes,
            completeLayerStreamSlotBytes: completeLayerStreamSlotBytes,
            otherFixedBytes: self.otherFixedBytes,
            retainedExpertBudgetBytes: retainedExpertBudgetBytes);
    }

    /// Validates one request's context admission before the engine starts:
    /// the context workspace charge must fit inside the ceiling after the
    /// model core and the named reserves. The failure names the deficit so
    /// the worker's rejection is actionable.
    public func validateContextAdmission(
        modelCorePayloadBytes: UInt64,
        contextWindowReserveBytes: UInt64
    ) -> WorkerMlxMemoryAdmissionVerdict {
        let budget: MlxRamBudgetSnapshot = self.composedBudget(
            modelCorePayloadBytes: modelCorePayloadBytes,
            contextWindowReserveBytes: contextWindowReserveBytes,
            completeLayerStreamSlotBytes: 0);
        if budget.retainedExpertBudgetBytes == 0
            && (modelCorePayloadBytes
                .addingReportingOverflow(contextWindowReserveBytes).partialValue
                .addingReportingOverflow(self.activationHeadroomBytes).partialValue
                > self.effectiveMlxMemoryCeilingBytes) {
            let deficitBytes: UInt64 = modelCorePayloadBytes
                .addingReportingOverflow(contextWindowReserveBytes).partialValue
                .addingReportingOverflow(self.activationHeadroomBytes).partialValue
                - self.effectiveMlxMemoryCeilingBytes;
            return .rejected(deficitBytes: deficitBytes);
        }
        return .admitted(budget: budget);
    }
}

/// The governor's context-admission verdict for one request.
public enum WorkerMlxMemoryAdmissionVerdict: Equatable, Sendable {

    /// The context fits; the composed budget carries the split.
    case admitted(budget: MlxRamBudgetSnapshot);

    /// The context does not fit; the deficit names the missing bytes.
    case rejected(deficitBytes: UInt64);
}
