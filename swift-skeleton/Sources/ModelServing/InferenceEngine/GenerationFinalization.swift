import Foundation;

import IpcProtocol;

/// Final engine residency state for one layer of sparse experts.
public struct ExpertResidencyTelemetry: Equatable {

    public let totalLayerCount: UInt32;
    public let residentExpertCount: UInt32;
    public let residentExpertPayloadBytes: UInt64;

    public init(
        totalLayerCount: UInt32,
        residentExpertCount: UInt32,
        residentExpertPayloadBytes: UInt64
    ) {
        self.totalLayerCount = totalLayerCount;
        self.residentExpertCount = residentExpertCount;
        self.residentExpertPayloadBytes = residentExpertPayloadBytes;
    }
}

/// Final engine state reported once a generation ends or is cancelled.
public struct GenerationFinalization: Equatable {

    public let expertMemoryMode: ExpertMemoryMode?;
    public let mlxMemorySnapshot: WorkerMlxMemorySnapshot?;
    public let expertResidencyTelemetry: ExpertResidencyTelemetry?;

    public init(
        expertMemoryMode: ExpertMemoryMode? = nil,
        mlxMemorySnapshot: WorkerMlxMemorySnapshot? = nil,
        expertResidencyTelemetry: ExpertResidencyTelemetry? = nil
    ) {
        self.expertMemoryMode = expertMemoryMode;
        self.mlxMemorySnapshot = mlxMemorySnapshot;
        self.expertResidencyTelemetry = expertResidencyTelemetry;
    }

    /// Whether any final state is worth a `generationFinalized` event; the
    /// worker skips the event when the engine released nothing observable.
    public var hasReportableState: Bool {
        return self.expertMemoryMode != nil
            || self.mlxMemorySnapshot != nil
            || self.expertResidencyTelemetry != nil;
    }
}
