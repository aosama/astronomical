import Foundation

/**
 * Explicit outcome and operation-specific metadata for one measured boundary,
 * mirroring the Rust type from
 * apps/supervisor/src/supervisor_performance_attribution.rs and the builders
 * from apps/supervisor/src/supervisor_performance_measurement.rs.
 */
public struct SupervisorPerformanceMeasurement: Sendable {

    public var outcome: SupervisorPerformanceOutcome

    public var catalogEntryCount: Int?

    public var downloadDetail: SupervisorDownloadMeasurementDetail?

    private init(outcome: SupervisorPerformanceOutcome) {
        self.outcome = outcome
        self.catalogEntryCount = nil
        self.downloadDetail = nil
    }

    public static func success() -> SupervisorPerformanceMeasurement {
        return SupervisorPerformanceMeasurement(outcome: .success)
    }

    public static func failure() -> SupervisorPerformanceMeasurement {
        return SupervisorPerformanceMeasurement(outcome: .failure)
    }

    public static func paused() -> SupervisorPerformanceMeasurement {
        return SupervisorPerformanceMeasurement(outcome: .paused)
    }

    public static func cancelled() -> SupervisorPerformanceMeasurement {
        return SupervisorPerformanceMeasurement(outcome: .cancelled)
    }

    public static func successfulCatalogLoad(catalogEntryCount: Int) -> SupervisorPerformanceMeasurement {
        var measurement: SupervisorPerformanceMeasurement = SupervisorPerformanceMeasurement(outcome: .success)
        measurement.catalogEntryCount = catalogEntryCount
        return measurement
    }

    /**
     * Attaches disk-preflight evidence to this measurement.
     *
     * - Throws: `SupervisorPerformanceAttributionError.invalidDownloadIdentity`
     *   when the artifact identity pair is not canonical.
     */
    public func withDiskPreflight(
        huggingfaceId: String,
        revision: String,
        requiredBytes: UInt64,
        availableBytes: UInt64
    ) throws -> SupervisorPerformanceMeasurement {
        return try self.withDownloadDetail(.diskPreflight(
            requiredBytes: requiredBytes,
            availableBytes: availableBytes),
            huggingfaceId: huggingfaceId,
            revision: revision)
    }

    /**
     * Attaches manifest-fetch evidence to this measurement.
     *
     * - Throws: `SupervisorPerformanceAttributionError.invalidDownloadIdentity`
     *   when the artifact identity pair is not canonical.
     */
    public func withManifestFetch(
        huggingfaceId: String,
        revision: String,
        manifestFileCount: Int,
        manifestTotalBytes: UInt64
    ) throws -> SupervisorPerformanceMeasurement {
        return try self.withDownloadDetail(.manifestFetch(
            manifestFileCount: manifestFileCount,
            manifestTotalBytes: manifestTotalBytes),
            huggingfaceId: huggingfaceId,
            revision: revision)
    }

    /**
     * Attaches executable-preflight evidence to this measurement.
     *
     * - Throws: `SupervisorPerformanceAttributionError` when the artifact
     *   identity pair is not canonical or the relative path is not safe.
     */
    public func withExecutablePreflight(
        huggingfaceId: String,
        revision: String,
        manifestFileCount: Int
    ) throws -> SupervisorPerformanceMeasurement {
        return try self.withDownloadDetail(.executablePreflight(
            manifestFileCount: manifestFileCount),
            huggingfaceId: huggingfaceId,
            revision: revision)
    }

    /**
     * Attaches file-transfer evidence to this measurement.
     *
     * - Throws: `SupervisorPerformanceAttributionError` when the artifact
     *   identity pair is not canonical or the relative path is not a bounded
     *   safe relative path.
     */
    public func withFileTransfer(
        huggingfaceId: String,
        revision: String,
        relativeFilePath: String,
        resumeOffsetBytes: UInt64,
        transferredBytes: UInt64
    ) throws -> SupervisorPerformanceMeasurement {
        guard SupervisorRelativePathPolicy.isSafeRelativePath(relativeFilePath) else {
            throw SupervisorPerformanceAttributionError.invalidRelativeFilePath(
                problem: "supervisor attribution requires a bounded safe relative file path")
        }
        return try self.withDownloadDetail(.fileTransfer(
            relativeFilePath: relativeFilePath,
            resumeOffsetBytes: resumeOffsetBytes,
            transferredBytes: transferredBytes),
            huggingfaceId: huggingfaceId,
            revision: revision)
    }

    /**
     * Attaches verification evidence to this measurement.
     *
     * - Throws: `SupervisorPerformanceAttributionError.invalidDownloadIdentity`
     *   when the artifact identity pair is not canonical.
     */
    public func withVerification(
        huggingfaceId: String,
        revision: String,
        verifiedFileCount: Int,
        verifiedBytes: UInt64
    ) throws -> SupervisorPerformanceMeasurement {
        return try self.withDownloadDetail(.verification(
            verifiedFileCount: verifiedFileCount,
            verifiedBytes: verifiedBytes),
            huggingfaceId: huggingfaceId,
            revision: revision)
    }

    /**
     * Attaches publication evidence to this measurement.
     *
     * - Throws: `SupervisorPerformanceAttributionError.invalidDownloadIdentity`
     *   when the artifact identity pair is not canonical.
     */
    public func withPublication(
        huggingfaceId: String,
        revision: String
    ) throws -> SupervisorPerformanceMeasurement {
        return try self.withDownloadDetail(.publication, huggingfaceId: huggingfaceId, revision: revision)
    }

    /**
     * Attaches discovery-refresh evidence to this measurement.
     *
     * - Throws: `SupervisorPerformanceAttributionError.invalidDownloadIdentity`
     *   when the artifact identity pair is not canonical.
     */
    public func withDiscoveryRefresh(
        huggingfaceId: String,
        revision: String
    ) throws -> SupervisorPerformanceMeasurement {
        return try self.withDownloadDetail(.discoveryRefresh, huggingfaceId: huggingfaceId, revision: revision)
    }

    private func withDownloadDetail(
        _ operationDetail: SupervisorDownloadOperationDetail,
        huggingfaceId: String,
        revision: String
    ) throws -> SupervisorPerformanceMeasurement {
        var measurement: SupervisorPerformanceMeasurement = self
        measurement.downloadDetail = try SupervisorDownloadMeasurementDetail.new(
            huggingfaceId: huggingfaceId,
            revision: revision,
            operationDetail: operationDetail)
        return measurement
    }

    /**
     * The operation/detail pairing contract, mirroring the Rust
     * `matches_operation` table: a download operation must carry its matching
     * download detail, and only catalog loads and Qwen seed loads run without
     * any download detail.
     */
    public func matchesOperation(_ operation: SupervisorPerformanceOperation) -> Bool {
        if operation == .libraryCatalogLoad || operation == .qwenThinkingChannelSeedLoad {
            return self.downloadDetail == nil
        }
        guard let detail: SupervisorDownloadMeasurementDetail = self.downloadDetail else {
            return false
        }
        return detail.operationDetail.pairedOperation == operation
    }
}
