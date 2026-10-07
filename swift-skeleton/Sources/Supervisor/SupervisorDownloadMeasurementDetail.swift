import Foundation

import IpcProtocol

/**
 * Validated download-specific fields embedded in supervisor performance
 * records, mirroring the Rust struct from
 * apps/supervisor/src/supervisor_download_attribution.rs. Serializes
 * flattened into the parent attribution row.
 */
public struct SupervisorDownloadMeasurementDetail: Equatable, Sendable {

    public let huggingfaceId: String

    public let revision: String

    public let operationDetail: SupervisorDownloadOperationDetail

    /**
     * Builds a detail after validating the artifact identity pair.
     *
     * - Throws: `SupervisorPerformanceAttributionError.invalidDownloadIdentity`
     *   when the Hugging Face identity or immutable revision is not canonical.
     */
    public static func new(
        huggingfaceId: String,
        revision: String,
        operationDetail: SupervisorDownloadOperationDetail
    ) throws -> SupervisorDownloadMeasurementDetail {
        guard DownloadCatalog.isValidHuggingFaceId(huggingfaceId) else {
            throw SupervisorPerformanceAttributionError.invalidDownloadIdentity(
                problem: "supervisor attribution requires a validated Hugging Face identity")
        }
        guard DownloadCatalog.isValidImmutableRevision(revision) else {
            throw SupervisorPerformanceAttributionError.invalidDownloadIdentity(
                problem: "supervisor attribution requires a validated immutable revision")
        }
        return SupervisorDownloadMeasurementDetail.validated(
            huggingfaceId: huggingfaceId,
            revision: revision,
            operationDetail: operationDetail)
    }

    /// Builds a detail from an already-validated identity pair.
    public static func validated(
        huggingfaceId: String,
        revision: String,
        operationDetail: SupervisorDownloadOperationDetail
    ) -> SupervisorDownloadMeasurementDetail {
        return SupervisorDownloadMeasurementDetail(
            huggingfaceId: huggingfaceId,
            revision: revision,
            operationDetail: operationDetail)
    }

    /// Appends this detail's flattened entries (identity first, then the
    /// operation-specific fields) into the parent wire object.
    public func appendFlattenedEntries(into wireObject: JsonWireObject) -> JsonWireObject {
        var builtObject: JsonWireObject = wireObject
        builtObject.appendEntry(key: "huggingface_id", value: .string(self.huggingfaceId))
        builtObject.appendEntry(key: "revision", value: .string(self.revision))
        return self.operationDetail.appendFlattenedEntries(into: builtObject)
    }
}
