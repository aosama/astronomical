import Foundation

import IpcProtocol

/**
 * Serialized record shape for supervisor performance attribution, mirroring
 * the Rust struct from apps/supervisor/src/supervisor_performance_record.rs.
 *
 * `catalogEntryCount` is omitted when nil and the download detail is
 * flattened into the parent object, exactly like the serde attributes.
 */
public struct SupervisorPerformanceAttributionRecord {

    public let operationName: String

    public let startedAtUnixMillis: UInt64

    public let endedAtUnixMillis: UInt64

    public let elapsedNanoseconds: UInt64

    public let outcomeName: String

    public let catalogEntryCount: Int?

    public let downloadDetail: SupervisorDownloadMeasurementDetail?

    public init(
        operationName: String,
        startedAtUnixMillis: UInt64,
        endedAtUnixMillis: UInt64,
        elapsedNanoseconds: UInt64,
        outcomeName: String,
        catalogEntryCount: Int?,
        downloadDetail: SupervisorDownloadMeasurementDetail?
    ) {
        self.operationName = operationName
        self.startedAtUnixMillis = startedAtUnixMillis
        self.endedAtUnixMillis = endedAtUnixMillis
        self.elapsedNanoseconds = elapsedNanoseconds
        self.outcomeName = outcomeName
        self.catalogEntryCount = catalogEntryCount
        self.downloadDetail = downloadDetail
    }

    /// The serde-shaped JSON object written to the attribution JSONL file.
    public func jsonlWireValue() -> JsonWireValue {
        var wireObject: JsonWireObject = JsonWireObject(entries: [])
        wireObject.appendEntry(key: "operation", value: .string(self.operationName))
        wireObject.appendEntry(key: "started_at_unix_millis", value: .unsignedInteger(self.startedAtUnixMillis))
        wireObject.appendEntry(key: "ended_at_unix_millis", value: .unsignedInteger(self.endedAtUnixMillis))
        wireObject.appendEntry(key: "elapsed_nanoseconds", value: .unsignedInteger(self.elapsedNanoseconds))
        wireObject.appendEntry(key: "outcome", value: .string(self.outcomeName))
        if let catalogEntryCount: Int = self.catalogEntryCount {
            wireObject.appendEntry(key: "catalog_entry_count", value: .unsignedInteger(UInt64(catalogEntryCount)))
        }
        if let downloadDetail: SupervisorDownloadMeasurementDetail = self.downloadDetail {
            wireObject = downloadDetail.appendFlattenedEntries(into: wireObject)
        }
        return .object(wireObject)
    }
}
