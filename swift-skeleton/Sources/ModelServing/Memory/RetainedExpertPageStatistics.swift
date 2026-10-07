import Foundation

/// Observed ownership state of a retained expert page cache.
///
/// Telemetry reports the same byte figures the policies decided on — no
/// re-measurement drift between decision and report.
public struct RetainedExpertPageStatistics: Equatable, Sendable {

    /// Retained layers currently owned.
    public var entryCount: Int

    /// Sum of payload bytes for every owned slot; metadata excluded.
    public var residentPayloadByteCount: UInt64

    /// Effective ceiling: the tighter of the long-lived budget and any
    /// live request-pressure freeze.
    public var maximumResidentPayloadByteCount: UInt64

    /// Total pages dropped since construction.
    public var evictionCount: UInt64

    /// Expert pages read from disk since construction.
    public var diskPageLoadCount: UInt64

    /// Disk batches (grouped page reads) since construction.
    public var diskBatchLoadCount: UInt64

    /// Owned stable complete layers.
    public var completeLayerCount: Int

    /// Payload bytes owned by stable complete layers.
    public var completeLayerPayloadByteCount: UInt64

    /// Owned elastic routed-expert pages.
    public var partialLayerCount: Int

    /// Payload bytes owned by elastic routed-expert pages.
    public var partialLayerPayloadByteCount: UInt64

    /// Complete layers committed by mandatory reads.
    public var mandatoryReadPromotionCount: UInt64

    /// Stable complete layers dropped since construction.
    public var completeLayerEvictionCount: UInt64

    /// Elastic routed pages dropped since construction.
    public var partialLayerEvictionCount: UInt64

    public init(
        entryCount: Int,
        residentPayloadByteCount: UInt64,
        maximumResidentPayloadByteCount: UInt64,
        evictionCount: UInt64,
        diskPageLoadCount: UInt64,
        diskBatchLoadCount: UInt64,
        completeLayerCount: Int,
        completeLayerPayloadByteCount: UInt64,
        partialLayerCount: Int,
        partialLayerPayloadByteCount: UInt64,
        mandatoryReadPromotionCount: UInt64,
        completeLayerEvictionCount: UInt64,
        partialLayerEvictionCount: UInt64
    ) {
        self.entryCount = entryCount
        self.residentPayloadByteCount = residentPayloadByteCount
        self.maximumResidentPayloadByteCount = maximumResidentPayloadByteCount
        self.evictionCount = evictionCount
        self.diskPageLoadCount = diskPageLoadCount
        self.diskBatchLoadCount = diskBatchLoadCount
        self.completeLayerCount = completeLayerCount
        self.completeLayerPayloadByteCount = completeLayerPayloadByteCount
        self.partialLayerCount = partialLayerCount
        self.partialLayerPayloadByteCount = partialLayerPayloadByteCount
        self.mandatoryReadPromotionCount = mandatoryReadPromotionCount
        self.completeLayerEvictionCount = completeLayerEvictionCount
        self.partialLayerEvictionCount = partialLayerEvictionCount
    }
}
