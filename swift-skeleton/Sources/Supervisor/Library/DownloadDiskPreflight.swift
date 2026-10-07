import Foundation;

import AstronomicalConfig;

/**
 * Disk-capacity admission for staged Library downloads, migrating
 * apps/supervisor/src/library/download_disk_preflight.rs: the release
 * estimate is checked with a minimum-one-byte one-percent staging margin,
 * a resumed job with an exact manifest is checked against only its
 * remaining bytes, and every failure — query, arithmetic, admission —
 * stays typed with the evidence the operator needs.
 */

/// Typed failure of one volume-capacity query.
public enum DiskCapacityQueryError: Error, Equatable, Sendable {

    /// The volume answered that the caller may not read its capacity.
    case permissionDenied;

    /// The volume view exists but could not answer, with the transport's
    /// own wording of why.
    case volumeUnavailable(reason: String);
}

/// Queries free bytes on the volume containing an existing path.
public protocol DiskCapacityQuery: Sendable {

    func availableSpaceBytes(existingSameVolumePath: FilePath) -> Result<UInt64, DiskCapacityQueryError>;
}

/// The staging-margin divisor: one percent of the estimate.
private let onePercentDivisor: UInt64 = 100;

/// Production capacity query backed by the operating system volume view.
public struct FileManagerDiskCapacityQuery: DiskCapacityQuery {

    public init() {}

    public func availableSpaceBytes(existingSameVolumePath: FilePath) -> Result<UInt64, DiskCapacityQueryError> {
        let volumeUrl: URL = URL(fileURLWithPath: existingSameVolumePath.string);
        let volumeKey: URLResourceKey = .volumeAvailableCapacityForImportantUsageKey;
        guard let volumeValues: URLResourceValues = try? volumeUrl.resourceValues(
            forKeys: Set<URLResourceKey>([volumeKey])),
            let availableCapacity: Int64 = volumeValues.volumeAvailableCapacityForImportantUsage,
            availableCapacity >= 0
        else {
            return .failure(.volumeUnavailable(
                reason: "the volume reported no available capacity for important usage"));
        }
        return .success(UInt64(availableCapacity));
    }
}

/// The disk-admission slice of a durable download job: whether the exact
/// file manifest is known yet, and how many manifest bytes still have to
/// land on disk.
public protocol DiskPreflightDownloadJob: Sendable {

    var hasExactManifest: Bool { get }
    var remainingBytes: UInt64 { get }
}

/// Applies download-specific capacity policy through an injectable
/// volume query.
public struct DownloadDiskPreflight<CapacityQuery: DiskCapacityQuery>: Sendable {

    private let capacityQuery: CapacityQuery;

    public init(capacityQuery: CapacityQuery) {
        self.capacityQuery = capacityQuery;
    }

    public func checkInitialDownload(
        existingSameVolumePath: FilePath,
        catalogApproximateBytes: UInt64
    ) throws -> DownloadDiskCapacityCheck {
        let marginBytes: UInt64 = DownloadDiskPreflight<CapacityQuery>.stagingMarginBytes(
            catalogApproximateBytes: catalogApproximateBytes);
        let requirementAddition: (partialValue: UInt64, overflow: Bool) =
            catalogApproximateBytes.addingReportingOverflow(marginBytes);
        guard requirementAddition.overflow == false else {
            throw DownloadDiskPreflightError.requiredBytesOverflow(
                catalogApproximateBytes: catalogApproximateBytes,
                marginBytes: marginBytes);
        }
        return try self.checkRequiredBytes(
            existingSameVolumePath: existingSameVolumePath,
            requiredBytes: requirementAddition.partialValue);
    }

    /// Checks only bytes that remain after the exact manifest and staged
    /// files are known; an estimate-only job cannot take this branch.
    public func checkJobRemainingBytes(
        existingSameVolumePath: FilePath,
        downloadJob: DiskPreflightDownloadJob
    ) throws -> DownloadDiskCapacityCheck {
        if downloadJob.hasExactManifest == false {
            throw DownloadDiskPreflightError.exactManifestRequired;
        }
        return try self.checkRequiredBytes(
            existingSameVolumePath: existingSameVolumePath,
            requiredBytes: downloadJob.remainingBytes);
    }

    private func checkRequiredBytes(
        existingSameVolumePath: FilePath,
        requiredBytes: UInt64
    ) throws -> DownloadDiskCapacityCheck {
        let capacityOutcome: Result<UInt64, DiskCapacityQueryError> = self.capacityQuery.availableSpaceBytes(
            existingSameVolumePath: existingSameVolumePath);
        let availableBytes: UInt64;
        switch (capacityOutcome) {
        case let .success(queriedAvailableBytes):
            availableBytes = queriedAvailableBytes;
        case let .failure(queryFailure):
            throw DownloadDiskPreflightError.queryCapacity(
                path: existingSameVolumePath,
                requiredBytes: requiredBytes,
                source: queryFailure);
        }
        if requiredBytes > availableBytes {
            throw DownloadDiskPreflightError.insufficientSpace(
                requiredBytes: requiredBytes,
                availableBytes: availableBytes);
        }
        return DownloadDiskCapacityCheck(
            requiredBytes: requiredBytes,
            availableBytes: availableBytes);
    }

    /// One percent of the estimate, ceiled, never below one byte — the
    /// staging scratch room the margin must guarantee.
    private static func stagingMarginBytes(catalogApproximateBytes: UInt64) -> UInt64 {
        let wholePercentBytes: UInt64 = catalogApproximateBytes / onePercentDivisor;
        let hasFractionalPercent: Bool = catalogApproximateBytes % onePercentDivisor != 0;
        let ceiledPercentBytes: UInt64 = wholePercentBytes &+ (hasFractionalPercent ? 1 : 0);
        return max(ceiledPercentBytes, 1);
    }
}

/// Capacity evidence retained for attribution and later orchestration.
public struct DownloadDiskCapacityCheck: Equatable, Sendable {

    public let requiredBytes: UInt64;
    public let availableBytes: UInt64;

    public init(requiredBytes: UInt64, availableBytes: UInt64) {
        self.requiredBytes = requiredBytes;
        self.availableBytes = availableBytes;
    }
}

/// Typed disk-query, arithmetic, or admission failure.
public enum DownloadDiskPreflightError: Error, Equatable, Sendable {

    /// The remaining-byte check needs the exact manifest, not the catalog
    /// estimate.
    case exactManifestRequired;

    /// The staged requirement does not fit in the byte type; admission
    /// fails closed before any query runs.
    case requiredBytesOverflow(catalogApproximateBytes: UInt64, marginBytes: UInt64);

    /// The volume could not answer; the path and requirement ride along so
    /// the operator sees which volume and how much was being asked for.
    case queryCapacity(path: FilePath, requiredBytes: UInt64, source: DiskCapacityQueryError);

    /// The volume answered, but below the requirement.
    case insufficientSpace(requiredBytes: UInt64, availableBytes: UInt64);
}
