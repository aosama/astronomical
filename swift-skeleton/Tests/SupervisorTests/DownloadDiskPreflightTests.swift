import Foundation;

import Testing;

import AstronomicalConfig;
import JourneyCategories;

@testable import Supervisor;

/**
 * Hermetic admission contracts for Library download disk capacity,
 * migrating apps/supervisor/tests/hermetic/download_disk_preflight.rs: the
 * release estimate is admitted when the catalog bytes plus a ceiled
 * one-percent margin (never below one byte) fit; a resumed job with an
 * exact manifest is checked against exactly its remaining bytes with no
 * extra margin; and every rejection — insufficient space, unrepresentable
 * requirement, estimate-only job, or failed volume query — stays typed
 * with the path, requirement, and cause preserved.
 */
@Suite(.tags(.hermeticJourney))
final class DownloadDiskPreflightTests {

    private static let fictionalLibraryPath: FilePath = FilePath(string: "fictional-library");

    @Test
    func should_admit_initial_download_when_catalog_bytes_and_one_percent_margin_fit() throws {
        let capacityQuery: FakeDiskCapacityQuery = FakeDiskCapacityQuery(availableBytes: 4_040_000_000);
        let preflight: DownloadDiskPreflight<FakeDiskCapacityQuery> = DownloadDiskPreflight(
            capacityQuery: capacityQuery);

        let capacityCheck: DownloadDiskCapacityCheck = try preflight.checkInitialDownload(
            existingSameVolumePath: DownloadDiskPreflightTests.fictionalLibraryPath,
            catalogApproximateBytes: 4_000_000_000);

        #expect(capacityCheck.requiredBytes == 4_040_000_000);
        #expect(capacityCheck.availableBytes == 4_040_000_000);
        #expect(
            capacityQuery.queriedPaths() == [DownloadDiskPreflightTests.fictionalLibraryPath],
            "the query must run against the same-volume path");
    }

    @Test
    func should_apply_one_percent_with_a_minimum_one_byte_margin() throws {
        let marginCases: Array<(catalogApproximateBytes: UInt64, expectedRequiredBytes: UInt64)> = [
            (catalogApproximateBytes: 1, expectedRequiredBytes: 2),
            (catalogApproximateBytes: 100, expectedRequiredBytes: 101),
            (catalogApproximateBytes: 101, expectedRequiredBytes: 103),
        ];
        for marginCase in marginCases {
            let preflight: DownloadDiskPreflight<FakeDiskCapacityQuery> = DownloadDiskPreflight(
                capacityQuery: FakeDiskCapacityQuery(availableBytes: marginCase.expectedRequiredBytes));

            let capacityCheck: DownloadDiskCapacityCheck = try preflight.checkInitialDownload(
                existingSameVolumePath: DownloadDiskPreflightTests.fictionalLibraryPath,
                catalogApproximateBytes: marginCase.catalogApproximateBytes);

            #expect(
                capacityCheck.requiredBytes == marginCase.expectedRequiredBytes,
                "the ceiled one-percent margin with a one-byte floor must apply");
        }
    }

    @Test
    func should_reject_initial_download_with_required_and_available_bytes() throws {
        let preflight: DownloadDiskPreflight<FakeDiskCapacityQuery> = DownloadDiskPreflight(
            capacityQuery: FakeDiskCapacityQuery(availableBytes: 1_009));

        let preflightError: DownloadDiskPreflightError = try DownloadDiskPreflightTests.requirePreflightFailure({
            try preflight.checkInitialDownload(
                existingSameVolumePath: DownloadDiskPreflightTests.fictionalLibraryPath,
                catalogApproximateBytes: 1_000)
        });

        #expect(
            preflightError
                == DownloadDiskPreflightError.insufficientSpace(requiredBytes: 1_010, availableBytes: 1_009));
    }

    @Test
    func should_admit_exact_manifest_remaining_bytes_without_an_extra_margin() throws {
        let preflight: DownloadDiskPreflight<FakeDiskCapacityQuery> = DownloadDiskPreflight(
            capacityQuery: FakeDiskCapacityQuery(availableBytes: 101));

        let capacityCheck: DownloadDiskCapacityCheck = try preflight.checkJobRemainingBytes(
            existingSameVolumePath: DownloadDiskPreflightTests.fictionalLibraryPath,
            downloadJob: DownloadDiskPreflightTests.jobWithProgress(bytesCompleted: 12, manifestBytesTotal: 113));

        #expect(capacityCheck.requiredBytes == 101);
        #expect(capacityCheck.availableBytes == 101);
    }

    @Test
    func should_reject_exact_manifest_remaining_bytes_with_typed_capacity_evidence() throws {
        let preflight: DownloadDiskPreflight<FakeDiskCapacityQuery> = DownloadDiskPreflight(
            capacityQuery: FakeDiskCapacityQuery(availableBytes: 100));

        let preflightError: DownloadDiskPreflightError = try DownloadDiskPreflightTests.requirePreflightFailure({
            try preflight.checkJobRemainingBytes(
                existingSameVolumePath: DownloadDiskPreflightTests.fictionalLibraryPath,
                downloadJob: DownloadDiskPreflightTests.jobWithProgress(bytesCompleted: 12, manifestBytesTotal: 113))
        });

        #expect(
            preflightError
                == DownloadDiskPreflightError.insufficientSpace(requiredBytes: 101, availableBytes: 100));
    }

    @Test
    func should_reject_remaining_byte_check_before_an_exact_manifest_exists() throws {
        let capacityQuery: FakeDiskCapacityQuery = FakeDiskCapacityQuery(availableBytes: 1_000);
        let preflight: DownloadDiskPreflight<FakeDiskCapacityQuery> = DownloadDiskPreflight(
            capacityQuery: capacityQuery);
        let premanifestJob: FixtureDownloadJob = DownloadDiskPreflightTests.jobWithProgress(
            bytesCompleted: 0,
            manifestBytesTotal: 0);

        let preflightError: DownloadDiskPreflightError = try DownloadDiskPreflightTests.requirePreflightFailure({
            try preflight.checkJobRemainingBytes(
                existingSameVolumePath: DownloadDiskPreflightTests.fictionalLibraryPath,
                downloadJob: premanifestJob)
        });

        #expect(preflightError == DownloadDiskPreflightError.exactManifestRequired);
        #expect(
            capacityQuery.queriedPaths().isEmpty,
            "an estimate-only job must fail before the volume is queried");
    }

    @Test
    func should_report_checked_initial_requirement_overflow_before_querying_capacity() throws {
        let capacityQuery: FakeDiskCapacityQuery = FakeDiskCapacityQuery(availableBytes: UInt64.max);
        let preflight: DownloadDiskPreflight<FakeDiskCapacityQuery> = DownloadDiskPreflight(
            capacityQuery: capacityQuery);

        let preflightError: DownloadDiskPreflightError = try DownloadDiskPreflightTests.requirePreflightFailure({
            try preflight.checkInitialDownload(
                existingSameVolumePath: DownloadDiskPreflightTests.fictionalLibraryPath,
                catalogApproximateBytes: UInt64.max)
        });

        #expect(
            preflightError == DownloadDiskPreflightError.requiredBytesOverflow(
                catalogApproximateBytes: UInt64.max,
                marginBytes: 184_467_440_737_095_517));
        #expect(
            capacityQuery.queriedPaths().isEmpty,
            "an unrepresentable requirement must fail before the volume is queried");
    }

    @Test
    func should_preserve_capacity_query_path_required_bytes_and_io_cause() throws {
        let preflight: DownloadDiskPreflight<FakeDiskCapacityQuery> = DownloadDiskPreflight(
            capacityQuery: FakeDiskCapacityQuery(queryFailure: .permissionDenied));

        let preflightError: DownloadDiskPreflightError = try DownloadDiskPreflightTests.requirePreflightFailure({
            try preflight.checkJobRemainingBytes(
                existingSameVolumePath: DownloadDiskPreflightTests.fictionalLibraryPath,
                downloadJob: DownloadDiskPreflightTests.jobWithProgress(bytesCompleted: 250, manifestBytesTotal: 1_000))
        });

        #expect(
            preflightError == DownloadDiskPreflightError.queryCapacity(
                path: DownloadDiskPreflightTests.fictionalLibraryPath,
                requiredBytes: 750,
                source: .permissionDenied));
    }

    /// Runs one admission attempt that must fail, and returns its typed
    /// preflight error; an admission or an untyped failure is a journey
    /// defect, never a silent pass.
    private static func requirePreflightFailure(
        _ admissionAttempt: () throws -> DownloadDiskCapacityCheck
    ) throws -> DownloadDiskPreflightError {
        do {
            _ = try admissionAttempt();
        } catch let typedPreflightError as DownloadDiskPreflightError {
            return typedPreflightError;
        } catch let admissionError {
            throw DownloadDiskPreflightProbeFailure.untypedFailure("\(admissionError)");
        }
        throw DownloadDiskPreflightProbeFailure.admittedUnexpectedly;
    }

    // MARK: Fixtures

    /// One paused job with an exact single-file manifest; an empty manifest
    /// carries no exact knowledge, exactly the premanifest record shape.
    private static func jobWithProgress(
        bytesCompleted: UInt64,
        manifestBytesTotal: UInt64
    ) -> FixtureDownloadJob {
        return FixtureDownloadJob(
            manifestFileBytes: manifestBytesTotal,
            bytesCompleted: bytesCompleted);
    }
}

/// The durable-job slice the preflight reads: remaining manifest bytes with
/// a manifest-present flag.
struct FixtureDownloadJob: DiskPreflightDownloadJob {

    private let manifestFileBytes: UInt64;
    private let bytesCompleted: UInt64;

    init(manifestFileBytes: UInt64, bytesCompleted: UInt64) {
        self.manifestFileBytes = manifestFileBytes;
        self.bytesCompleted = bytesCompleted;
    }

    var hasExactManifest: Bool {
        return self.manifestFileBytes > 0;
    }

    var remainingBytes: UInt64 {
        return self.manifestFileBytes - self.bytesCompleted;
    }
}

/// Thread-safe scripted volume query remembering every path it answered.
final class FakeDiskCapacityQuery: DiskCapacityQuery, @unchecked Sendable {

    private let stateLock: NSLock = NSLock();
    private let availableBytesOutcome: Result<UInt64, DiskCapacityQueryError>;
    private var queriedPathStrings: Array<String> = [];

    init(availableBytes: UInt64) {
        self.availableBytesOutcome = .success(availableBytes);
    }

    init(queryFailure: DiskCapacityQueryError) {
        self.availableBytesOutcome = .failure(queryFailure);
    }

    func availableSpaceBytes(existingSameVolumePath: FilePath) -> Result<UInt64, DiskCapacityQueryError> {
        self.stateLock.lock();
        self.queriedPathStrings.append(existingSameVolumePath.string);
        self.stateLock.unlock();
        return self.availableBytesOutcome;
    }

    func queriedPaths() -> Array<FilePath> {
        self.stateLock.lock();
        let currentQueriedPathStrings: Array<String> = self.queriedPathStrings;
        self.stateLock.unlock();
        return currentQueriedPathStrings.map({ (queriedPathString: String) -> FilePath in
            return FilePath(string: queriedPathString);
        });
    }
}

/// Typed probe failures when an admission attempt should have failed.
enum DownloadDiskPreflightProbeFailure: Error {

    case admittedUnexpectedly;
    case untypedFailure(String);
}
