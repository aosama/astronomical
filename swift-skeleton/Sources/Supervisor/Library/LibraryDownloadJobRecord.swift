import Foundation

import AstronomicalConfig

/// Durable state of the one Library download this instance owns. Kept thin
/// per #1059: the hub blob cache owns partial bytes, this record owns the
/// user-visible lifecycle and is written only on state transitions so live
/// progress never amplifies durable writes.
public struct LibraryDownloadJobRecord: Equatable, Sendable, Codable {
    public let huggingfaceId: String
    public let revision: String
    public var state: LibraryDownloadJobRecordState
    public var bytesCompleted: UInt64
    public var bytesTotal: UInt64
    public var errorCode: String?
    public var updatedAtUnixMillis: UInt64

    public init(
        huggingfaceId: String,
        revision: String,
        state: LibraryDownloadJobRecordState,
        bytesCompleted: UInt64,
        bytesTotal: UInt64,
        errorCode: String?,
        updatedAtUnixMillis: UInt64
    ) {
        self.huggingfaceId = huggingfaceId
        self.revision = revision
        self.state = state
        self.bytesCompleted = bytesCompleted
        self.bytesTotal = bytesTotal
        self.errorCode = errorCode
        self.updatedAtUnixMillis = updatedAtUnixMillis
    }
}

/// The persisted lifecycle states, wire-named exactly like the Rust job states.
public enum LibraryDownloadJobRecordState: String, Equatable, Sendable, Codable {
    case checkingDisk = "checking_disk"
    case fetchingManifest = "fetching_manifest"
    case downloading = "downloading"
    case paused = "paused"
    case verifying = "verifying"
    case publishing = "publishing"
    case failed = "failed"

    public var isInterruptedByRestart: Bool {
        switch self {
        case .checkingDisk, .fetchingManifest, .downloading, .verifying:
            return true
        case .paused, .publishing, .failed:
            return false
        }
    }
}

/// Stable public failures, path-free by contract.
public enum LibraryDownloadPublicErrorCode: String, Sendable {
    case libraryBusy = "library_busy"
    case catalogEntryNotFound = "catalog_entry_not_found"
    case modelNotPublic = "model_not_public"
    case insufficientDisk = "insufficient_disk"
    case downloadGated = "download_gated"
    case checksumMismatch = "checksum_mismatch"
    case downloadFailed = "download_failed"
    case modelAlreadyPresent = "model_already_present"
    case modelNotExecutable = "model_not_executable"
}

/// Filesystem home of the durable job record: one JSON document under the
/// instance state directory; deleted when the job ends (published or cancelled).
public struct LibraryDownloadJobRecordStore: Sendable {
    private let recordFilePath: URL

    public init(stateDirectory: FilePath) {
        self.recordFilePath = URL(
            fileURLWithPath: stateDirectory.appending(component: "library-download-job.json").string)
    }

    public func load() throws -> LibraryDownloadJobRecord? {
        guard let recordBytes: Data = FileManager.default.contents(atPath: self.recordFilePath.path) else {
            return nil
        }
        return try JSONDecoder().decode(LibraryDownloadJobRecord.self, from: recordBytes)
    }

    public func save(_ record: LibraryDownloadJobRecord) throws {
        let recordBytes: Data = try JSONEncoder().encode(record)
        try recordBytes.write(to: self.recordFilePath, options: [.atomic])
    }

    public func delete() {
        try? FileManager.default.removeItem(at: self.recordFilePath)
    }
}
