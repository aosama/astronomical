import Foundation

import IpcProtocol

/**
 * Validated download-specific fields embedded in supervisor performance
 * records, mirroring the Rust enum from
 * apps/supervisor/src/supervisor_download_attribution.rs. The detail is
 * flattened into the parent record's JSON, exactly like serde's `flatten`.
 */
public enum SupervisorDownloadOperationDetail: Equatable, Sendable {

    case diskPreflight(requiredBytes: UInt64, availableBytes: UInt64)
    case manifestFetch(manifestFileCount: Int, manifestTotalBytes: UInt64)
    case executablePreflight(manifestFileCount: Int)
    case fileTransfer(
        relativeFilePath: String,
        resumeOffsetBytes: UInt64,
        transferredBytes: UInt64)
    case verification(verifiedFileCount: Int, verifiedBytes: UInt64)
    case publication
    case discoveryRefresh

    /// Appends this detail's flattened entries into the parent wire object.
    public func appendFlattenedEntries(into wireObject: JsonWireObject) -> JsonWireObject {
        var builtObject: JsonWireObject = wireObject
        switch (self) {
        case let .diskPreflight(requiredBytes, availableBytes):
            builtObject.appendEntry(key: "required_bytes", value: .unsignedInteger(requiredBytes))
            builtObject.appendEntry(key: "available_bytes", value: .unsignedInteger(availableBytes))
        case let .manifestFetch(manifestFileCount, manifestTotalBytes):
            builtObject.appendEntry(key: "manifest_file_count", value: .unsignedInteger(UInt64(manifestFileCount)))
            builtObject.appendEntry(key: "manifest_total_bytes", value: .unsignedInteger(manifestTotalBytes))
        case let .executablePreflight(manifestFileCount):
            builtObject.appendEntry(key: "manifest_file_count", value: .unsignedInteger(UInt64(manifestFileCount)))
        case let .fileTransfer(relativeFilePath, resumeOffsetBytes, transferredBytes):
            builtObject.appendEntry(key: "relative_file_path", value: .string(relativeFilePath))
            builtObject.appendEntry(key: "resume_offset_bytes", value: .unsignedInteger(resumeOffsetBytes))
            builtObject.appendEntry(key: "transferred_bytes", value: .unsignedInteger(transferredBytes))
        case let .verification(verifiedFileCount, verifiedBytes):
            builtObject.appendEntry(key: "verified_file_count", value: .unsignedInteger(UInt64(verifiedFileCount)))
            builtObject.appendEntry(key: "verified_bytes", value: .unsignedInteger(verifiedBytes))
        case .publication:
            return builtObject
        case .discoveryRefresh:
            return builtObject
        }
        return builtObject
    }

    /// The matching operation for this detail, mirroring the Rust
    /// `matches_operation` pairing table: every download detail pairs with
    /// exactly one download operation.
    public var pairedOperation: SupervisorPerformanceOperation {
        switch (self) {
        case .diskPreflight: return .diskPreflight
        case .manifestFetch: return .manifestFetch
        case .executablePreflight: return .executablePreflight
        case .fileTransfer: return .fileTransfer
        case .verification: return .verification
        case .publication: return .publication
        case .discoveryRefresh: return .discoveryRefresh
        }
    }
}

/// Bounded safe relative-path policy for file-transfer attribution rows,
/// mirroring `is_safe_relative_path` from the Rust download attribution module.
public enum SupervisorRelativePathPolicy {

    public static let MAXIMUM_RELATIVE_PATH_BYTES: Int = 1_024

    public static func isSafeRelativePath(_ relativePath: String) -> Bool {
        let relativePathBytes: Array<UInt8> = Array(relativePath.utf8)
        if relativePath.isEmpty
            || relativePathBytes.count > SupervisorRelativePathPolicy.MAXIMUM_RELATIVE_PATH_BYTES
            || relativePathBytes.allSatisfy({ (pathByte: UInt8) -> Bool in return pathByte < 0x80 }) == false
            || relativePath.contains("\\")
            || relativePathBytes.contains(where: { (pathByte: UInt8) -> Bool in return pathByte < 0x20 }) {
            return false
        }
        if relativePath.hasPrefix("/") {
            return false
        }
        let pathComponents: Array<Substring> = relativePath.split(separator: "/", omittingEmptySubsequences: false)
        // A canonical relative path never has empty components ("a//b", "a/./b",
        // "a/b/") and re-joining its components reproduces it exactly.
        let isCanonical: Bool = pathComponents.allSatisfy({ (pathComponent: Substring) -> Bool in
            return !pathComponent.isEmpty && pathComponent != "." && pathComponent != ".."
        })
        if isCanonical == false {
            return false
        }
        return pathComponents.joined(separator: "/") == relativePath
    }
}
