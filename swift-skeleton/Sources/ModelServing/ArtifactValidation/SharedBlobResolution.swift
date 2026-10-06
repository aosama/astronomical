import Foundation;
import IpcProtocol;

/// Verified resolution of Hugging Face shared cache blobs. Port of
/// crates/model-serving/src/artifact_validation/shared_blob_resolution.rs.
public enum SharedBlobResolution {

    /// Directory, directly under a Hugging Face hub root, that holds shared blobs.
    private static let SHARED_BLOB_STORE_DIRECTORY_NAME: String = "blobs";
    /// Directory, inside one cache entry, that holds the per-revision tree metadata.
    private static let TREES_DIRECTORY_NAME: String = "trees";
    /// Snapshot tree key holding the per-file record map.
    private static let TREE_FILES_KEY: String = "files";
    /// Snapshot tree key holding a file's plain byte size.
    private static let TREE_SIZE_KEY: String = "size";
    /// Snapshot tree key holding a classic LFS blob's SHA-256 digest.
    private static let TREE_LFS_SHA256_KEY: String = "lfs_sha256";
    /// Snapshot tree key holding a Xet-backed blob's content hash.
    private static let TREE_XET_HASH_KEY: String = "xet_hash";
    /// A snapshot tree record for a small cache never approaches this bound; the
    /// limit exists so a corrupt or hostile tree file cannot be read into memory
    /// without limit.
    private static let MAXIMUM_TREE_METADATA_BYTES: UInt64 = 8 * 1024 * 1024;
    private static let SHA256_DIGEST_HEX_CHARACTERS: Int = 64;
    /// Number of leading digest characters used as the shared store's subdirectory.
    private static let CONTENT_ADDRESS_DIRECTORY_HEX_CHARACTERS: Int = 2;

    /// One file record from a Hugging Face cache entry's snapshot tree metadata.
    private struct SnapshotTreeRecord {
        /// Digest the shared store names this object after, preferring the Xet
        /// hash because Xet-backed caches store the object under that name.
        let primaryRecordedDigest: String;
        /// Every digest the tree records for this file, in preference order.
        let recordedContentDigests: Array<String>;
        /// Exact byte length the tree records for this file.
        let recordedSizeBytes: UInt64;
    }

    /// Resolves the canonical resolved target path as a verified shared Hugging
    /// Face cache blob when the snapshot symlink legitimately reaches the
    /// hub-level blob store. Returns nil when the target is not inside a
    /// hub-level shared blob store, leaving the caller's own confinement
    /// decision in charge.
    ///
    /// Verification is provenance-based rather than a byte re-hash. The store
    /// names every object after the digest recorded in the snapshot tree, so
    /// accepting a target requires all of: the store is a real (non-symlink)
    /// directory, the target sits inside it under the recorded content address,
    /// the target is a regular file, and its length matches the recorded size.
    /// Re-hashing would add a second full read of multi-gigabyte shards on the
    /// model-load critical path, which this codebase deliberately avoids.
    public static func resolveVerifiedSharedHubBlobPath(
        hubRootDirectory: String, snapshotDirectory: String,
        canonicalResolvedTargetPath: String, requiredFileName: String) throws -> String? {
        let sharedBlobStoreDirectory: String = SharedBlobResolution.joinPath(
            hubRootDirectory, SHARED_BLOB_STORE_DIRECTORY_NAME);
        guard let sharedBlobStoreMetadata: [FileAttributeKey: Any] =
            SharedBlobResolution.inspectedAttributes(atPath: sharedBlobStoreDirectory) else {
            return nil;
        }
        if SharedBlobResolution.attributesDescribeSymlink(sharedBlobStoreMetadata)
            || SharedBlobResolution.attributesDescribeDirectory(sharedBlobStoreMetadata) == false {
            return nil;
        }
        let canonicalSharedBlobStoreDirectory: String =
            SharedBlobResolution.canonicalizedPath(sharedBlobStoreDirectory);
        if SharedBlobResolution.isPath(
            canonicalResolvedTargetPath, withinDirectory: canonicalSharedBlobStoreDirectory) == false {
            return nil;
        }

        let snapshotTreeRecord: SnapshotTreeRecord = try SharedBlobResolution.readSnapshotTreeRecord(
            snapshotDirectory: snapshotDirectory, requiredFileName: requiredFileName);
        let isRecordedContentAddressedObject: Bool = snapshotTreeRecord.recordedContentDigests
            .contains(where: { (recordedContentDigest: String) -> Bool in
                return SharedBlobResolution.contentAddressedBlobPathMatches(
                    sharedBlobStoreDirectory: canonicalSharedBlobStoreDirectory,
                    contentDigest: recordedContentDigest,
                    canonicalResolvedTargetPath: canonicalResolvedTargetPath);
            });
        if isRecordedContentAddressedObject == false {
            throw ArtifactValidationError.huggingFaceSharedBlobIdentityMismatch(
                fileName: requiredFileName,
                recordedDigestText: snapshotTreeRecord.primaryRecordedDigest);
        }

        guard let sharedBlobMetadata: [FileAttributeKey: Any] =
            SharedBlobResolution.inspectedAttributes(atPath: canonicalResolvedTargetPath) else {
            throw SharedBlobResolution.unavailableSharedBlobMetadata(
                requiredFileName: requiredFileName,
                problem: "failed to inspect the resolved shared cache blob");
        }
        if SharedBlobResolution.attributesDescribeRegularFile(sharedBlobMetadata) == false {
            throw ArtifactValidationError.requiredFileIsNotRegular(fileName: requiredFileName);
        }
        let actualSharedBlobSizeBytes: UInt64 = SharedBlobResolution.fileSize(
            fromAttributes: sharedBlobMetadata);
        if actualSharedBlobSizeBytes != snapshotTreeRecord.recordedSizeBytes {
            throw ArtifactValidationError.huggingFaceSharedBlobSizeMismatch(
                fileName: requiredFileName,
                recordedSizeBytes: snapshotTreeRecord.recordedSizeBytes,
                actualSizeBytes: actualSharedBlobSizeBytes);
        }
        return canonicalResolvedTargetPath;
    }

    /// Reads the snapshot tree record that anchors trust for one cache file.
    ///
    /// The tree metadata is written by the Hugging Face client next to the blobs
    /// and is not part of the snapshot itself, so a snapshot whose tree record
    /// is missing or unreadable is rejected rather than trusted on name alone.
    private static func readSnapshotTreeRecord(
        snapshotDirectory: String, requiredFileName: String) throws -> SnapshotTreeRecord {
        let treeMetadataPath: String = SharedBlobResolution.snapshotTreeMetadataPath(
            snapshotDirectory: snapshotDirectory);
        let treeMetadataBytes: Data = try SharedBlobResolution.readBoundedTreeMetadata(
            treeMetadataPath: treeMetadataPath, requiredFileName: requiredFileName);
        let treeMetadataObject: JsonWireObject;
        do {
            let treeMetadataWireValue: JsonWireValue = try JsonWireParser.parseDocument(
                documentBytes: treeMetadataBytes);
            treeMetadataObject = try JsonWireValue.extractObject(treeMetadataWireValue);
        } catch let jsonWireProblem as JsonWireProblem {
            throw SharedBlobResolution.unavailableSharedBlobMetadata(
                requiredFileName: requiredFileName, problem: jsonWireProblem.description);
        } catch {
            throw SharedBlobResolution.unavailableSharedBlobMetadata(
                requiredFileName: requiredFileName,
                problem: "malformed snapshot tree metadata JSON");
        }
        // Rust reaches the per-file record through one optional-value chain, so
        // a missing files map and a missing file record share the same failure
        // text; keep that behavior here.
        var fileRecordObject: JsonWireObject? = nil;
        if let filesWireValue: JsonWireValue = treeMetadataObject.value(forKey: TREE_FILES_KEY),
            let filesObject: JsonWireObject = try? JsonWireValue.extractObject(filesWireValue) {
            if let fileRecordWireValue: JsonWireValue = filesObject.value(forKey: requiredFileName) {
                fileRecordObject = try? JsonWireValue.extractObject(fileRecordWireValue);
            }
        }
        guard let resolvedFileRecordObject: JsonWireObject = fileRecordObject else {
            throw SharedBlobResolution.unavailableSharedBlobMetadata(
                requiredFileName: requiredFileName,
                problem: "snapshot tree metadata has no record for this file");
        }
        guard let recordedSizeBytes: UInt64 = try resolvedFileRecordObject
            .decodeOptionalUInt64AllowingAbsent(fieldName: TREE_SIZE_KEY) else {
            throw SharedBlobResolution.unavailableSharedBlobMetadata(
                requiredFileName: requiredFileName,
                problem: "snapshot tree record has no byte size");
        }
        var recordedContentDigests: Array<String> = Array();
        for digestKey: String in [TREE_XET_HASH_KEY, TREE_LFS_SHA256_KEY] {
            guard let digestText: String = try resolvedFileRecordObject
                .decodeOptionalStringAllowingAbsent(fieldName: digestKey) else {
                continue;
            }
            recordedContentDigests.append(try SharedBlobResolution.decodeContentDigest(
                digestText: digestText, requiredFileName: requiredFileName));
        }
        guard let primaryRecordedDigest: String = recordedContentDigests.first else {
            throw SharedBlobResolution.unavailableSharedBlobMetadata(
                requiredFileName: requiredFileName,
                problem: "snapshot tree record has no content digest");
        }
        return SnapshotTreeRecord(
            primaryRecordedDigest: primaryRecordedDigest,
            recordedContentDigests: recordedContentDigests,
            recordedSizeBytes: recordedSizeBytes);
    }

    private static func snapshotTreeMetadataPath(snapshotDirectory: String) -> String {
        let revisionName: String = (snapshotDirectory as NSString).lastPathComponent;
        let modelCacheDirectory: String = ((snapshotDirectory as NSString).deletingLastPathComponent
            as NSString).deletingLastPathComponent;
        return SharedBlobResolution.joinPath(
            SharedBlobResolution.joinPath(modelCacheDirectory, TREES_DIRECTORY_NAME),
            revisionName + ".json");
    }

    private static func readBoundedTreeMetadata(
        treeMetadataPath: String, requiredFileName: String) throws -> Data {
        guard let treeMetadataAttributes: [FileAttributeKey: Any] =
            SharedBlobResolution.inspectedAttributes(atPath: treeMetadataPath) else {
            throw SharedBlobResolution.unavailableSharedBlobMetadata(
                requiredFileName: requiredFileName,
                problem: "failed to inspect the snapshot tree metadata");
        }
        let treeMetadataBytes: UInt64 = SharedBlobResolution.fileSize(
            fromAttributes: treeMetadataAttributes);
        if treeMetadataBytes > MAXIMUM_TREE_METADATA_BYTES {
            throw SharedBlobResolution.unavailableSharedBlobMetadata(
                requiredFileName: requiredFileName,
                problem: "snapshot tree metadata is larger than the accepted bound");
        }
        guard let treeMetadataData: Data = FileManager.default.contents(atPath: treeMetadataPath) else {
            throw SharedBlobResolution.unavailableSharedBlobMetadata(
                requiredFileName: requiredFileName,
                problem: "failed to read the snapshot tree metadata");
        }
        return treeMetadataData;
    }

    /// Accepts only a lowercase hexadecimal content digest of the length the
    /// shared store's directory layout assumes.
    private static func decodeContentDigest(
        digestText: String, requiredFileName: String) throws -> String {
        let isLowercaseHexadecimal: Bool = digestText.allSatisfy({ (digestCharacter: Character) -> Bool in
            return (digestCharacter >= "0" && digestCharacter <= "9")
                || (digestCharacter >= "a" && digestCharacter <= "f");
        });
        if digestText.count != SHA256_DIGEST_HEX_CHARACTERS || isLowercaseHexadecimal == false {
            throw SharedBlobResolution.unavailableSharedBlobMetadata(
                requiredFileName: requiredFileName,
                problem: "snapshot tree record has a malformed content digest");
        }
        return digestText;
    }

    /// Checks both Hugging Face shared-store filename layouts.
    ///
    /// Classic LFS blobs use the digest remainder after the two-character
    /// directory prefix. Xet-backed blobs retain the complete digest as the
    /// filename. Both layouts are constrained to the canonical hub-level store
    /// and the digest recorded by the snapshot tree.
    private static func contentAddressedBlobPathMatches(
        sharedBlobStoreDirectory: String, contentDigest: String,
        canonicalResolvedTargetPath: String) -> Bool {
        guard contentDigest.count >= CONTENT_ADDRESS_DIRECTORY_HEX_CHARACTERS else {
            return false;
        }
        let directoryPrefix: String = String(
            contentDigest.prefix(CONTENT_ADDRESS_DIRECTORY_HEX_CHARACTERS));
        let digestRemainder: String = String(
            contentDigest.dropFirst(CONTENT_ADDRESS_DIRECTORY_HEX_CHARACTERS));
        let prefixedStorePath: String = SharedBlobResolution.joinPath(
            SharedBlobResolution.joinPath(sharedBlobStoreDirectory, directoryPrefix), digestRemainder);
        let fullDigestStorePath: String = SharedBlobResolution.joinPath(
            SharedBlobResolution.joinPath(sharedBlobStoreDirectory, directoryPrefix), contentDigest);
        return canonicalResolvedTargetPath == prefixedStorePath
            || canonicalResolvedTargetPath == fullDigestStorePath;
    }

    private static func unavailableSharedBlobMetadata(
        requiredFileName: String, problem: String) -> ArtifactValidationError {
        return ArtifactValidationError.huggingFaceSharedBlobMetadataUnavailable(
            fileName: requiredFileName, problem: problem);
    }

    // MARK: - Shared filesystem helpers for the artifact-validation layer.

    static func joinPath(_ basePath: String, _ appendedComponent: String) -> String {
        if basePath.hasSuffix("/") {
            return basePath + appendedComponent;
        }
        return basePath + "/" + appendedComponent;
    }

    /// Resolves every symlink in the path, mirroring `fs::canonicalize` for
    /// paths that exist on disk.
    static func canonicalizedPath(_ path: String) -> String {
        return URL(fileURLWithPath: path, isDirectory: false).resolvingSymlinksInPath().path;
    }

    /// lstat-equivalent inspection: attributes describe the item itself, never
    /// a symlink target. nil when the path cannot be inspected.
    static func inspectedAttributes(atPath path: String) -> [FileAttributeKey: Any]? {
        do {
            return try FileManager.default.attributesOfItem(atPath: path);
        } catch {
            return nil;
        }
    }

    static func attributesDescribeSymlink(_ attributes: [FileAttributeKey: Any]) -> Bool {
        return (attributes[.type] as? FileAttributeType) == .typeSymbolicLink;
    }

    static func attributesDescribeDirectory(_ attributes: [FileAttributeKey: Any]) -> Bool {
        return (attributes[.type] as? FileAttributeType) == .typeDirectory;
    }

    static func attributesDescribeRegularFile(_ attributes: [FileAttributeKey: Any]) -> Bool {
        return (attributes[.type] as? FileAttributeType) == .typeRegular;
    }

    static func fileSize(fromAttributes attributes: [FileAttributeKey: Any]) -> UInt64 {
        return (attributes[.size] as? NSNumber)?.uint64Value ?? 0;
    }

    /// Component-wise prefix test matching Rust's `Path::starts_with`.
    static func isPath(_ candidatePath: String, withinDirectory directoryPath: String) -> Bool {
        if candidatePath == directoryPath {
            return true;
        }
        return candidatePath.hasPrefix(directoryPath + "/");
    }
}
