import Foundation;

/**
 * Resolution of `models--<org>--<repo>/snapshots/<hash>` cache entries, the
 * ported form of the Rust `model_discovery_huggingface_cache` and
 * `model_identity` helpers.
 */
internal enum DiscoveryHuggingFaceCache {
    /** A resolved snapshot plus the model identifier the cache layout encodes. */
    internal struct Snapshot {
        internal let modelId: String;
        internal let snapshotDirectory: FilePath;

        internal init(modelId: String, snapshotDirectory: FilePath) {
            self.modelId = modelId;
            self.snapshotDirectory = snapshotDirectory;
        }
    }

    private static let cacheDirectoryPrefix: String = "models--";
    private static let cacheDirectorySeparator: String = "--";
    private static let referenceFileNames: Array<String> = Array<String>(["main", "master"]);

    /**
     * Resolves a cache entry via `refs/main` then `refs/master`, falling back
     * to the newest-modified snapshot directory when no reference resolves.
     * Snapshot listing uses `lstat` metadata because the Rust walker treats a
     * symlinked snapshot entry as a directory entry without following it.
     */
    internal static func resolveCacheEntry(huggingFaceCacheDirectory: FilePath) -> Snapshot? {
        guard let decodedModelId: String = decodeCacheDirectoryName(
            directoryName: DiscoveryPathNavigation.lastComponentName(of: huggingFaceCacheDirectory) ?? ""
        ) else {
            return nil;
        }
        let snapshotsDirectory: FilePath = huggingFaceCacheDirectory.appending(component: "snapshots");
        guard DiscoveryPathNavigation.isExistingDirectory(path: snapshotsDirectory) else {
            return nil;
        }
        let leafModelId: String = leafModelId(ofDecodedModelId: decodedModelId);
        for referenceFileName: String in self.referenceFileNames {
            let referenceFile: FilePath = huggingFaceCacheDirectory
                .appending(component: "refs")
                .appending(component: referenceFileName);
            guard let referencedSnapshotName: String = self.readTrimmedFileContents(path: referenceFile) else {
                continue;
            }
            let referencedSnapshotDirectory: FilePath = snapshotsDirectory.appending(component: referencedSnapshotName);
            guard DiscoveryPathNavigation.isExistingDirectory(path: referencedSnapshotDirectory) else {
                continue;
            }
            return Snapshot(modelId: leafModelId, snapshotDirectory: referencedSnapshotDirectory);
        }
        let snapshotEntryNames: Array<String>;
        do {
            snapshotEntryNames = try self.directoryEntryNames(path: snapshotsDirectory);
        } catch {
            return nil;
        }
        var newestSnapshotDirectory: FilePath?;
        var newestSnapshotModificationNanoseconds: UInt64 = 0;
        for snapshotEntryName: String in snapshotEntryNames {
            let snapshotEntryPath: FilePath = snapshotsDirectory.appending(component: snapshotEntryName);
            guard
                let modificationNanoseconds: UInt64 = DiscoveryPathNavigation.modificationTimeNanoseconds(path: snapshotEntryPath),
                DiscoveryPathNavigation.isSymlinkTargetedDirectory(path: snapshotEntryPath)
            else {
                continue;
            }
            if newestSnapshotDirectory == nil || modificationNanoseconds > newestSnapshotModificationNanoseconds {
                newestSnapshotDirectory = snapshotEntryPath;
                newestSnapshotModificationNanoseconds = modificationNanoseconds;
            }
        }
        guard let resolvedSnapshotDirectory: FilePath = newestSnapshotDirectory else {
            return nil;
        }
        return Snapshot(modelId: leafModelId, snapshotDirectory: resolvedSnapshotDirectory);
    }

    /** Strips the `models--` prefix; the remainder is the organization when no `--` remains. */
    internal static func decodeCacheDirectoryName(directoryName: String) -> String? {
        guard directoryName.hasPrefix(self.cacheDirectoryPrefix) else {
            return nil;
        }
        let decodedModelId: String = String(directoryName.dropFirst(self.cacheDirectoryPrefix.count));
        if decodedModelId.isEmpty {
            return nil;
        }
        guard let firstSeparatorRange: Range<String.Index> = decodedModelId.range(of: self.cacheDirectorySeparator) else {
            return decodedModelId;
        }
        return String(decodedModelId[..<firstSeparatorRange.lowerBound]);
    }

    /** The repository name: the final component of the decoded identifier, mirroring `rsplit`. */
    internal static func leafModelId(ofDecodedModelId decodedModelId: String) -> String {
        guard let lastSeparatorIndex: String.Index = decodedModelId.lastIndex(of: "/") else {
            return decodedModelId;
        }
        return String(decodedModelId[decodedModelId.index(after: lastSeparatorIndex)...]);
    }

    private static func directoryEntryNames(path: FilePath) throws -> Array<String> {
        let entryNames: Array<String> = try FileManager.default.contentsOfDirectory(atPath: path.string);
        return entryNames.sorted();
    }

    private static func readTrimmedFileContents(path: FilePath) -> String? {
        guard let fileBytes: Data = FileManager.default.contents(atPath: path.string) else {
            return nil;
        }
        guard let fileText: String = String(data: fileBytes, encoding: .utf8) else {
            return nil;
        }
        let trimmedText: String = fileText.trimmingCharacters(in: .whitespacesAndNewlines);
        if trimmedText.isEmpty {
            return nil;
        }
        return trimmedText;
    }
}
