import Foundation;

import ModelServing;

/// Measured SSD space and prompt-cache blocks removed by one clear
/// operation, port of the Rust `PersistentPromptCacheClearOutcome`.
public struct PersistentPromptCacheClearOutcome: Equatable, Sendable {

    public let modelId: String?;

    public let blocksRemoved: UInt64;

    public let bytesFreed: UInt64;

    init(modelId: String?, blocksRemoved: UInt64, bytesFreed: UInt64) {
        self.modelId = modelId;
        self.blocksRemoved = blocksRemoved;
        self.bytesFreed = bytesFreed;
    }
}

/// Safe deletion of global or model-scoped persistent prompt-cache trees,
/// port of the Rust `clear_persistent_prompt_cache_directory`. The worker
/// is the sole owner of this operation; the supervisor sends an IPC command
/// and never reads or mutates cache files directly.
public enum PersistentPromptCacheDirectoryClear {

    /// Deletes either every cache namespace or all revisions of one model.
    ///
    /// Model identities may contain multiple normal path components, such
    /// as `organization/model`, because that is the cache layout used by
    /// model IDs. Absolute paths, `.`, and `..` are rejected before any
    /// filesystem access.
    public static func clearPersistentPromptCacheDirectory(
        globalPromptCacheRootDirectory: URL,
        modelId: String?
    ) throws -> PersistentPromptCacheClearOutcome {
        let clearTargetDirectory: URL? = try Self.clearTargetDirectory(
            globalPromptCacheRootDirectory: globalPromptCacheRootDirectory,
            modelId: modelId);
        guard let clearTargetDirectory: URL = clearTargetDirectory else {
            return PersistentPromptCacheClearOutcome(
                modelId: modelId, blocksRemoved: 0, bytesFreed: 0);
        }
        let (blocksRemoved, bytesFreed) = try Self.measureClearTarget(
            clearTargetDirectory: clearTargetDirectory);
        if modelId != nil {
            try PersistentPromptCacheStoreFile.removeCacheOwnedDirectoryOrConfirmAbsent(
                directoryPath: clearTargetDirectory);
        } else {
            try Self.removeGlobalRootContents(
                globalPromptCacheRootDirectory: clearTargetDirectory);
        }
        return PersistentPromptCacheClearOutcome(
            modelId: modelId, blocksRemoved: blocksRemoved, bytesFreed: bytesFreed);
    }

    private static func clearTargetDirectory(
        globalPromptCacheRootDirectory: URL,
        modelId: String?
    ) throws -> URL? {
        if try Self.verifyExistingRealDirectory(directoryPath: globalPromptCacheRootDirectory)
            == false {
            return nil;
        }
        guard let modelId: String = modelId else {
            return globalPromptCacheRootDirectory;
        }
        if modelId.isEmpty || modelId.contains("\0") || modelId.contains("\\") {
            throw PersistentPromptCacheDiskStoreError.unsafePromptCacheDirectory(
                directoryPath: modelId);
        }
        let modelPathComponents: [String] = modelId.split(separator: "/").map(String.init);
        if modelPathComponents.isEmpty {
            throw PersistentPromptCacheDiskStoreError.unsafePromptCacheDirectory(
                directoryPath: modelId);
        }
        var modelCacheDirectory: URL = globalPromptCacheRootDirectory;
        for pathComponent: String in modelPathComponents {
            if pathComponent == "." || pathComponent == ".." {
                throw PersistentPromptCacheDiskStoreError.unsafePromptCacheDirectory(
                    directoryPath: modelId);
            }
            modelCacheDirectory.appendPathComponent(pathComponent);
            if try Self.verifyDescendantComponentIsRealDirectory(
                componentDirectory: modelCacheDirectory) == false {
                return nil;
            }
        }
        return modelCacheDirectory;
    }

    private static func verifyDescendantComponentIsRealDirectory(
        componentDirectory: URL
    ) throws -> Bool {
        do {
            let fileAttributes: [FileAttributeKey: Any] = try FileManager.default
                .attributesOfItem(atPath: componentDirectory.path);
            guard (fileAttributes[.type] as? FileAttributeType) == .typeDirectory else {
                throw PersistentPromptCacheDiskStoreError.unsafePromptCacheDirectory(
                    directoryPath: componentDirectory.path);
            }
            return true;
        } catch let statError as NSError
            where statError.code == NSFileNoSuchFileError
                || statError.code == NSFileReadNoSuchFileError {
            return false;
        } catch let unsafeError as PersistentPromptCacheDiskStoreError {
            throw unsafeError;
        } catch {
            throw PersistentPromptCacheDiskStoreError.readBlockMetadata(
                blockFilePath: componentDirectory.path, problem: String(describing: error));
        }
    }

    private static func verifyExistingRealDirectory(
        directoryPath: URL
    ) throws -> Bool {
        do {
            let fileAttributes: [FileAttributeKey: Any] = try FileManager.default
                .attributesOfItem(atPath: directoryPath.path);
            guard (fileAttributes[.type] as? FileAttributeType) == .typeDirectory else {
                throw PersistentPromptCacheDiskStoreError.unsafePromptCacheDirectory(
                    directoryPath: directoryPath.path);
            }
            return true;
        } catch let statError as NSError
            where statError.code == NSFileNoSuchFileError
                || statError.code == NSFileReadNoSuchFileError {
            return false;
        } catch let unsafeError as PersistentPromptCacheDiskStoreError {
            throw unsafeError;
        } catch {
            throw PersistentPromptCacheDiskStoreError.readBlockMetadata(
                blockFilePath: directoryPath.path, problem: String(describing: error));
        }
    }

    private static func measureClearTarget(
        clearTargetDirectory: URL
    ) throws -> (blocksRemoved: UInt64, bytesFreed: UInt64) {
        var blocksRemoved: UInt64 = 0;
        var bytesFreed: UInt64 = 0;
        try Self.measureDirectoryContents(
            directoryPath: clearTargetDirectory,
            directoryContainsBlocks: false,
            blocksRemoved: &blocksRemoved,
            bytesFreed: &bytesFreed);
        return (blocksRemoved, bytesFreed);
    }

    private static func measureDirectoryContents(
        directoryPath: URL,
        directoryContainsBlocks: Bool,
        blocksRemoved: inout UInt64,
        bytesFreed: inout UInt64
    ) throws {
        let directoryEntries: [URL];
        do {
            directoryEntries = try FileManager.default.contentsOfDirectory(
                at: directoryPath, includingPropertiesForKeys: nil, options: []);
        } catch {
            throw PersistentPromptCacheDiskStoreError.readPromptCacheDirectory(
                directoryPath: directoryPath.path, problem: String(describing: error));
        }
        for entryPath: URL in directoryEntries {
            var isDirectory: ObjCBool = ObjCBool(false);
            let entryExists: Bool = FileManager.default.fileExists(
                atPath: entryPath.path, isDirectory: &isDirectory);
            if entryExists == false {
                continue;
            }
            if isDirectory.boolValue {
                if directoryContainsBlocks {
                    blocksRemoved = blocksRemoved &+ 1;
                }
                let childContainsBlocks: Bool = entryPath.lastPathComponent == "blocks";
                try Self.measureDirectoryContents(
                    directoryPath: entryPath,
                    directoryContainsBlocks: childContainsBlocks,
                    blocksRemoved: &blocksRemoved,
                    bytesFreed: &bytesFreed);
            } else {
                bytesFreed = bytesFreed &+ (try Self.cacheOwnedFileBytes(filePath: entryPath));
            }
        }
    }

    private static func cacheOwnedFileBytes(filePath: URL) throws -> UInt64 {
        do {
            let fileAttributes: [FileAttributeKey: Any] = try FileManager.default
                .attributesOfItem(atPath: filePath.path);
            return (fileAttributes[.size] as? NSNumber)?.uint64Value ?? 0;
        } catch {
            throw PersistentPromptCacheDiskStoreError.readBlockMetadata(
                blockFilePath: filePath.path, problem: String(describing: error));
        }
    }

    private static func removeGlobalRootContents(
        globalPromptCacheRootDirectory: URL
    ) throws {
        let directoryEntries: [URL];
        do {
            directoryEntries = try FileManager.default.contentsOfDirectory(
                at: globalPromptCacheRootDirectory, includingPropertiesForKeys: nil,
                options: []);
        } catch {
            throw PersistentPromptCacheDiskStoreError.readPromptCacheDirectory(
                directoryPath: globalPromptCacheRootDirectory.path,
                problem: String(describing: error));
        }
        for entryPath: URL in directoryEntries {
            var isDirectory: ObjCBool = ObjCBool(false);
            FileManager.default.fileExists(atPath: entryPath.path, isDirectory: &isDirectory);
            if isDirectory.boolValue {
                try PersistentPromptCacheStoreFile.removeCacheOwnedDirectoryOrConfirmAbsent(
                    directoryPath: entryPath);
            } else {
                try PersistentPromptCacheStoreFile.removeCacheOwnedFileOrConfirmAbsent(
                    filePath: entryPath);
            }
        }
    }
}
