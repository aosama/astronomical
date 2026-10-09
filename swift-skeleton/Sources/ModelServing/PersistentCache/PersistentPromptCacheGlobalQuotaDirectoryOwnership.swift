import Foundation;


/// One global prompt-cache byte ceiling across every model and revision,
/// port of the Rust `disk_store_global_quota` directory-ownership section.
/// The quota owner operates on scan-produced units: abandoned transactions
/// are always removed first, committed blocks are removed only as complete
/// subtrees, and directories belonging to the active publication ancestry
/// are protected.
enum PersistentPromptCacheGlobalQuota {

    private static let RETIRED_SPECULATIVE_PREFILL_CACHE_DIRECTORIES: [String] = [
        "speculative_prefill_selections",
        "speculative_prefill_target_states",
    ];

    /// Reclaims retired speculative-prefill directories as obsolete-format
    /// evidence during store startup, before quota considers deleting
    /// useful content.
    static func removeRetiredSpeculativePrefillCacheDirectories(
        activeModelPromptCacheDirectory: URL,
        startupCleanupEvidence: inout PersistentPromptCacheStartupCleanupEvidence
    ) throws {
        for retiredDirectoryName: String in RETIRED_SPECULATIVE_PREFILL_CACHE_DIRECTORIES {
            let retiredDirectoryPath: URL = activeModelPromptCacheDirectory
                .appendingPathComponent(retiredDirectoryName);
            let fileAttributes: [FileAttributeKey: Any];
            do {
                fileAttributes = try FileManager.default.attributesOfItem(
                    atPath: retiredDirectoryPath.path);
            } catch let statError as NSError
                where statError.code == NSFileNoSuchFileError
                    || statError.code == NSFileReadNoSuchFileError {
                continue;
            } catch {
                throw PersistentPromptCacheDiskStoreError.readBlockMetadata(
                    blockFilePath: retiredDirectoryPath.path,
                    problem: String(describing: error));
            }
            var isDirectory: ObjCBool = ObjCBool(false);
            FileManager.default.fileExists(
                atPath: retiredDirectoryPath.path, isDirectory: &isDirectory);
            let retiredBytes: UInt64;
            if isDirectory.boolValue {
                retiredBytes = try PersistentPromptCacheDiskStoreScan
                    .cacheOwnedDirectoryByteCount(directoryPath: retiredDirectoryPath);
                try PersistentPromptCacheStoreFile.removeCacheOwnedDirectoryOrConfirmAbsent(
                    directoryPath: retiredDirectoryPath);
            } else {
                retiredBytes = (fileAttributes[.size] as? NSNumber)?.uint64Value ?? 0;
                try PersistentPromptCacheStoreFile.removeCacheOwnedFileOrConfirmAbsent(
                    filePath: retiredDirectoryPath);
            }
            startupCleanupEvidence.recordArtifact(
                reason: .obsoleteFormat, removedByteCount: retiredBytes);
        }
    }

    /// Establishes the trusted directory tree: creation is itself the trust
    /// boundary, rejecting lexical escapes and verifying every created
    /// component is a real directory rather than following a symlink into
    /// user-owned data.
    static func preparePromptCacheDirectoryTree(
        globalPromptCacheRootDirectory: URL,
        activeModelPromptCacheDirectory: URL,
        activeModelStorageDirectories: [URL]
    ) throws {
        try Self.rejectParentDirectoryComponents(
            directoryPath: globalPromptCacheRootDirectory);
        try Self.rejectParentDirectoryComponents(
            directoryPath: activeModelPromptCacheDirectory);
        do {
            try FileManager.default.createDirectory(
                at: globalPromptCacheRootDirectory, withIntermediateDirectories: true);
        } catch {
            throw PersistentPromptCacheDiskStoreError.createPromptCacheDirectory(
                directoryPath: globalPromptCacheRootDirectory.path,
                problem: String(describing: error));
        }
        try Self.verifyRealDirectory(directoryPath: globalPromptCacheRootDirectory);
        let rootPathPrefix: String = globalPromptCacheRootDirectory.standardizedFileURL.path;
        let activeModelPath: String = activeModelPromptCacheDirectory.standardizedFileURL.path;
        if activeModelPath != rootPathPrefix
            && activeModelPath.hasPrefix(rootPathPrefix + "/") == false {
            throw PersistentPromptCacheDiskStoreError
                .activePromptCacheDirectoryOutsideGlobalRoot(
                    activeModelPromptCacheDirectory: activeModelPath,
                    globalPromptCacheRootDirectory: rootPathPrefix);
        }
        try Self.createDescendantDirectoriesWithoutSymlinks(
            globalPromptCacheRootDirectory: globalPromptCacheRootDirectory,
            descendantDirectory: activeModelPromptCacheDirectory);
        for activeModelStorageDirectory: URL in activeModelStorageDirectories {
            try Self.createDescendantDirectoriesWithoutSymlinks(
                globalPromptCacheRootDirectory: globalPromptCacheRootDirectory,
                descendantDirectory: activeModelStorageDirectory);
        }
    }

    static func rejectParentDirectoryComponents(
        directoryPath: URL
    ) throws {
        let pathComponents: [String] = (directoryPath.path as NSString).pathComponents;
        if pathComponents.contains("..") {
            throw PersistentPromptCacheDiskStoreError.unsafePromptCacheDirectory(
                directoryPath: directoryPath.path);
        }
    }

    /// Creates every missing component of `descendantDirectory` beneath the
    /// verified global root, refusing any pre-existing symlink component.
    private static func createDescendantDirectoriesWithoutSymlinks(
        globalPromptCacheRootDirectory: URL,
        descendantDirectory: URL
    ) throws {
        let rootPathPrefix: String = globalPromptCacheRootDirectory.standardizedFileURL.path;
        let descendantPath: String = descendantDirectory.standardizedFileURL.path;
        guard descendantPath.hasPrefix(rootPathPrefix + "/")
            || descendantPath == rootPathPrefix else {
            throw PersistentPromptCacheDiskStoreError
                .activePromptCacheDirectoryOutsideGlobalRoot(
                    activeModelPromptCacheDirectory: descendantPath,
                    globalPromptCacheRootDirectory: rootPathPrefix);
        }
        let descendantComponents: [String] = descendantPath.dropFirst(
            rootPathPrefix.count)
            .split(separator: "/", omittingEmptySubsequences: true)
            .map(String.init);
        var currentDirectory: URL = globalPromptCacheRootDirectory;
        for descendantComponent: String in descendantComponents {
            currentDirectory.appendPathComponent(descendantComponent);
            var isSymlink: Bool = false;
            if let fileAttributes: [FileAttributeKey: Any] = try? FileManager.default
                .attributesOfItem(atPath: currentDirectory.path) {
                isSymlink = (fileAttributes[.type] as? FileAttributeType) == .typeSymbolicLink;
            }
            if isSymlink {
                throw PersistentPromptCacheDiskStoreError.unsafePromptCacheDirectory(
                    directoryPath: currentDirectory.path);
            }
            do {
                try FileManager.default.createDirectory(
                    at: currentDirectory, withIntermediateDirectories: false);
            } catch let createError as NSError
                where createError.code == NSFileWriteFileExistsError {
                continue;
            } catch {
                throw PersistentPromptCacheDiskStoreError.createPromptCacheDirectory(
                    directoryPath: currentDirectory.path,
                    problem: String(describing: error));
            }
            try Self.verifyRealDirectory(directoryPath: currentDirectory);
        }
    }

    private static func verifyRealDirectory(
        directoryPath: URL
    ) throws {
        let fileAttributes: [FileAttributeKey: Any];
        do {
            fileAttributes = try FileManager.default.attributesOfItem(
                atPath: directoryPath.path);
        } catch {
            throw PersistentPromptCacheDiskStoreError.createPromptCacheDirectory(
                directoryPath: directoryPath.path, problem: String(describing: error));
        }
        if (fileAttributes[.type] as? FileAttributeType) != .typeDirectory {
            throw PersistentPromptCacheDiskStoreError.unsafePromptCacheDirectory(
                directoryPath: directoryPath.path);
        }
    }
}
