import Foundation;

import Testing;

import ModelServing;

@testable import ModelServing;

/// Filesystem-safety journeys for the cache directory tree: symlinks never
/// resolve outside the cache root, symlinked path components and escaped
/// roots are rejected with typed errors, and a deleted active-model tree is
/// recreated by the next publication rather than followed or leaked into.
final class PersistentPromptCacheDirectorySafetyTests {

    @Test
    func should_recreate_deleted_active_model_directories_before_replacement_write() throws {
        let modelContract: PersistentPromptCacheModelContract = try PersistentPromptCacheFixture
            .ornithModelContract();
        let globalPromptCacheRoot: URL = FileManager.default.temporaryDirectory
            .appendingPathComponent("safety-\(UUID().uuidString)", isDirectory: true);
        defer { try? FileManager.default.removeItem(at: globalPromptCacheRoot); }
        let activeModelDirectory: URL = globalPromptCacheRoot
            .appendingPathComponent("active-model/revision", isDirectory: true);
        let diskStore: PersistentPromptCacheDiskStore = try PersistentPromptCacheFixture
            .openDiskStore(
                globalRoot: globalPromptCacheRoot,
                activeModelPromptCacheDirectory: activeModelDirectory,
                modelContract: modelContract);
        let promptTokens: [UInt32] = PersistentPromptCacheFixture
            .promptTokensWithCompleteBlocksAndTrailingTokens(
                modelContract: modelContract, completeBlockCount: 1, trailingTokenCount: 0);
        let blockKey: PersistentPromptCacheBlockKey = try PersistentPromptCacheBlockKey
            .forRootBlock(
                modelContract: modelContract,
                blockTokens: Array(promptTokens[..<modelContract.blockTokenCount]));
        let staging: PersistentPromptCacheStateFileStaging = PersistentPromptCacheFixture
            .SyntheticStateFileStaging();
        #expect(try diskStore.publishBlock(
            staging: staging, blockKey: blockKey, parentBlockKey: nil) == .published);

        try FileManager.default.removeItem(at: activeModelDirectory);

        // The stale live-index reset rides on the load path that lands with
        // the capture/restore slice; the publication itself must already
        // discard the stale entry when its directory vanished.
        #expect(try diskStore.publishBlock(
            staging: staging, blockKey: blockKey, parentBlockKey: nil) == .published,
            "the replacement write should recreate deleted cache directories");
        #expect(FileManager.default.fileExists(
            atPath: activeModelDirectory.appendingPathComponent(
                "blocks", isDirectory: true).path));
        #expect(FileManager.default.fileExists(
            atPath: Self.blockDirectoryPath(activeModelDirectory: activeModelDirectory,
                blockKey: blockKey)
                .appendingPathComponent(PersistentPromptCacheStoreFile.SEQUENCE_STATE_FILE_NAME)
                .path));
    }

    @Test
    func should_never_follow_a_global_prompt_cache_symlink_outside_the_root() throws {
        let globalPromptCacheRoot: URL = FileManager.default.temporaryDirectory
            .appendingPathComponent("safety-\(UUID().uuidString)", isDirectory: true);
        try FileManager.default.createDirectory(
            at: globalPromptCacheRoot, withIntermediateDirectories: true);
        defer { try? FileManager.default.removeItem(at: globalPromptCacheRoot); }
        let externalDirectory: URL = FileManager.default.temporaryDirectory
            .appendingPathComponent("safety-external-\(UUID().uuidString)", isDirectory: true);
        try FileManager.default.createDirectory(
            at: externalDirectory, withIntermediateDirectories: true);
        let externalFilePath: URL = externalDirectory.appendingPathComponent("must-remain.bin");
        try Data(count: 1_024).write(to: externalFilePath);
        let rootOwnedSymlinkPath: URL = globalPromptCacheRoot
            .appendingPathComponent("outside-link");
        try FileManager.default.createSymbolicLink(
            at: rootOwnedSymlinkPath, withDestinationURL: URL(fileURLWithPath: externalDirectory.path));

        _ = try PersistentPromptCacheFixture.openDiskStore(
            globalRoot: globalPromptCacheRoot,
            activeModelPromptCacheDirectory: globalPromptCacheRoot
                .appendingPathComponent("active-model/revision", isDirectory: true),
            modelContract: try PersistentPromptCacheFixture.ornithModelContract(),
            globalPromptCacheMaximumSizeBytes: 0);

        #expect(FileManager.default.fileExists(atPath: externalFilePath.path),
            "global quota enforcement must remove only the symlink itself");
        #expect((try? FileManager.default.attributesOfItem(
            atPath: rootOwnedSymlinkPath.path)) == nil);
        try? FileManager.default.removeItem(at: externalDirectory);
    }

    @Test
    func should_never_follow_an_active_model_safetensors_symlink() throws {
        let globalPromptCacheRoot: URL = FileManager.default.temporaryDirectory
            .appendingPathComponent("safety-\(UUID().uuidString)", isDirectory: true);
        try FileManager.default.createDirectory(
            at: globalPromptCacheRoot, withIntermediateDirectories: true);
        defer { try? FileManager.default.removeItem(at: globalPromptCacheRoot); }
        let activeModelKvBlocksDirectory: URL = globalPromptCacheRoot
            .appendingPathComponent("active-model/revision/kv_blocks", isDirectory: true);
        try FileManager.default.createDirectory(
            at: activeModelKvBlocksDirectory, withIntermediateDirectories: true);
        let externalDirectory: URL = FileManager.default.temporaryDirectory
            .appendingPathComponent("safety-external-\(UUID().uuidString)", isDirectory: true);
        try FileManager.default.createDirectory(
            at: externalDirectory, withIntermediateDirectories: true);
        let externalFilePath: URL = externalDirectory
            .appendingPathComponent("must-remain.safetensors");
        try Data("external".utf8).write(to: externalFilePath);
        let activeModelSymlinkPath: URL = activeModelKvBlocksDirectory
            .appendingPathComponent("\(String(repeating: "e", count: 64)).safetensors");
        try FileManager.default.createSymbolicLink(
            at: activeModelSymlinkPath, withDestinationURL: URL(fileURLWithPath: externalFilePath.path));

        _ = try PersistentPromptCacheFixture.openDiskStore(
            globalRoot: globalPromptCacheRoot,
            activeModelPromptCacheDirectory: globalPromptCacheRoot
                .appendingPathComponent("active-model/revision", isDirectory: true),
            modelContract: try PersistentPromptCacheFixture.ornithModelContract(),
            globalPromptCacheMaximumSizeBytes: 0);

        #expect(FileManager.default.fileExists(atPath: externalFilePath.path),
            "startup must remove the symlink itself without following it");
        #expect((try? FileManager.default.attributesOfItem(
            atPath: activeModelSymlinkPath.path)) == nil);
        try? FileManager.default.removeItem(at: externalDirectory);
    }

    @Test
    func should_reject_a_symlink_as_the_global_prompt_cache_root() throws {
        let configuredParentDirectory: URL = FileManager.default.temporaryDirectory
            .appendingPathComponent("safety-\(UUID().uuidString)", isDirectory: true);
        try FileManager.default.createDirectory(
            at: configuredParentDirectory, withIntermediateDirectories: true);
        defer { try? FileManager.default.removeItem(at: configuredParentDirectory); }
        let externalDirectory: URL = FileManager.default.temporaryDirectory
            .appendingPathComponent("safety-external-\(UUID().uuidString)", isDirectory: true);
        try FileManager.default.createDirectory(
            at: externalDirectory, withIntermediateDirectories: true);
        let externalMarkerFilePath: URL = externalDirectory.appendingPathComponent(
            "must-remain.bin");
        try Data("external".utf8).write(to: externalMarkerFilePath);
        let symlinkedRoot: URL = configuredParentDirectory
            .appendingPathComponent("cache-root");
        try FileManager.default.createSymbolicLink(
            at: symlinkedRoot, withDestinationURL: URL(fileURLWithPath: externalDirectory.path));

        #expect(throws: PersistentPromptCacheDiskStoreError.unsafePromptCacheDirectory(
            directoryPath: symlinkedRoot.path)) {
            _ = try PersistentPromptCacheFixture.openDiskStore(
                globalRoot: symlinkedRoot,
                activeModelPromptCacheDirectory: symlinkedRoot
                    .appendingPathComponent("active-model/revision", isDirectory: true),
                modelContract: try PersistentPromptCacheFixture.ornithModelContract(),
                globalPromptCacheMaximumSizeBytes: 0);
        };
        #expect(FileManager.default.fileExists(atPath: externalMarkerFilePath.path));
        #expect(FileManager.default.fileExists(
            atPath: externalDirectory.appendingPathComponent(
                "active-model", isDirectory: true).path) == false);
        try? FileManager.default.removeItem(at: externalDirectory);
    }

    @Test
    func should_reject_a_symlinked_active_model_directory_component() throws {
        let globalPromptCacheRoot: URL = FileManager.default.temporaryDirectory
            .appendingPathComponent("safety-\(UUID().uuidString)", isDirectory: true);
        try FileManager.default.createDirectory(
            at: globalPromptCacheRoot, withIntermediateDirectories: true);
        defer { try? FileManager.default.removeItem(at: globalPromptCacheRoot); }
        let externalDirectory: URL = FileManager.default.temporaryDirectory
            .appendingPathComponent("safety-external-\(UUID().uuidString)", isDirectory: true);
        try FileManager.default.createDirectory(
            at: externalDirectory, withIntermediateDirectories: true);
        let symlinkedModelDirectory: URL = globalPromptCacheRoot
            .appendingPathComponent("active-model");
        try FileManager.default.createSymbolicLink(
            at: symlinkedModelDirectory, withDestinationURL: URL(fileURLWithPath: externalDirectory.path));

        #expect(throws: PersistentPromptCacheDiskStoreError.unsafePromptCacheDirectory(
            directoryPath: symlinkedModelDirectory.path)) {
            _ = try PersistentPromptCacheFixture.openDiskStore(
                globalRoot: globalPromptCacheRoot,
                activeModelPromptCacheDirectory: symlinkedModelDirectory
                    .appendingPathComponent("revision", isDirectory: true),
                modelContract: try PersistentPromptCacheFixture.ornithModelContract(),
                globalPromptCacheMaximumSizeBytes: 0);
        };
        #expect(FileManager.default.fileExists(
            atPath: externalDirectory.appendingPathComponent(
                "revision", isDirectory: true).path) == false);
        try? FileManager.default.removeItem(at: externalDirectory);
    }

    @Test
    func should_reject_an_active_model_directory_outside_the_global_root() throws {
        let globalPromptCacheRoot: URL = FileManager.default.temporaryDirectory
            .appendingPathComponent("safety-\(UUID().uuidString)", isDirectory: true);
        try FileManager.default.createDirectory(
            at: globalPromptCacheRoot, withIntermediateDirectories: true);
        defer { try? FileManager.default.removeItem(at: globalPromptCacheRoot); }
        let unrelatedActiveModelDirectory: URL = FileManager.default.temporaryDirectory
            .appendingPathComponent("safety-unrelated-\(UUID().uuidString)", isDirectory: true);
        try FileManager.default.createDirectory(
            at: unrelatedActiveModelDirectory, withIntermediateDirectories: true);
        defer { try? FileManager.default.removeItem(at: unrelatedActiveModelDirectory); }

        #expect(
            throws: PersistentPromptCacheDiskStoreError.activePromptCacheDirectoryOutsideGlobalRoot(
                activeModelPromptCacheDirectory: unrelatedActiveModelDirectory.path,
                globalPromptCacheRootDirectory: globalPromptCacheRoot.path)) {
            _ = try PersistentPromptCacheFixture.openDiskStore(
                globalRoot: globalPromptCacheRoot,
                activeModelPromptCacheDirectory: unrelatedActiveModelDirectory,
                modelContract: try PersistentPromptCacheFixture.ornithModelContract(),
                globalPromptCacheMaximumSizeBytes: 0);
        };
    }

    @Test
    func should_reject_parent_directory_components_inside_the_active_model_path() throws {
        let globalPromptCacheRoot: URL = FileManager.default.temporaryDirectory
            .appendingPathComponent("safety-\(UUID().uuidString)", isDirectory: true);
        try FileManager.default.createDirectory(
            at: globalPromptCacheRoot, withIntermediateDirectories: true);
        defer { try? FileManager.default.removeItem(at: globalPromptCacheRoot); }
        let escapedActiveModelDirectory: URL = globalPromptCacheRoot
            .appendingPathComponent("model/../escaped-revision", isDirectory: true);

        #expect(throws: PersistentPromptCacheDiskStoreError.unsafePromptCacheDirectory(
            directoryPath: escapedActiveModelDirectory.path)) {
            _ = try PersistentPromptCacheFixture.openDiskStore(
                globalRoot: globalPromptCacheRoot,
                activeModelPromptCacheDirectory: escapedActiveModelDirectory,
                modelContract: try PersistentPromptCacheFixture.ornithModelContract(),
                globalPromptCacheMaximumSizeBytes: 0);
        };
        #expect(FileManager.default.fileExists(
            atPath: globalPromptCacheRoot.appendingPathComponent(
                "escaped-revision", isDirectory: true).path) == false);
    }

    private static func blockDirectoryPath(
        activeModelDirectory: URL, blockKey: PersistentPromptCacheBlockKey
    ) -> URL {
        return activeModelDirectory.appendingPathComponent("blocks", isDirectory: true)
            .appendingPathComponent(
                PersistentPromptCacheStoreFile.hexEncode(blockKey.blockHash()),
                isDirectory: true);
    }
}
