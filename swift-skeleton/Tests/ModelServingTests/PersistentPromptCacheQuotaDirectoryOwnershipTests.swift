import Foundation;

import Testing;

import ModelServing;

@testable import ModelServing;

/// Hermetic journeys for the quota layer's directory ownership: the
/// trusted tree is created beneath the global root, a symlinked component
/// is rejected instead of followed, retired speculative-prefill directories
/// are reclaimed as obsolete-format evidence, and a lexical escape in the
/// root path fails closed.
final class PersistentPromptCacheQuotaDirectoryOwnershipTests {

    @Test
    func should_prepare_the_trusted_tree_and_reclaim_retired_directories() throws {
        let globalRoot: URL = FileManager.default.temporaryDirectory
            .appendingPathComponent("quota-\(UUID().uuidString)", isDirectory: true);
        defer { try? FileManager.default.removeItem(at: globalRoot); }
        let activeModelDirectory: URL = globalRoot
            .appendingPathComponent("org/model-a/rev-1", isDirectory: true);
        let blocksDirectory: URL = activeModelDirectory.appendingPathComponent(
            "blocks", isDirectory: true);
        let visualEmbeddingsDirectory: URL = activeModelDirectory.appendingPathComponent(
            "visual_embeddings", isDirectory: true);
        // A retired speculative-prefill directory plus a retired loose file.
        let retiredDirectory: URL = activeModelDirectory.appendingPathComponent(
            "speculative_prefill_selections", isDirectory: true);
        try FileManager.default.createDirectory(
            at: retiredDirectory, withIntermediateDirectories: true);
        try Data(repeating: 3, count: 128).write(
            to: retiredDirectory.appendingPathComponent("selection.bin"));

        var cleanupEvidence: PersistentPromptCacheStartupCleanupEvidence =
            PersistentPromptCacheStartupCleanupEvidence();
        try PersistentPromptCacheGlobalQuota.removeRetiredSpeculativePrefillCacheDirectories(
            activeModelPromptCacheDirectory: activeModelDirectory,
            startupCleanupEvidence: &cleanupEvidence);
        try PersistentPromptCacheGlobalQuota.preparePromptCacheDirectoryTree(
            globalPromptCacheRootDirectory: globalRoot,
            activeModelPromptCacheDirectory: activeModelDirectory,
            activeModelStorageDirectories: [blocksDirectory, visualEmbeddingsDirectory]);

        #expect(cleanupEvidence.obsoleteFormat.artifactCount >= 1,
            "the retired speculative-prefill directory must be reclaimed");
        #expect(FileManager.default.fileExists(atPath: blocksDirectory.path));
        #expect(FileManager.default.fileExists(atPath: visualEmbeddingsDirectory.path));

        #expect(throws: (any Error).self) {
            try PersistentPromptCacheGlobalQuota.rejectParentDirectoryComponents(
                directoryPath: globalRoot.appendingPathComponent("../escape"));
        };
    }

    @Test
    func should_reject_a_symlinked_component_instead_of_following_it() throws {
        let globalRoot: URL = FileManager.default.temporaryDirectory
            .appendingPathComponent("quota-\(UUID().uuidString)", isDirectory: true);
        defer { try? FileManager.default.removeItem(at: globalRoot); }
        try FileManager.default.createDirectory(
            at: globalRoot, withIntermediateDirectories: true);
        let outsideDirectory: URL = FileManager.default.temporaryDirectory
            .appendingPathComponent("quota-outside-\(UUID().uuidString)", isDirectory: true);
        try FileManager.default.createDirectory(
            at: outsideDirectory, withIntermediateDirectories: true);
        defer { try? FileManager.default.removeItem(at: outsideDirectory); }
        let symlinkTarget: URL = globalRoot.appendingPathComponent("linked", isDirectory: true);
        try FileManager.default.createSymbolicLink(
            at: symlinkTarget, withDestinationURL: outsideDirectory);

        #expect(throws: (any Error).self) {
            try PersistentPromptCacheGlobalQuota.preparePromptCacheDirectoryTree(
                globalPromptCacheRootDirectory: globalRoot,
                activeModelPromptCacheDirectory: symlinkTarget
                    .appendingPathComponent("rev-1", isDirectory: true),
                activeModelStorageDirectories: []);
        };
        #expect(FileManager.default.fileExists(
            atPath: symlinkTarget.appendingPathComponent("rev-1").path) == false,
            "creation must never follow a symlinked component");
    }
}
