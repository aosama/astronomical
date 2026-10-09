import Foundation;

import Testing;

import MLX;

import JourneyCategories;

import ModelServingTestSupport;

@testable import ModelServing;

/// Hermetic journeys for the persisted projected visual embeddings: save
/// and load round-trips on real BF16 tensors, oldest-first eviction under
/// the shared global quota, and corrupt-file untracking that accepts a
/// clean replacement. Port of the Rust
/// `persistent_prompt_cache_visual_embeddings` direct-MLX journeys.
extension MlxGpuJourneyContainer {

    @Suite(.tags(.hermeticMlxJourney))
    final class PersistentPromptCacheVisualEmbeddingTests {

        init() {
            signal(SIGPIPE, SIG_IGN);
            MLXMetallibLocator.overrideMetallibPathIfNecessary();
        }

    private static func visualEmbeddingModelContract()
        -> PersistentVisualEmbeddingModelContract {
        return PersistentVisualEmbeddingModelContract(
            modelId: PersistentPromptCacheFixture.ORNITH_MODEL_ID,
            modelRevision: PersistentPromptCacheFixture.ORNITH_MODEL_REVISION,
            projectedEmbeddingHiddenSize: 2_048,
            maximumVisualEmbeddingTokenCount: 4_096);
    }

    private static func visualEmbeddingKey(digestByte: UInt8)
        -> PersistentVisualEmbeddingKey {
        return PersistentVisualEmbeddingKey.forImage(
            encodedImageSha256: Data([UInt8](repeating: digestByte, count: 32)),
            modelId: PersistentPromptCacheFixture.ORNITH_MODEL_ID,
            modelRevision: PersistentPromptCacheFixture.ORNITH_MODEL_REVISION);
    }

    private static func zeroVisualEmbeddings(rowCount: Int) -> MLXArray {
        return MLXArray.zeros([rowCount, 2_048], type: Float.self).asType(.bfloat16);
    }

    @Test
    func should_save_and_load_one_visual_embedding_file() throws {
        let globalRoot: URL = FileManager.default.temporaryDirectory
            .appendingPathComponent("prompt-cache-\(UUID().uuidString)", isDirectory: true);
        try FileManager.default.createDirectory(
            at: globalRoot, withIntermediateDirectories: true);
        let diskStore: PersistentPromptCacheDiskStore = try PersistentPromptCacheFixture
            .openDiskStore(
                globalRoot: globalRoot,
                modelContract: try PersistentPromptCacheFixture.ornithModelContract());
        let visualEmbeddingKey: PersistentVisualEmbeddingKey = Self.visualEmbeddingKey(
            digestByte: 7);
        let visualEmbeddings: MLXArray = Self.zeroVisualEmbeddings(rowCount: 2);

        try diskStore.saveVisualEmbedding(
            visualEmbeddingKey: visualEmbeddingKey, visualEmbeddings: visualEmbeddings);

        #expect(diskStore.visualEmbeddingCount() == 1);
        #expect(diskStore.visualEmbeddingTotalSizeBytes() > 0);
        #expect(diskStore.hasVisualEmbedding(
            visualEmbeddingHash: visualEmbeddingKey.visualEmbeddingHash));
        #expect(diskStore.visualEmbeddingTotalSizeBytes() == diskStore.totalSizeBytes(),
            "a store holding only visual embeddings must account their bytes as its total");
        let loadedVisualEmbeddings: MLXArray? = try diskStore.loadVisualEmbedding(
            visualEmbeddingKey: visualEmbeddingKey,
            modelContract: Self.visualEmbeddingModelContract());
        #expect(loadedVisualEmbeddings?.shape == [2, 2_048]);
        #expect(loadedVisualEmbeddings?.dtype == .bfloat16);

        let rescannedStore: PersistentPromptCacheDiskStore = try PersistentPromptCacheFixture
            .openDiskStore(
                globalRoot: globalRoot,
                modelContract: try PersistentPromptCacheFixture.ornithModelContract());
        // The engine's model loading runs the visual scan right after the
        // store opens; mirror that order before asserting recovery.
        try rescannedStore.scanVisualEmbeddings(
            modelContract: Self.visualEmbeddingModelContract());
        #expect(rescannedStore.visualEmbeddingCount() == 1,
            "the rescan must recover the visual embedding from disk");
    }

    @Test
    func should_evict_the_oldest_visual_embedding_under_shared_quota_pressure() throws {
        let globalRoot: URL = FileManager.default.temporaryDirectory
            .appendingPathComponent("prompt-cache-\(UUID().uuidString)", isDirectory: true);
        try FileManager.default.createDirectory(
            at: globalRoot, withIntermediateDirectories: true);
        let firstVisualEmbeddingKey: PersistentVisualEmbeddingKey = Self.visualEmbeddingKey(
            digestByte: 7);
        let secondVisualEmbeddingKey: PersistentVisualEmbeddingKey = Self.visualEmbeddingKey(
            digestByte: 8);
        let oneVisualEmbeddingQuotaBytes: UInt64 = UInt64(8 * 2_048 * 2) + 17 * 1024;
        let diskStore: PersistentPromptCacheDiskStore = try PersistentPromptCacheFixture
            .openDiskStore(
                globalRoot: globalRoot,
                modelContract: try PersistentPromptCacheFixture.ornithModelContract(),
                globalPromptCacheMaximumSizeBytes: oneVisualEmbeddingQuotaBytes);

        try diskStore.saveVisualEmbedding(
            visualEmbeddingKey: firstVisualEmbeddingKey,
            visualEmbeddings: Self.zeroVisualEmbeddings(rowCount: 8));
        Thread.sleep(forTimeInterval: 0.05);
        try diskStore.saveVisualEmbedding(
            visualEmbeddingKey: secondVisualEmbeddingKey,
            visualEmbeddings: Self.zeroVisualEmbeddings(rowCount: 8));

        #expect(diskStore.visualEmbeddingCount() == 1);
        #expect(diskStore.hasVisualEmbedding(
            visualEmbeddingHash: firstVisualEmbeddingKey.visualEmbeddingHash) == false);
        #expect(diskStore.hasVisualEmbedding(
            visualEmbeddingHash: secondVisualEmbeddingKey.visualEmbeddingHash));
        #expect(diskStore.totalSizeBytes() <= oneVisualEmbeddingQuotaBytes);
    }

    @Test
    func should_untrack_invalid_visual_embedding_load_and_accept_replacement() throws {
        let globalRoot: URL = FileManager.default.temporaryDirectory
            .appendingPathComponent("prompt-cache-\(UUID().uuidString)", isDirectory: true);
        try FileManager.default.createDirectory(
            at: globalRoot, withIntermediateDirectories: true);
        let diskStore: PersistentPromptCacheDiskStore = try PersistentPromptCacheFixture
            .openDiskStore(
                globalRoot: globalRoot,
                modelContract: try PersistentPromptCacheFixture.ornithModelContract());
        let visualEmbeddingKey: PersistentVisualEmbeddingKey = Self.visualEmbeddingKey(
            digestByte: 9);
        let visualEmbeddings: MLXArray = Self.zeroVisualEmbeddings(rowCount: 2);
        try diskStore.saveVisualEmbedding(
            visualEmbeddingKey: visualEmbeddingKey, visualEmbeddings: visualEmbeddings);
        let visualEmbeddingFilePath: URL = diskStore.visualEmbeddingsDirectory
            .appendingPathComponent(PersistentPromptCacheStoreFile.hexEncode(
                visualEmbeddingKey.visualEmbeddingHash) + ".safetensors");
        try Data("not a safetensors file".utf8).write(to: visualEmbeddingFilePath);

        #expect(throws: PersistentPromptCacheDiskStoreError.self) {
            _ = try diskStore.loadVisualEmbedding(
                visualEmbeddingKey: visualEmbeddingKey,
                modelContract: Self.visualEmbeddingModelContract());
        }
        #expect(diskStore.visualEmbeddingCount() == 0,
            "the corrupt visual embedding must leave the index after a failed load");

        try diskStore.saveVisualEmbedding(
            visualEmbeddingKey: visualEmbeddingKey, visualEmbeddings: visualEmbeddings);
        let replacementEmbeddings: MLXArray? = try diskStore.loadVisualEmbedding(
            visualEmbeddingKey: visualEmbeddingKey,
            modelContract: Self.visualEmbeddingModelContract());
        #expect(replacementEmbeddings?.shape == [2, 2_048]);
        #expect(replacementEmbeddings?.dtype == .bfloat16);
    }
    }
}
