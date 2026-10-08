import Foundation;

import MLX;

import RuntimeIntegration;

/// Saves, loads, and scans persisted projected visual embeddings, port of
/// the Rust `disk_store_visual_embeddings`. The published file is one
/// BF16 tensor named `visual_embeddings` whose metadata re-derives the
/// filename digest, so a renamed or cross-model file can never load.
extension PersistentPromptCacheDiskStore {

    /// Whether one visual embedding is tracked, verifying the tracked file
    /// still exists and untracking stale entries immediately.
    public func hasVisualEmbedding(visualEmbeddingHash: Data) -> Bool {
        let trackedFilePath: String?;
        self.stateLock.lock();
        trackedFilePath = self.trackedFiles.visualEmbeddingFile(
            fileHash: visualEmbeddingHash)?.filePath;
        self.stateLock.unlock();
        guard let trackedFilePath: String = trackedFilePath
        else {
            return false;
        }
        if FileManager.default.fileExists(atPath: trackedFilePath) {
            return true;
        }
        self.untrackVisualEmbeddingAndSubtractAccounting(
            visualEmbeddingHash: visualEmbeddingHash);
        return false;
    }

    /// Publishes one projected visual embedding durably: write to a
    /// temporary name, rename onto the final digest name, track, and only
    /// then admit global quota (evicting oldest content when required). A
    /// failure after the rename deletes the saved file before surfacing.
    public func saveVisualEmbedding(
        visualEmbeddingKey: PersistentVisualEmbeddingKey,
        visualEmbeddings: MLXArray
    ) throws {
        self.writeOperationsLock.lock();
        defer { self.writeOperationsLock.unlock(); }
        try self.prepareActiveModelStorageDirectories();
        let savedFilePath: URL = try Self.writeVisualEmbeddingSafetensorsFile(
            visualEmbeddingsDirectory: self.visualEmbeddingsDirectory,
            visualEmbeddingKey: visualEmbeddingKey,
            visualEmbeddings: visualEmbeddings,
            modelId: self.modelContract.modelId,
            modelRevision: self.modelContract.modelRevision);
        let savedFileSizeBytes: UInt64 = (try? PersistentPromptCacheDiskStoreScan
            .cacheOwnedFileByteCount(filePath: savedFilePath)) ?? 0;
        if savedFileSizeBytes > self.globalPromptCacheMaximumSizeBytes {
            try? PersistentPromptCacheStoreFile.removeCacheOwnedFileOrConfirmAbsent(
                filePath: savedFilePath);
            throw PersistentPromptCacheDiskStoreError.sizeBoundExceeded(
                maximumSizeBytes: self.globalPromptCacheMaximumSizeBytes,
                estimatedBlockBytes: savedFileSizeBytes);
        }
        self.stateLock.lock();
        self.trackedFiles.insertVisualEmbedding(
            fileHash: visualEmbeddingKey.visualEmbeddingHash,
            trackedFile: PersistentPromptCacheDiskStoreIndex.TrackedFile(
                filePath: savedFilePath.path, fileSizeBytes: savedFileSizeBytes));
        self.stateLock.unlock();
        do {
            try self.enforceGlobalPromptCacheQuota();
        } catch {
            try? PersistentPromptCacheStoreFile.removeCacheOwnedFileOrConfirmAbsent(
                filePath: savedFilePath);
            self.untrackVisualEmbeddingAndSubtractAccounting(
                visualEmbeddingHash: visualEmbeddingKey.visualEmbeddingHash);
            throw error;
        }
    }

    /// Loads one persisted visual embedding's tensor by identity, or nil
    /// when the identity is not tracked. A corrupt file is deleted and
    /// untracked before the typed validation failure is raised.
    public func loadVisualEmbedding(
        visualEmbeddingKey: PersistentVisualEmbeddingKey,
        modelContract: PersistentVisualEmbeddingModelContract
    ) throws -> MLXArray? {
        let visualEmbeddingHash: Data = visualEmbeddingKey.visualEmbeddingHash;
        let trackedFilePath: String?;
        self.stateLock.lock();
        trackedFilePath = self.trackedFiles.visualEmbeddingFile(
            fileHash: visualEmbeddingHash)?.filePath;
        self.stateLock.unlock();
        guard let visualEmbeddingFilePath: String = trackedFilePath
        else {
            return nil;
        }
        let visualEmbeddingFileUrl: URL = URL(fileURLWithPath: visualEmbeddingFilePath);
        do {
            _ = try PersistentVisualEmbeddingFileHeader.readFromFile(
                visualEmbeddingFileUrl: visualEmbeddingFileUrl, modelContract: modelContract);
        } catch {
            try PersistentPromptCacheStoreFile.removeCacheOwnedFileOrConfirmAbsent(
                filePath: visualEmbeddingFileUrl);
            self.untrackVisualEmbeddingAndSubtractAccounting(
                visualEmbeddingHash: visualEmbeddingHash);
            throw PersistentPromptCacheDiskStoreError.validateModelSpecificArtifact(
                artifactFilePath: visualEmbeddingFilePath,
                problem: String(describing: error));
        }
        let loadedArrays: [String: MLXArray];
        do {
            loadedArrays = try MLX.loadArrays(url: visualEmbeddingFileUrl);
        } catch let loadError as MlxRuntimeError {
            throw PersistentPromptCacheDiskStoreError.loadSafetensors(source: loadError);
        } catch {
            throw PersistentPromptCacheDiskStoreError.loadSafetensors(
                source: .runtimeOperation(
                    operation: "load visual embedding",
                    description: String(describing: error)));
        }
        guard let visualEmbeddingTensor: MLXArray = loadedArrays[
            PersistentVisualEmbeddingFileHeader.TENSOR_NAME]
        else {
            throw PersistentPromptCacheDiskStoreError.loadSafetensors(
                source: .tensorLookupFailed(
                    tensorName: PersistentVisualEmbeddingFileHeader.TENSOR_NAME));
        }
        return visualEmbeddingTensor;
    }

    /// Startup scan for one active model: every visual embedding file whose
    /// closed-format header does not validate is removed with cleanup
    /// evidence before quota considers eviction.
    public func scanVisualEmbeddings(
        modelContract: PersistentVisualEmbeddingModelContract
    ) throws {
        var startupCleanupEvidence: PersistentPromptCacheStartupCleanupEvidence =
            PersistentPromptCacheStartupCleanupEvidence();
        // The index lock guards only the tracked-file mutation; quota
        // enforcement takes the same lock internally, so it must never run
        // under this call's lock.
        self.stateLock.lock();
        var scanFailure: Error? = nil;
        do {
            try PersistentPromptCacheDiskStoreScan.scanCurrentFormatDirectory(
                directory: self.visualEmbeddingsDirectory,
                trackedFiles: &self.trackedFiles,
                startupCleanupEvidence: &startupCleanupEvidence,
                headerValidator: { (visualEmbeddingFileUrl: URL) -> Bool in
                    return (try? PersistentVisualEmbeddingFileHeader.readFromFile(
                        visualEmbeddingFileUrl: visualEmbeddingFileUrl,
                        modelContract: modelContract)) != nil;
                });
        } catch {
            scanFailure = error;
        }
        self.stateLock.unlock();
        if let scanFailure: Error = scanFailure {
            throw scanFailure;
        }
        self.recordStartupCleanupEvidence(startupCleanupEvidence);
        try self.enforceStartupGlobalPromptCacheQuota(protectedBlockDirectoryPaths: []);
    }

    /// Writes one visual embedding safetensors file through a temporary
    /// name and renames it onto its final digest name, so readers only ever
    /// observe complete files.
    private static func writeVisualEmbeddingSafetensorsFile(
        visualEmbeddingsDirectory: URL,
        visualEmbeddingKey: PersistentVisualEmbeddingKey,
        visualEmbeddings: MLXArray,
        modelId: String,
        modelRevision: String
    ) throws -> URL {
        let visualEmbeddingShape: [Int] = visualEmbeddings.shape;
        if visualEmbeddingShape.count != 2 || visualEmbeddingShape[0] <= 0 {
            throw PersistentPromptCacheDiskStoreError.saveSafetensors(
                source: .runtimeOperation(
                    operation: "save visual embedding",
                    description: "visual embeddings must be rank-two with a positive "
                        + "token row count"));
        }
        let visualTokenCount: Int = visualEmbeddingShape[0];
        let finalFileName: String = PersistentPromptCacheStoreFile.hexEncode(
            visualEmbeddingKey.visualEmbeddingHash) + ".safetensors";
        let finalFileUrl: URL = visualEmbeddingsDirectory
            .appendingPathComponent(finalFileName);
        let temporaryFileUrl: URL = visualEmbeddingsDirectory
            .appendingPathComponent(finalFileName + ".tmp");
        try PersistentPromptCacheStoreFile.removeCacheOwnedFileOrConfirmAbsent(
            filePath: temporaryFileUrl);
        if FileManager.default.createFile(
            atPath: temporaryFileUrl.path, contents: Data(), attributes: nil) == false {
            throw PersistentPromptCacheDiskStoreError.openTempFile(
                tempFilePath: temporaryFileUrl.path,
                problem: "the temporary visual embedding file could not be created");
        }
        let outputFileHandle: FileHandle;
        do {
            outputFileHandle = try FileHandle(forWritingTo: temporaryFileUrl);
        } catch {
            throw PersistentPromptCacheDiskStoreError.openTempFile(
                tempFilePath: temporaryFileUrl.path, problem: String(describing: error));
        }
        defer { try? outputFileHandle.close(); }
        do {
            // Sorted-key compact JSON: "__metadata__" sorts before the
            // "visual_embeddings" tensor name.
            let encodedImageSha256Hex: String = PersistentPromptCacheStoreFile.hexEncode(
                visualEmbeddingKey.encodedImageSha256);
            let metadataJson: String = "{"
                + "\"encoded_image_sha256\":\"\(encodedImageSha256Hex)\","
                + "\"format_version\":\"\(PersistentVisualEmbeddingKey.FORMAT_VERSION)\","
                + "\"model_id\":\"\(modelId)\","
                + "\"model_revision\":\"\(modelRevision)\","
                + "\"visual_token_count\":\"\(visualTokenCount)\""
                + "}";
            let materializedEmbeddings: MLXArray = visualEmbeddings.contiguous();
            materializedEmbeddings.eval();
            let payloadBytes: Data = materializedEmbeddings
                .asData(access: .noCopyIfContiguous).data;
            let payloadByteCount: UInt64 = UInt64(payloadBytes.count);
            let headerJson: String = "{"
                + "\"__metadata__\":\(metadataJson),"
                + "\"visual_embeddings\":{"
                + "\"data_offsets\":[0,\(payloadByteCount)],"
                + "\"dtype\":\"BF16\","
                + "\"shape\":[\(visualTokenCount),\(visualEmbeddingShape[1])]}"
                + "}";
            let headerJsonBytes: Data = Data(headerJson.utf8);
            var headerLengthBytes: UInt64 = UInt64(headerJsonBytes.count);
            let headerLengthData: Data = withUnsafeBytes(
                of: &headerLengthBytes) { (valueBuffer: UnsafeRawBufferPointer) -> Data in
                var littleEndianBuffer: [UInt8] = [UInt8](repeating: 0, count: 8);
                for bufferIndex: Int in 0..<8 {
                    littleEndianBuffer[bufferIndex] = valueBuffer[bufferIndex];
                }
                return Data(littleEndianBuffer);
            };
            try outputFileHandle.write(contentsOf: headerLengthData);
            try outputFileHandle.write(contentsOf: headerJsonBytes);
            try outputFileHandle.write(contentsOf: payloadBytes);
            try outputFileHandle.synchronize();
        } catch let storeError as PersistentPromptCacheDiskStoreError {
            try? PersistentPromptCacheStoreFile.removeCacheOwnedFileOrConfirmAbsent(
                filePath: temporaryFileUrl);
            throw storeError;
        } catch {
            try? PersistentPromptCacheStoreFile.removeCacheOwnedFileOrConfirmAbsent(
                filePath: temporaryFileUrl);
            throw PersistentPromptCacheDiskStoreError.writeTempFile(
                tempFilePath: temporaryFileUrl.path, problem: String(describing: error));
        }
        do {
            try FileManager.default.moveItem(at: temporaryFileUrl, to: finalFileUrl);
        } catch {
            try? PersistentPromptCacheStoreFile.removeCacheOwnedFileOrConfirmAbsent(
                filePath: temporaryFileUrl);
            throw PersistentPromptCacheDiskStoreError.renameTempFile(
                tempFilePath: temporaryFileUrl.path,
                blockFilePath: finalFileUrl.path,
                problem: String(describing: error));
        }
        return finalFileUrl;
    }

    /// Removes one stale or evicted visual embedding from the index and
    /// subtracts its bytes from both global counters.
    private func untrackVisualEmbeddingAndSubtractAccounting(visualEmbeddingHash: Data) {
        self.stateLock.lock();
        defer { self.stateLock.unlock(); }
        guard let removedTrackedFile: PersistentPromptCacheDiskStoreIndex.TrackedFile =
            self.trackedFiles.removeVisualEmbedding(fileHash: visualEmbeddingHash)
        else {
            return;
        }
        self.globalPromptCacheTotalSizeBytes = self.globalPromptCacheTotalSizeBytes
            .subtractingReportingOverflow(removedTrackedFile.fileSizeBytes).partialValue;
        self.globalVisualEmbeddingTotalSizeBytes = self.globalVisualEmbeddingTotalSizeBytes
            .subtractingReportingOverflow(removedTrackedFile.fileSizeBytes).partialValue;
    }
}
