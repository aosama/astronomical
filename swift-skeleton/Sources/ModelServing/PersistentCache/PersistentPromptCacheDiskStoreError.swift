import Foundation;

import RuntimeIntegration;

/// Bounded failure surface of the persistent prompt-cache disk store, port
/// of the Rust `PersistentPromptCacheDiskStoreError`. The enum grows with
/// each ported store concern; every case stays a typed, wire-safe
/// description of an untrusted on-disk artifact or an exhausted budget.
public enum PersistentPromptCacheDiskStoreError: Error, Equatable, Sendable {

    case readBlockManifest(manifestFilePath: String, problem: String);

    case parseBlockManifest(manifestFilePath: String, problem: String);

    case serializeBlockManifest(problem: String);

    case openTempFile(tempFilePath: String, problem: String);

    case writeTempFile(tempFilePath: String, problem: String);

    case synchronizeTempFile(tempFilePath: String, problem: String);

    case renameTempFile(tempFilePath: String, blockFilePath: String, problem: String);

    case invalidBlockManifest(manifestFilePath: String, description: String);

    case removeCacheOwnedFile(filePath: String, problem: String);

    case readPromptCacheDirectory(directoryPath: String, problem: String);

    case readBlockMetadata(blockFilePath: String, problem: String);

    case openBlockFile(blockFilePath: String, problem: String);

    case removePromptCacheFile(filePath: String, problem: String);

    case unsafePromptCacheDirectory(directoryPath: String);

    case createPromptCacheDirectory(directoryPath: String, problem: String);

    case activePromptCacheDirectoryOutsideGlobalRoot(
        activeModelPromptCacheDirectory: String,
        globalPromptCacheRootDirectory: String);

    case globalPromptCacheSizeOverflow(rootDirectory: String);

    case globalPromptCacheQuotaNotSatisfied(
        maximumSizeBytes: UInt64,
        remainingSizeBytes: UInt64);

    case existingBlockTopologyMismatch(blockHash: Data);

    case sizeBoundExceeded(maximumSizeBytes: UInt64, estimatedBlockBytes: UInt64);

    case writtenFileSizeMismatch(
        filePath: String,
        reportedSizeBytes: UInt64,
        actualSizeBytes: UInt64);

    case invalidRequestedBlockAncestry(blockIndex: UInt32);

    case parentStateNotPublished(blockIndex: UInt32);

    case validateBlock(blockFilePath: String, problem: String);

    /// Wraps an MLX writer failure from the capture/restore slice's direct
    /// publication path so retry classification can inspect the source.
    case saveSafetensors(source: MlxRuntimeError);

    case writeSafetensorsDescriptor(filePath: String, problem: String);

    /// The exact active-memory bytes a retry must release before this
    /// publication can succeed, when the failure was MLX active-memory
    /// pressure; every other failure is not retryable as memory pressure.
    public func activeMemoryDeficitBytes() -> UInt64? {
        guard case let .saveSafetensors(
            .activeMemoryLimitExceeded(
                activeMemoryBytes: activeMemoryBytes,
                attemptedAllocationBytes: attemptedAllocationBytes,
                allowedActiveMemoryBytes: allowedActiveMemoryBytes)) = self
        else {
            return nil;
        }
        let (requestedBytes, addOverflowed) = activeMemoryBytes
            .addingReportingOverflow(attemptedAllocationBytes);
        if addOverflowed {
            return UInt64.max;
        }
        let deficitBytes: Int = requestedBytes
            .subtractingReportingOverflow(allowedActiveMemoryBytes).partialValue;
        return UInt64(max(deficitBytes, 0));
    }

    public var errorDescription: String? {
        switch self {
        case let .readBlockManifest(manifestFilePath, problem):
            return "failed to read the prompt-cache block manifest at \(manifestFilePath): \(problem)";
        case let .parseBlockManifest(manifestFilePath, problem):
            return "failed to parse the prompt-cache block manifest at \(manifestFilePath): \(problem)";
        case let .serializeBlockManifest(problem):
            return "failed to serialize the prompt-cache block manifest: \(problem)";
        case let .openTempFile(tempFilePath, problem):
            return "failed to open the prompt-cache temporary file at \(tempFilePath): \(problem)";
        case let .writeTempFile(tempFilePath, problem):
            return "failed to write the prompt-cache temporary file at \(tempFilePath): \(problem)";
        case let .synchronizeTempFile(tempFilePath, problem):
            return "failed to synchronize the prompt-cache temporary file at \(tempFilePath): \(problem)";
        case let .renameTempFile(tempFilePath, blockFilePath, problem):
            return "failed to publish \(tempFilePath) onto \(blockFilePath): \(problem)";
        case let .invalidBlockManifest(manifestFilePath, description):
            return "invalid prompt-cache block manifest at \(manifestFilePath): \(description)";
        case let .removeCacheOwnedFile(filePath, problem):
            return "failed to remove the cache-owned file at \(filePath): \(problem)";
        case let .readPromptCacheDirectory(directoryPath, problem):
            return "failed to read the prompt-cache directory at \(directoryPath): \(problem)";
        case let .readBlockMetadata(blockFilePath, problem):
            return "failed to read prompt-cache file metadata at \(blockFilePath): \(problem)";
        case let .openBlockFile(blockFilePath, problem):
            return "failed to open the prompt-cache block file at \(blockFilePath): \(problem)";
        case let .removePromptCacheFile(filePath, problem):
            return "failed to remove the prompt-cache file at \(filePath): \(problem)";
        case let .unsafePromptCacheDirectory(directoryPath):
            return "the prompt-cache directory at \(directoryPath) is not a safe cache-owned path";
        case let .createPromptCacheDirectory(directoryPath, problem):
            return "failed to create the prompt-cache directory at \(directoryPath): \(problem)";
        case let .activePromptCacheDirectoryOutsideGlobalRoot(
            activeModelPromptCacheDirectory, globalPromptCacheRootDirectory):
            return "the active model prompt-cache directory \(activeModelPromptCacheDirectory) "
                + "lies outside the global root \(globalPromptCacheRootDirectory)";
        case let .globalPromptCacheSizeOverflow(rootDirectory):
            return "the global prompt-cache byte total overflowed at \(rootDirectory)";
        case let .globalPromptCacheQuotaNotSatisfied(maximumSizeBytes, remainingSizeBytes):
            return "the global prompt-cache quota of \(maximumSizeBytes) bytes could not be "
                + "satisfied; \(remainingSizeBytes) bytes remain";
        case let .existingBlockTopologyMismatch(blockHash):
            return "a committed block for hash "
                + blockHash.map({ (hashByte: UInt8) -> String in
                    return String(format: "%02x", hashByte);
                }).joined()
                + " already exists with different topology";
        case let .sizeBoundExceeded(maximumSizeBytes, estimatedBlockBytes):
            return "the staged artifact is \(estimatedBlockBytes) bytes, exceeding its "
                + "\(maximumSizeBytes)-byte bound";
        case let .writtenFileSizeMismatch(filePath, reportedSizeBytes, actualSizeBytes):
            return "the staged file at \(filePath) reports \(reportedSizeBytes) bytes but wrote "
                + "\(actualSizeBytes)";
        case let .invalidRequestedBlockAncestry(blockIndex):
            return "the requested block at index \(blockIndex) has no valid parent ancestry";
        case let .parentStateNotPublished(blockIndex):
            return "the parent of the requested block at index \(blockIndex) is not published";
        case let .validateBlock(blockFilePath, problem):
            return "the block file at \(blockFilePath) failed validation: \(problem)";
        case let .saveSafetensors(source):
            return "the safetensors state write failed: \(source)";
        case let .writeSafetensorsDescriptor(filePath, problem):
            return "the safetensors descriptor at \(filePath) could not be written: \(problem)";
        }
    }
}
