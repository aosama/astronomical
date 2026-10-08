import Foundation;

import ModelServing;

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
        }
    }
}
