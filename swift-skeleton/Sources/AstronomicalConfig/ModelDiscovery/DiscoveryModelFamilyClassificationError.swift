import Foundation;

/**
 * Failures while reading the family marker documents (`config.json` or
 * `model_index.json`) before family dispatch, mirroring
 * `ModelFamilyClassificationError`.
 */
internal enum ModelFamilyClassificationError: Error, CustomStringConvertible {
    case readConfig(modelDirectory: FilePath, underlyingError: any Error);
    case parseConfig(modelDirectory: FilePath, underlyingDescription: String);
    case configTooLarge(actualBytes: UInt64, maximumBytes: UInt64);
    case readPipelineIndex(modelDirectory: FilePath, underlyingError: any Error);
    case parsePipelineIndex(modelDirectory: FilePath, underlyingDescription: String);
    case pipelineIndexTooLarge(actualBytes: UInt64, maximumBytes: UInt64);

    internal var description: String {
        switch (self) {
        case .readConfig(let modelDirectory, let underlyingError):
            return "failed to read the family configuration in \(modelDirectory): \(underlyingError)";
        case .parseConfig(let modelDirectory, let underlyingDescription):
            return "failed to parse the family configuration in \(modelDirectory): \(underlyingDescription)";
        case .configTooLarge(let actualBytes, let maximumBytes):
            return "family configuration is \(actualBytes) bytes, exceeding the \(maximumBytes) byte limit";
        case .readPipelineIndex(let modelDirectory, let underlyingError):
            return "failed to read the pipeline index in \(modelDirectory): \(underlyingError)";
        case .parsePipelineIndex(let modelDirectory, let underlyingDescription):
            return "failed to parse the pipeline index in \(modelDirectory): \(underlyingDescription)";
        case .pipelineIndexTooLarge(let actualBytes, let maximumBytes):
            return "pipeline index is \(actualBytes) bytes, exceeding the \(maximumBytes) byte limit";
        }
    }
}
