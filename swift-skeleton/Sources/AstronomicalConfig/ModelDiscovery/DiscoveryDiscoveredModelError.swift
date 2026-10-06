import Foundation;

/** Failures while scanning a configured model root, mirroring `DiscoveredModelError`. */
public enum DiscoveryDiscoveredModelError: Error, CustomStringConvertible {
    case readDirectory(directoryPath: FilePath, underlyingError: any Error);
    case duplicateModelId(modelId: String, modelDirectories: Array<FilePath>);

    public var description: String {
        switch (self) {
        case .readDirectory(let directoryPath, let underlyingError):
            return "failed to read the model directory \(directoryPath): \(underlyingError)";
        case .duplicateModelId(let modelId, let modelDirectories):
            let directoryList: String = modelDirectories
                .map({ (directoryPath: FilePath) -> String in directoryPath.string })
                .joined(separator: ", ");
            return "duplicate model id \"\(modelId)\" found in directories: \(directoryList)";
        }
    }
}
