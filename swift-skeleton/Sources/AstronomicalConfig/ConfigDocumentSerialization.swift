import Foundation;

/// Typed-configuration serialization shared by the write-transaction modules.
/// The Rust code serializes through serde_json::to_vec_pretty; the Swift
/// equivalent serializes the already-typed UserConfigFile object tree.
internal enum ConfigDocumentSerialization {

    internal static func serializeConfigFileBytes(configFilePath: FilePath, userConfigFile: UserConfigFile) throws -> Data {
        let candidateJsonObject: Any = userConfigFile.toJsonObject();
        if JSONSerialization.isValidJSONObject(candidateJsonObject) == false {
            throw AstronomicalConfigError.serializeConfigFile(configFilePath: configFilePath);
        }
        do {
            let serializedCandidateBytes = try JSONSerialization.data(withJSONObject: candidateJsonObject, options: [.prettyPrinted, .sortedKeys]);
            return serializedCandidateBytes;
        } catch let serializationError as NSError {
            throw AstronomicalConfigError.writeConfigFile(configFilePath: configFilePath, underlyingError: serializationError);
        }
    }
}
