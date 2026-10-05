import Foundation;
import CryptoKit;

/// Canonical configuration fingerprint, porting
/// crates/config/src/configuration_generation.rs. The generation must be
/// independent of member order and whitespace, so the typed object tree is
/// serialized with recursively sorted keys in compact form before hashing.
internal enum ConfigurationGeneration {

    internal static func configurationGeneration(configFilePath: FilePath, userConfigFile: UserConfigFile, performanceAttributionEnabled: Bool = false) throws -> String {
        let passStart = ConfigPerformanceAttribution.startedPass(operationName: "configuration_generation", performanceAttributionEnabled: performanceAttributionEnabled);
        let canonicalJsonObject: Any = userConfigFile.toJsonObject();
        guard JSONSerialization.isValidJSONObject(canonicalJsonObject) else {
            ConfigPerformanceAttribution.finishedPass(operationName: "configuration_generation", passStart: passStart, passOutcome: "failure", performanceAttributionEnabled: performanceAttributionEnabled);
            throw AstronomicalConfigError.serializeConfigFile(configFilePath: configFilePath);
        }
        do {
            let canonicalJsonBytes = try JSONSerialization.data(withJSONObject: canonicalJsonObject, options: [.sortedKeys]);
            let sha256Digest = SHA256.hash(data: canonicalJsonBytes);
            let generationHex = sha256Digest.map { (digestByte: UInt8) -> String in String(format: "%02x", digestByte) }.joined();
            ConfigPerformanceAttribution.finishedPass(operationName: "configuration_generation", passStart: passStart, passOutcome: "success", performanceAttributionEnabled: performanceAttributionEnabled);
            return generationHex;
        } catch let serializationError as NSError {
            ConfigPerformanceAttribution.finishedPass(operationName: "configuration_generation", passStart: passStart, passOutcome: "failure", performanceAttributionEnabled: performanceAttributionEnabled);
            throw AstronomicalConfigError.writeConfigFile(configFilePath: configFilePath, underlyingError: serializationError);
        }
    }
}
