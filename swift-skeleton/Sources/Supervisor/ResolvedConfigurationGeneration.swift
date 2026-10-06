import Foundation;

import AstronomicalConfig;
import CryptoKit;

/// Derives a path-free identity for the complete resolved serving snapshot.
///
/// Migrates apps/supervisor/src/resolved_configuration_generation.rs: the
/// digest mixes the config document generation with the discovered models,
/// the ordered policy catalog (whose worker policy serializes through the
/// IPC wire format), and the unmatched configured identities. Field order is
/// normalized by sorting; encoding is length-prefixed little-endian.
public enum ResolvedConfigurationGeneration {

    public static func derive(
        documentGeneration: String,
        discoveredModels: Array<DiscoveryDiscoveredModel>,
        modelPolicyCatalog: Dictionary<String, RuntimeModelPolicy>,
        unmatchedModelConfigIds: Array<String>
    ) throws -> String {
        var digestBytes: Array<UInt8> = Array<UInt8>();
        ResolvedConfigurationGeneration.updateText(into: &digestBytes, text: documentGeneration);

        let orderedDiscoveredModels: Array<DiscoveryDiscoveredModel> = discoveredModels.sorted { (left: DiscoveryDiscoveredModel, right: DiscoveryDiscoveredModel) -> Bool in
            return left.modelId < right.modelId;
        };
        for discoveredModel: DiscoveryDiscoveredModel in orderedDiscoveredModels {
            ResolvedConfigurationGeneration.updateText(into: &digestBytes, text: discoveredModel.modelId);
            ResolvedConfigurationGeneration.updateText(into: &digestBytes, text: discoveredModel.revision);
            ResolvedConfigurationGeneration.updateText(
                into: &digestBytes,
                text: ResolvedConfigurationGeneration.modelFamilyIdentity(discoveredModel.modelFamily));
            ResolvedConfigurationGeneration.updateOptionalText(
                into: &digestBytes,
                text: discoveredModel.providerModelId);
            ResolvedConfigurationGeneration.updateOptionalText(
                into: &digestBytes,
                text: discoveredModel.license.map { (modelLicense: ModelLicense) -> String in
                    return modelLicense.spdxIdentifier;
                });
            ResolvedConfigurationGeneration.updateModelCapabilities(
                into: &digestBytes,
                modelCapabilities: discoveredModel.capabilities);
            ResolvedConfigurationGeneration.updateNumber(into: &digestBytes, number: discoveredModel.modelSizeBytes);
        }

        let orderedModelPolicyIds: Array<String> = modelPolicyCatalog.keys.sorted();
        for modelId: String in orderedModelPolicyIds {
            let modelPolicy: RuntimeModelPolicy = modelPolicyCatalog[modelId]!;
            ResolvedConfigurationGeneration.updateText(into: &digestBytes, text: modelId);
            ResolvedConfigurationGeneration.updateNumber(
                into: &digestBytes,
                number: UInt64(modelPolicy.generationDefaults.maximumOutputTokens));
            ResolvedConfigurationGeneration.updateOptionalNumber(
                into: &digestBytes,
                number: modelPolicy.generationDefaults.temperatureThousandths.map { (temperatureThousandths: UInt16) -> UInt64 in
                    return UInt64(temperatureThousandths);
                });
            ResolvedConfigurationGeneration.updateOptionalNumber(
                into: &digestBytes,
                number: modelPolicy.generationDefaults.topPThousandths.map { (topPThousandths: UInt16) -> UInt64 in
                    return UInt64(topPThousandths);
                });
            ResolvedConfigurationGeneration.updateBytes(
                into: &digestBytes,
                bytes: Array(try modelPolicy.workerModelConfiguration.serializedJsonBytes()));
        }

        for unmatchedModelId: String in unmatchedModelConfigIds {
            ResolvedConfigurationGeneration.updateText(into: &digestBytes, text: unmatchedModelId);
        }
        return ResolvedConfigurationGeneration.lowercaseHex(digestBytes);
    }

    /// Identifies the exact live state produced when only memory applies
    /// from a larger candidate.
    public static func deriveMemoryOnlyTransition(
        priorResolvedGeneration: String,
        maximumMlxMemoryBytes: UInt64?
    ) -> String {
        var digestBytes: Array<UInt8> = Array<UInt8>();
        ResolvedConfigurationGeneration.updateText(
            into: &digestBytes,
            text: "astronomical-memory-only-configuration-transition-v1");
        ResolvedConfigurationGeneration.updateText(into: &digestBytes, text: priorResolvedGeneration);
        ResolvedConfigurationGeneration.updateOptionalNumber(into: &digestBytes, number: maximumMlxMemoryBytes);
        return ResolvedConfigurationGeneration.lowercaseHex(digestBytes);
    }

    private static func modelFamilyIdentity(_ modelFamily: ModelFamily) -> String {
        return modelFamily.rawValue;
    }

    private static func updateModelCapabilities(
        into digestBytes: inout Array<UInt8>,
        modelCapabilities: DiscoveryModelCapabilities
    ) -> Void {
        switch (modelCapabilities) {
        case let .chat(capabilities):
            ResolvedConfigurationGeneration.updateText(into: &digestBytes, text: "chat");
            ResolvedConfigurationGeneration.updateNumber(into: &digestBytes, number: UInt64(capabilities.contextWindowTokens));
            ResolvedConfigurationGeneration.updateNumber(into: &digestBytes, number: UInt64(capabilities.maximumInputTokens));
            ResolvedConfigurationGeneration.updateNumber(into: &digestBytes, number: UInt64(capabilities.maximumOutputTokens));
            ResolvedConfigurationGeneration.updateBoolean(into: &digestBytes, state: capabilities.supportsVision);
            ResolvedConfigurationGeneration.updateBoolean(into: &digestBytes, state: capabilities.supportsReasoning);
            ResolvedConfigurationGeneration.updateBoolean(into: &digestBytes, state: capabilities.supportsToolCalls);
        case let .imageGeneration(capabilities):
            ResolvedConfigurationGeneration.updateText(into: &digestBytes, text: "image_generation");
            ResolvedConfigurationGeneration.updateBoolean(into: &digestBytes, state: capabilities.supportsTextToImage);
            ResolvedConfigurationGeneration.updateBoolean(into: &digestBytes, state: capabilities.supportsImageEditing);
            ResolvedConfigurationGeneration.updateBoolean(into: &digestBytes, state: capabilities.supportsMultipleReferenceImages);
        case let .embeddings(capabilities):
            ResolvedConfigurationGeneration.updateText(into: &digestBytes, text: "embeddings");
            ResolvedConfigurationGeneration.updateNumber(into: &digestBytes, number: UInt64(capabilities.vectorWidth));
            ResolvedConfigurationGeneration.updateNumber(into: &digestBytes, number: UInt64(capabilities.maximumInputTokens));
        }
    }

    private static func updateText(into digestBytes: inout Array<UInt8>, text: String) -> Void {
        ResolvedConfigurationGeneration.updateBytes(into: &digestBytes, bytes: Array(text.utf8));
    }

    private static func updateOptionalText(into digestBytes: inout Array<UInt8>, text: String?) -> Void {
        guard let presentText: String = text else {
            digestBytes.append(0);
            return;
        }
        digestBytes.append(1);
        ResolvedConfigurationGeneration.updateText(into: &digestBytes, text: presentText);
    }

    private static func updateBoolean(into digestBytes: inout Array<UInt8>, state: Bool) -> Void {
        digestBytes.append(state ? 1 : 0);
    }

    private static func updateBytes(into digestBytes: inout Array<UInt8>, bytes: Array<UInt8>) -> Void {
        var lengthBytes: Array<UInt8> = Array<UInt8>(repeating: 0, count: 8);
        var byteLength: UInt64 = UInt64(bytes.count);
        for lengthIndex: Int in 0..<8 {
            lengthBytes[lengthIndex] = UInt8(byteLength & 0xFF);
            byteLength >>= 8;
        }
        digestBytes.append(contentsOf: lengthBytes);
        digestBytes.append(contentsOf: bytes);
    }

    private static func updateNumber(into digestBytes: inout Array<UInt8>, number: UInt64) -> Void {
        var numberBytes: Array<UInt8> = Array<UInt8>(repeating: 0, count: 8);
        var numberValue: UInt64 = number;
        for numberIndex: Int in 0..<8 {
            numberBytes[numberIndex] = UInt8(numberValue & 0xFF);
            numberValue >>= 8;
        }
        digestBytes.append(contentsOf: numberBytes);
    }

    private static func updateOptionalNumber(into digestBytes: inout Array<UInt8>, number: UInt64?) -> Void {
        guard let presentNumber: UInt64 = number else {
            digestBytes.append(0);
            return;
        }
        digestBytes.append(1);
        ResolvedConfigurationGeneration.updateNumber(into: &digestBytes, number: presentNumber);
    }

    private static func lowercaseHex(_ bytes: Array<UInt8>) -> String {
        let shaDigest: SHA256Digest = SHA256.hash(data: Data(bytes));
        return shaDigest.map { (digestByte: UInt8) -> String in
            return String(format: "%02x", digestByte);
        }.joined();
    }
}
