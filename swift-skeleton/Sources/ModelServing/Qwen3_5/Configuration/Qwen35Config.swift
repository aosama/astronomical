import Foundation;
import IpcProtocol;

/// Validated Qwen3.5 text configuration used to derive execution shapes and
/// behavior. Port of crates/model-serving/src/qwen3_5/configuration/config.rs.

private let EXPECTED_MOE_ARCHITECTURE: String = "Qwen3_5MoeForConditionalGeneration";
private let EXPECTED_DENSE_ARCHITECTURE: String = "Qwen3_5ForConditionalGeneration";
private let EXPECTED_MOE_MODEL_TYPE: String = "qwen3_5_moe";
private let EXPECTED_DENSE_MODEL_TYPE: String = "qwen3_5";
private let EXPECTED_TORCH_DTYPE: String = "bfloat16";

/// Physical representation of executable weights.
///
/// Native BF16 tensors use ordinary dense MLX operations. Affine-quantized
/// tensors require packed-weight matrix multiplication with scales and biases.
public enum ModelWeightStorage: Equatable {
    case nativeBfloat16;
    case affineQuantized;
}

/// Feed-forward implementation declared by a validated Qwen3.5 checkpoint.
public enum Qwen3_5FeedForwardArchitecture: Equatable {
    case dense;
    case mixtureOfExperts;
}

public struct Qwen3_5Config: Equatable {
    var eosTokenIds: Array<UInt32>;
    var hasTiedEmbeddingsFlag: Bool;
    var quantizedModuleProfileMap: SortedModuleProfiles;
    var modelWeightStorageValue: ModelWeightStorage;
    var defaultQuantizationBitsValue: UInt32;
    var defaultQuantizationGroupSizeValue: UInt32;
    var textConfigValue: Qwen3_5TextConfig;
    var activationDtypeName: String;
    var feedForwardArchitectureValue: Qwen3_5FeedForwardArchitecture;

    /// Parses the config bytes retained by validated artifact ownership.
    public static func fromJsonBytes(configBytes: Array<UInt8>) throws -> Qwen3_5Config {
        let configWireValue: JsonWireValue;
        do {
            configWireValue = try JsonWireParser.parseDocument(
                documentBytes: Data(configBytes));
        } catch let jsonWireProblem as JsonWireProblem {
            throw Qwen3_5ConfigError.deserializeConfig(problem: jsonWireProblem.description);
        } catch {
            throw Qwen3_5ConfigError.deserializeConfig(problem: "malformed config JSON");
        }
        let configDocument: Qwen3_5ConfigDocument;
        do {
            configDocument = try Qwen3_5ConfigDocument.decoded(wireValue: configWireValue);
        } catch let jsonWireProblem as JsonWireProblem {
            throw Qwen3_5ConfigError.deserializeConfig(problem: jsonWireProblem.description);
        } catch {
            throw Qwen3_5ConfigError.deserializeConfig(problem: "unexpected config document shape");
        }
        let feedForwardArchitecture: Qwen3_5FeedForwardArchitecture;
        switch configDocument.modelType {
        case EXPECTED_DENSE_MODEL_TYPE:
            feedForwardArchitecture = .dense;
        case EXPECTED_MOE_MODEL_TYPE:
            feedForwardArchitecture = .mixtureOfExperts;
        case let unexpectedModelType:
            throw Qwen3_5ConfigError.invalidConfigValueDynamic(
                description: "model_type must be '\(EXPECTED_DENSE_MODEL_TYPE)' or '\(EXPECTED_MOE_MODEL_TYPE)', got '\(unexpectedModelType)'");
        }
        let expectedArchitecture: String;
        switch feedForwardArchitecture {
        case .dense:
            expectedArchitecture = EXPECTED_DENSE_ARCHITECTURE;
        case .mixtureOfExperts:
            expectedArchitecture = EXPECTED_MOE_ARCHITECTURE;
        }
        try Qwen3_5ConfigValidation.validateExactValue(
            fieldName: "architectures",
            actualValue: configDocument.architectures.joined(separator: ","),
            expectedValue: expectedArchitecture);
        // Resolve activation dtype: prefer top-level, fall back to text_config.dtype.
        // Qwen3.6 places dtype inside text_config instead of at the top level.
        let activationDtype: String = try configDocument.activationDtype
            ?? configDocument.textConfig.textConfigDtype
            ?? { throw Qwen3_5ConfigError.missingActivationDtype }();
        try Qwen3_5ConfigValidation.validateExactValue(
            fieldName: "dtype", actualValue: activationDtype, expectedValue: EXPECTED_TORCH_DTYPE);
        // Resolve eos_token_ids:
        // 1. Use top-level eos_token_id if present
        // 2. Otherwise fall back to text_config.eos_token_id
        // 3. Normalize single integers to arrays
        // 4. Append the Qwen chat EOS token (248046) if not already in the list
        // 5. Retain every declared stop token.
        let resolvedEosTokenIds: Array<UInt32> = try configDocument.eosTokenId
            ?? configDocument.textConfig.textConfigEosTokenId
            ?? { throw Qwen3_5ConfigError.missingEosTokenId }();
        var normalizedEosTokenIds: Array<UInt32> = resolvedEosTokenIds;
        if normalizedEosTokenIds.contains(Qwen3_5ConfigValidation.QWEN_CHAT_EOS_TOKEN_ID) == false {
            normalizedEosTokenIds.append(Qwen3_5ConfigValidation.QWEN_CHAT_EOS_TOKEN_ID);
        }
        if normalizedEosTokenIds.isEmpty {
            throw Qwen3_5ConfigError.invalidConfigValue(
                description: "eos_token_id must contain at least one token ID");
        }
        if normalizedEosTokenIds.count == 1,
            let padTokenId: UInt32 = configDocument.padTokenId,
            normalizedEosTokenIds.contains(padTokenId) == false {
            normalizedEosTokenIds.append(padTokenId);
        }
        try configDocument.textConfig.validate(feedForwardArchitecture: feedForwardArchitecture);
        var quantizedModuleProfiles: SortedModuleProfiles = SortedModuleProfiles(entries: Array());
        var modelWeightStorage: ModelWeightStorage = .nativeBfloat16;
        var defaultQuantizationBits: UInt32 = 0;
        var defaultQuantizationGroupSize: UInt32 = 0;
        switch (configDocument.quantization, configDocument.quantizationConfig) {
        case (nil, nil):
            break;
        case (let quantization?, nil), (nil, let quantization?):
            quantizedModuleProfiles = try quantization.validate(
                configSource: configDocument.textConfig,
                feedForwardArchitecture: feedForwardArchitecture);
            modelWeightStorage = .affineQuantized;
            defaultQuantizationBits = quantization.defaultBits();
            defaultQuantizationGroupSize = quantization.defaultGroupSize();
        case (let quantization?, let quantizationConfig?):
            if quantization != quantizationConfig {
                throw Qwen3_5ConfigError.quantizationCopiesDiffer;
            }
            quantizedModuleProfiles = try quantization.validate(
                configSource: configDocument.textConfig,
                feedForwardArchitecture: feedForwardArchitecture);
            modelWeightStorage = .affineQuantized;
            defaultQuantizationBits = quantization.defaultBits();
            defaultQuantizationGroupSize = quantization.defaultGroupSize();
        }
        return Qwen3_5Config(
            eosTokenIds: normalizedEosTokenIds,
            hasTiedEmbeddingsFlag: configDocument.tieWordEmbeddings,
            quantizedModuleProfileMap: quantizedModuleProfiles,
            modelWeightStorageValue: modelWeightStorage,
            defaultQuantizationBitsValue: defaultQuantizationBits,
            defaultQuantizationGroupSizeValue: defaultQuantizationGroupSize,
            textConfigValue: configDocument.textConfig,
            activationDtypeName: activationDtype,
            feedForwardArchitectureValue: feedForwardArchitecture);
    }
}
