import Foundation;

/// A bounded structural mismatch in the Qwen3.5 shard index, port of the
/// Qwen3_5ArtifactError enum from crates/model-serving/src/qwen3_5/artifacts.
/// Later artifact-validation units extend this enum in place.
public enum Qwen3_5ArtifactError: Error, Equatable {
    case indexTooLarge(actualIndexBytes: Int, maximumIndexBytes: Int);
    case deserializeIndex(problem: String);
    case invalidTensorNameLength(tensorName: String, maximumTensorNameBytes: Int);
    case unexpectedLanguageTensor(tensorName: String);
    case missingLanguageTensor(tensorName: String);
    case unexpectedMtpTensor(tensorName: String);
    case missingMtpTensor(tensorName: String);
    case unexpectedVisionTensor(tensorName: String);
    case missingVisionTensor(tensorName: String);
    case missingVisionConfig;
    case mixedVisionTensorStorage(tensorName: String, shardFileName: String);
    case visionTensorOutsideModelShards(tensorName: String, shardFileName: String);

    public var errorDescription: String? {
        switch self {
        case .indexTooLarge(let actualIndexBytes, let maximumIndexBytes):
            return "Qwen3.5 shard index is \(actualIndexBytes) bytes, exceeding \(maximumIndexBytes)";
        case .deserializeIndex:
            return "failed to decode the Qwen3.5 shard index";
        case .invalidTensorNameLength(let tensorName, let maximumTensorNameBytes):
            return "invalid Qwen3.5 tensor name length for '\(tensorName)' (maximum \(maximumTensorNameBytes) bytes)";
        case .unexpectedLanguageTensor(let tensorName):
            return "Qwen3.5 index contains unexpected executable language tensor '\(tensorName)'";
        case .missingLanguageTensor(let tensorName):
            return "Qwen3.5 index is missing executable language tensor '\(tensorName)'";
        case .unexpectedMtpTensor(let tensorName):
            return "Qwen3.5 index contains unexpected MTP tensor '\(tensorName)'";
        case .missingMtpTensor(let tensorName):
            return "Qwen3.5 index is missing MTP tensor '\(tensorName)'";
        case .unexpectedVisionTensor(let tensorName):
            return "Qwen3.5 index contains unexpected vision tensor '\(tensorName)'";
        case .missingVisionTensor(let tensorName):
            return "Qwen3.5 index is missing vision tensor '\(tensorName)'";
        case .missingVisionConfig:
            return "Qwen3.5 index contains visual tensors but config.json has no vision_config";
        case .mixedVisionTensorStorage(let tensorName, let shardFileName):
            return "Qwen3.5 vision tensor '\(tensorName)' mixes sidecar and embedded storage through '\(shardFileName)'";
        case .visionTensorOutsideModelShards(let tensorName, let shardFileName):
            return "Qwen3.5 vision tensor '\(tensorName)' resolves outside loaded model shards through '\(shardFileName)'";
        }
    }
}
