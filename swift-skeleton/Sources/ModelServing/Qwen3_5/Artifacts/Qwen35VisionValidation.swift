import Foundation;

/// Physical storage validated for a Qwen3.5 visual tower. Port of the
/// ValidatedVisionTowerStorage enum from
/// crates/model-serving/src/qwen3_5/artifacts/vision_validation.rs.
public enum ValidatedVisionTowerStorage: Equatable, Sendable {
    case absent;
    case embeddedInModelShards;
    case separateSidecar;

    public func hasValidatedVisionTower() -> Bool {
        return self != ValidatedVisionTowerStorage.absent;
    }

    public func hasSeparateSidecar() -> Bool {
        return self == ValidatedVisionTowerStorage.separateSidecar;
    }
}

/// Verifies that the shard index has either no visual tower or one complete
/// tower. Port of the vision_validation.rs free function of the same purpose.
public enum Qwen35VisionValidation {

    public static func validateVisionTowerInventory(
        shardIndex: Qwen3_5ShardIndex,
        visionConfig: Qwen3_5VisionConfig?) throws -> ValidatedVisionTowerStorage {
        let visionTensorNameToShardFileName: Array<(tensorName: String, shardFileName: String)> =
            shardIndex.visionTensorNameToShardFileName();
        if visionTensorNameToShardFileName.isEmpty {
            return ValidatedVisionTowerStorage.absent;
        }
        guard let resolvedVisionConfig: Qwen3_5VisionConfig = visionConfig else {
            throw Qwen3_5ArtifactError.missingVisionConfig;
        }

        var expectedVisionTensorNames: Set<String> = Set();
        for tensorProfile: TensorProfile in Qwen35VisionTensorSpec.visionTensorProfiles(
            visionConfig: resolvedVisionConfig) {
            expectedVisionTensorNames.insert(tensorProfile.name);
        }
        var actualVisionTensorNames: Set<String> = Set();
        for visionDeclaration: (tensorName: String, shardFileName: String) in visionTensorNameToShardFileName {
            actualVisionTensorNames.insert(visionDeclaration.tensorName);
        }
        // Rust reports the BTreeSet-difference minimum, i.e. the lexically
        // first disagreement; sorted().first preserves that choice.
        if let unexpectedTensorName: String = actualVisionTensorNames
            .subtracting(expectedVisionTensorNames).sorted().first {
            throw Qwen3_5ArtifactError.unexpectedVisionTensor(tensorName: unexpectedTensorName);
        }
        if let missingTensorName: String = expectedVisionTensorNames
            .subtracting(actualVisionTensorNames).sorted().first {
            throw Qwen3_5ArtifactError.missingVisionTensor(tensorName: missingTensorName);
        }

        let usesSeparateSidecar: Bool = visionTensorNameToShardFileName
            .contains(where: { (visionDeclaration: (tensorName: String, shardFileName: String)) -> Bool in
                return shardIndex.isVisionSidecarFile(shardFileName: visionDeclaration.shardFileName);
            });
        if usesSeparateSidecar {
            for visionDeclaration: (tensorName: String, shardFileName: String) in visionTensorNameToShardFileName {
                if shardIndex.isVisionSidecarFile(shardFileName: visionDeclaration.shardFileName) == false {
                    throw Qwen3_5ArtifactError.mixedVisionTensorStorage(
                        tensorName: visionDeclaration.tensorName,
                        shardFileName: visionDeclaration.shardFileName);
                }
            }
            return ValidatedVisionTowerStorage.separateSidecar;
        }

        let modelShardFileNames: Array<String> = shardIndex.modelShardFileNames();
        for visionDeclaration: (tensorName: String, shardFileName: String) in visionTensorNameToShardFileName {
            if modelShardFileNames.contains(visionDeclaration.shardFileName) == false {
                throw Qwen3_5ArtifactError.visionTensorOutsideModelShards(
                    tensorName: visionDeclaration.tensorName,
                    shardFileName: visionDeclaration.shardFileName);
            }
        }
        return ValidatedVisionTowerStorage.embeddedInModelShards;
    }
}
