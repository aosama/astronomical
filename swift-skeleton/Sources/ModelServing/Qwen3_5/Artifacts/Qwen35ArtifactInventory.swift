import Foundation;

/// Builds canonical tensor locations from the Qwen main index without sidecar
/// discovery. Port of
/// crates/model-serving/src/qwen3_5/artifacts/artifact_inventory.rs.
public enum Qwen35ArtifactInventory {

    /// Canonical locations for every language, MTP, and vision tensor the
    /// main index declares, all owned by the main-index declaration origin.
    public static func buildIndexTensorInventory(
        shardIndex: Qwen3_5ShardIndex) throws -> TensorInventory {
        let sourceIdByFileName: Dictionary<String, TensorSourceId> =
            try Qwen35ArtifactInventory.sourceIdByFileName(shardIndex: shardIndex);

        let inventory: TensorInventory = TensorInventory();
        var declarations: Array<(canonicalName: String, shardFileName: String, semanticRole: TensorSemanticRole)> = Array();
        for declaration: (tensorName: String, shardFileName: String) in shardIndex.languageTensorNameToShardFileName() {
            declarations.append((declaration.tensorName, declaration.shardFileName, .target));
        }
        for declaration: (tensorName: String, shardFileName: String) in shardIndex.visionTensorNameToShardFileName() {
            declarations.append((declaration.tensorName, declaration.shardFileName, .vision));
        }
        for declaration: (canonicalName: String, shardFileName: String, semanticRole: TensorSemanticRole) in declarations {
            guard let sourceId: TensorSourceId = sourceIdByFileName[declaration.shardFileName] else {
                throw ArtifactValidationError.profileMissingRequiredFile(
                    fileName: declaration.shardFileName);
            }
            do {
                try inventory.insert(location: TensorLocation(
                    canonicalName: declaration.canonicalName,
                    storedName: declaration.canonicalName,
                    sourceId: sourceId,
                    semanticRole: declaration.semanticRole,
                    declarationOrigin: .mainIndex));
            } catch {
                throw ArtifactValidationError.unexpectedTensor(tensorName: declaration.canonicalName);
            }
        }
        return inventory;
    }

    /// Deterministic source identities for every indexed shard file: names in
    /// lexical order numbered from one. Zero remains unused, while UInt32.max
    /// is reserved for the architecture sidecar; overflow returns a typed
    /// error instead of saturating into either identity.
    public static func sourceIdByFileName(
        shardIndex: Qwen3_5ShardIndex) throws -> Dictionary<String, TensorSourceId> {
        var sourceIdByFileName: Dictionary<String, TensorSourceId> = Dictionary();
        var uniqueFileNames: Set<String> = Set();
        for declaration: (tensorName: String, shardFileName: String) in shardIndex.languageTensorNameToShardFileName() {
            uniqueFileNames.insert(declaration.shardFileName);
        }
        for declaration: (tensorName: String, shardFileName: String) in shardIndex.visionTensorNameToShardFileName() {
            uniqueFileNames.insert(declaration.shardFileName);
        }
        for (sourcePosition, fileName): (Int, String) in uniqueFileNames.sorted().enumerated() {
            if sourcePosition >= Int(UInt32.max) {
                throw ArtifactValidationError.tensorSourceCountOverflow;
            }
            let (sourceNumber, positionOverflowed): (UInt32, Bool) =
                UInt32(sourcePosition).addingReportingOverflow(1);
            if positionOverflowed {
                throw ArtifactValidationError.tensorSourceCountOverflow;
            }
            if sourceNumber == UInt32.max {
                throw ArtifactValidationError.tensorSourceCountOverflow;
            }
            sourceIdByFileName[fileName] = TensorSourceId(sourceNumber: sourceNumber);
        }
        return sourceIdByFileName;
    }
}
