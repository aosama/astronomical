import Foundation;

/// A cause-preserving failure while validating the complete Qwen3.5
/// artifact. Port of Qwen3_5ArtifactValidationError from
/// crates/model-serving/src/qwen3_5/artifacts/artifact.rs.
public enum Qwen35ArtifactValidationError: Error, Equatable {
    case artifact(ArtifactValidationError);
    case config(Qwen3_5ConfigError);
    case optiQMetadata(OptiQMetadataError);
    case shardIndex(Qwen3_5ArtifactError);

    /// Returns a bounded explanation suitable for a public model-load error.
    public func publicFailureReason() -> String {
        switch self {
        case .artifact(let validationError):
            return Self.boundedPublicFailureReason(
                "Qwen3.5 artifact validation failed: \(validationError.publicFailureReason())");
        case .config(let configError):
            return "Qwen3.5 config validation failed: \(configError.errorDescription ?? String(describing: configError))";
        case .optiQMetadata(let metadataError):
            return "Qwen3.5 OptiQ metadata validation failed: \(metadataError.errorDescription ?? String(describing: metadataError))";
        case .shardIndex(let shardIndexError):
            return "Qwen3.5 shard-index validation failed: \(shardIndexError.errorDescription ?? String(describing: shardIndexError))";
        }
    }

    /// Artifact paths and blob names stay out of the public reason: the
    /// bounded reason replaces path separators and truncates on a character
    /// boundary with an explicit ellipsis.
    private static func boundedPublicFailureReason(_ unboundedReason: String) -> String {
        let maximumPublicReasonCharacters: Int = 512;
        let sanitizedReason: String = unboundedReason
            .replacingOccurrences(of: "/", with: "_")
            .replacingOccurrences(of: "\\", with: "_");
        guard sanitizedReason.count > maximumPublicReasonCharacters else {
            return sanitizedReason;
        }
        var boundedReason: String = String(sanitizedReason.prefix(maximumPublicReasonCharacters));
        boundedReason.removeLast();
        boundedReason.append("…");
        return boundedReason;
    }
}
