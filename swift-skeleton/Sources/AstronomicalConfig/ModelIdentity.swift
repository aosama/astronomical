import Foundation;

/// Model identifier helpers, porting crates/config/src/model_identity.rs.
internal enum ModelIdentity {

    internal static func resolveModelId(requestedModelId: String, knownModelIds: Array<String>) -> String {
        if knownModelIds.contains(requestedModelId) {
            return requestedModelId;
        }
        let splitModelIdParts = requestedModelId.split(separator: "/", maxSplits: 1, omittingEmptySubsequences: false);
        if splitModelIdParts.count == 2 {
            let providerStrippedModelId = String(splitModelIdParts[1]);
            if knownModelIds.contains(providerStrippedModelId) {
                return providerStrippedModelId;
            }
        }
        return requestedModelId;
    }

    /// The last path-style segment of a model identifier. Empty subsequences
    /// are preserved so a trailing slash yields an empty leaf exactly like
    /// Rust's rsplit('/').next().
    internal static func leafModelId(modelId: String) -> String {
        let leafCandidates = modelId.split(separator: "/", omittingEmptySubsequences: false);
        if let lastLeafCandidate = leafCandidates.last {
            return String(lastLeafCandidate);
        }
        return "";
    }

    internal static func nearModelMatches(requestedModelId: String, candidateModelIds: Array<String>) -> Array<String> {
        let normalizedRequestedModelId = requestedModelId.lowercased();
        let leadingTokenCandidates = normalizedRequestedModelId.split(
            omittingEmptySubsequences: false,
            whereSeparator: { (separatorCharacter: Character) -> Bool in
                return separatorCharacter == "/" || separatorCharacter == " " || separatorCharacter == "-";
            }
        );
        let leadingToken = leadingTokenCandidates.first.map { (leadingCandidate: Substring) -> String in String(leadingCandidate) } ?? "";
        var matchingCandidateModelIds: Array<String> = Array<String>();
        for candidateModelId in candidateModelIds {
            let normalizedCandidateModelId = candidateModelId.lowercased();
            if normalizedCandidateModelId.contains(normalizedRequestedModelId)
                || normalizedRequestedModelId.contains(normalizedCandidateModelId)
                || normalizedCandidateModelId.hasPrefix(leadingToken) {
                matchingCandidateModelIds.append(candidateModelId);
            }
        }
        matchingCandidateModelIds.sort();
        var deduplicatedCandidateModelIds: Array<String> = Array<String>();
        for candidateModelId in matchingCandidateModelIds {
            if deduplicatedCandidateModelIds.last == candidateModelId {
                continue;
            }
            deduplicatedCandidateModelIds.append(candidateModelId);
        }
        return Array(deduplicatedCandidateModelIds.prefix(3));
    }

    /// Decodes a Hugging Face cache directory name ("models--org--repo" or
    /// "models--org") into its model identifier ("org/repo" or "org").
    internal static func decodeHuggingfaceCacheDirectoryName(directoryName: String) -> String? {
        guard directoryName.hasPrefix("models--") else {
            return nil;
        }
        let decodedModelId = String(directoryName.dropFirst("models--".count));
        if decodedModelId.isEmpty {
            return nil;
        }
        let organizationAndRepositoryParts = decodedModelId.split(separator: "--", maxSplits: 1, omittingEmptySubsequences: false);
        let organizationPart = organizationAndRepositoryParts.count > 0 ? String(organizationAndRepositoryParts[0]) : "";
        if organizationAndRepositoryParts.count > 1 {
            let repositoryPart = String(organizationAndRepositoryParts[1]);
            return organizationPart + "/" + repositoryPart;
        }
        return organizationPart;
    }
}
