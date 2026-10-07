import Foundation

import AstronomicalConfig

/**
 * The identity and state name of the single in-flight download job, which is
 * all the catalog join needs. Kept as its own small shape so the projection
 * stays independent of the durable job model's internals.
 */
public struct LibraryDownloadJobSummary: Equatable, Sendable {
    public let huggingfaceId: String
    public let stateName: String

    public init(huggingfaceId: String, stateName: String) {
        self.huggingfaceId = huggingfaceId
        self.stateName = stateName
    }
}

/** Readiness state of one catalog entry, computed once per request. */
public struct LibraryCatalogEntryProjection: Sendable {
    public let catalogEntry: DownloadCatalogEntry
    public let readyOnThisMac: Bool
    public let discoveredModelDirectory: String?
    public let requestableModelId: String?
    public let downloadState: String?
    public let capabilities: DownloadCatalogCapabilities
}

/**
 * Joins every catalog entry with discovery and publication state. Both the
 * REST catalog endpoint and the daemon IPC catalog request answer the same
 * question — is this entry installed on this Mac, and under which
 * requestable identifier? — so the join lives here once.
 *
 * An entry is ready when the model is discovered (provider identity and
 * revision both match) or when a validated download publication exists for
 * its Hugging Face identity.
 */
public enum LibraryCatalogProjection {

    public static func projectCatalogEntries(
        downloadCatalog: DownloadCatalog,
        discoveredModels: Array<DiscoveryDiscoveredModel>,
        validatedPublications: Set<String>,
        currentJob: LibraryDownloadJobSummary?
    ) -> Array<LibraryCatalogEntryProjection> {
        var projections: Array<LibraryCatalogEntryProjection> = Array()
        projections.reserveCapacity(downloadCatalog.entryCount)
        for catalogEntry: DownloadCatalogEntry in downloadCatalog.entries {
            let huggingfaceId: String = catalogEntry.huggingfaceId
            let discoveredModel: DiscoveryDiscoveredModel? = discoveredModels.first(
                where: { (candidateModel: DiscoveryDiscoveredModel) -> Bool in
                    return candidateModel.providerModelId == huggingfaceId
                        && candidateModel.revision == catalogEntry.revision
                })
            let hasValidatedPublication: Bool = validatedPublications.contains(huggingfaceId)
            let isReadyOnThisMac: Bool = discoveredModel != nil || hasValidatedPublication
            let requestableModelId: String? = isReadyOnThisMac
                ? discoveredModel?.modelId
                    ?? LibraryCatalogProjection.requestableModelIdFromHuggingfaceId(huggingfaceId)
                : nil
            let downloadState: String? = {
                if let currentJob: LibraryDownloadJobSummary = currentJob,
                    currentJob.huggingfaceId == huggingfaceId,
                    !isReadyOnThisMac
                {
                    return currentJob.stateName
                }
                return nil
            }()
            projections.append(LibraryCatalogEntryProjection(
                catalogEntry: catalogEntry,
                readyOnThisMac: isReadyOnThisMac,
                discoveredModelDirectory: discoveredModel?.modelDirectory.string,
                requestableModelId: requestableModelId,
                downloadState: downloadState,
                capabilities: catalogEntry.capabilities))
        }
        return projections
    }

    /**
     * Derives the local requestable model identifier from the Hugging Face
     * identity's leaf segment. Discovery publishes Library models under their
     * leaf directory name, so "org/Model-Name" becomes requestable as
     * "Model-Name".
     */
    public static func requestableModelIdFromHuggingfaceId(_ huggingfaceId: String) -> String {
        let leafSegment: Substring = huggingfaceId.split(separator: "/").last ?? Substring(huggingfaceId)
        return String(leafSegment)
    }
}
