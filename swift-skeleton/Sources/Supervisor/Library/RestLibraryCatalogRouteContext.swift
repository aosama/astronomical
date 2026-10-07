import Foundation

import AstronomicalConfig

/**
 * The live state the catalog route answers from. The daemon supplies
 * providers that read the current discovery snapshot, the download
 * coordinator's validated publications and in-flight job, and the on-disk
 * destination directory for each catalog identity; the endpoint layer never
 * reaches into those subsystems directly.
 */
public struct RestLibraryCatalogRouteContext: Sendable {
    public let downloadCatalog: DownloadCatalog
    public let discoveredModelsProvider: @Sendable () -> Array<DiscoveryDiscoveredModel>
    public let validatedPublicationsProvider: @Sendable () -> Set<String>
    public let currentJobProvider: @Sendable () -> LibraryDownloadJobSummary?
    public let destinationDirectoryProvider: @Sendable (_ huggingfaceId: String) -> String?

    public init(
        downloadCatalog: DownloadCatalog,
        discoveredModelsProvider: @escaping @Sendable () -> Array<DiscoveryDiscoveredModel>,
        validatedPublicationsProvider: @escaping @Sendable () -> Set<String>,
        currentJobProvider: @escaping @Sendable () -> LibraryDownloadJobSummary?,
        destinationDirectoryProvider: @escaping @Sendable (_ huggingfaceId: String) -> String?
    ) {
        self.downloadCatalog = downloadCatalog
        self.discoveredModelsProvider = discoveredModelsProvider
        self.validatedPublicationsProvider = validatedPublicationsProvider
        self.currentJobProvider = currentJobProvider
        self.destinationDirectoryProvider = destinationDirectoryProvider
    }
}
