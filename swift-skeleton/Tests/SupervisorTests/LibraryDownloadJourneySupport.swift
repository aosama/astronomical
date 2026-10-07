import Foundation

import AstronomicalConfig
import IpcProtocol
import RestContract

@testable import Supervisor

/// The shared harness for the Library download REST journeys: the scripted
/// hub, coordinator, and serving route table over one temporary state tree,
/// with the restart seam and bounded poll helpers.
enum LibraryDownloadJourneyFailure: Error {
    case missingResponse
    case malformedStatusLine
    case missingResponseBody
    case nonJsonEnvelope
    case pollTimeout
}

/// One journey's scripted hub, coordinator, and REST server.
final class LibraryDownloadJourneyHarness {

    enum DiscoveryRefreshOutcome {
        case succeeds
        case fails
    }

    struct TransferPacing: Sendable {
        let chunkByteCount: Int
        let chunkDelayMilliseconds: Int

        static func unhurried() -> TransferPacing {
            return TransferPacing(chunkByteCount: 64 * 1024, chunkDelayMilliseconds: 8)
        }
    }

    var server: RestHttpServer
    let modelsDirectory: FilePath
    let stateDirectory: FilePath
    let expectedTotalBytes: UInt64
    let scriptedConfigBytes: Data
    let scriptedWeightsBytes: Data
    let scriptedHub: ScriptedHuggingFaceHub
    let hubEndpoint: URL
    let discoveryRefreshOutcome: DiscoveryRefreshOutcome
    let downloadCatalog: DownloadCatalog
    private let temporaryRoot: String

    init(
        discoveryRefreshOutcome: DiscoveryRefreshOutcome,
        gated: Bool = false,
        weightsByteCount: Int = 256 * 1024,
        pacing: TransferPacing = TransferPacing.unhurried()
    ) async throws {
        let configBytes: Data = Data("{\"model_family\": \"example\"}".utf8)
        let weightsBytes: Data = Data((0 ..< weightsByteCount).map({ (byteIndex: Int) -> UInt8 in
            return UInt8(byteIndex % 251)
        }))
        self.expectedTotalBytes = UInt64(configBytes.count + weightsBytes.count)
        self.scriptedConfigBytes = configBytes
        self.scriptedWeightsBytes = weightsBytes
        self.temporaryRoot = NSTemporaryDirectory() + "alibdl-\(UUID().uuidString.prefix(8))"
        self.stateDirectory = FilePath(string: self.temporaryRoot + "/state")
        self.modelsDirectory = FilePath(string: self.temporaryRoot + "/models")
        self.discoveryRefreshOutcome = discoveryRefreshOutcome
        try FileManager.default.createDirectory(atPath: self.stateDirectory.string, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(atPath: self.modelsDirectory.string, withIntermediateDirectories: true)

        self.scriptedHub = try ScriptedHuggingFaceHub(
            repositories: [
                ScriptedHuggingFaceHub.ScriptedRepository(
                    repositoryId: "astronomical-test/example-qwen",
                    revision: "0123456789abcdef0123456789abcdef01234567",
                    gated: gated,
                    isPrivate: false,
                    files: [
                        ScriptedHuggingFaceHub.ScriptedHubFile(
                            relativePath: "config.json",
                            bytes: configBytes,
                            servesAsLfs: false),
                        ScriptedHuggingFaceHub.ScriptedHubFile(
                            relativePath: "weights-00001-of-00001.safetensors",
                            bytes: weightsBytes),
                    ]),
            ],
            chunkByteCount: pacing.chunkByteCount,
            chunkDelayMilliseconds: pacing.chunkDelayMilliseconds)
        self.hubEndpoint = try await self.scriptedHub.start()
        self.downloadCatalog = try DownloadCatalog.parseJson(LibraryDownloadJourneyTests.catalogJson)
        self.server = try await LibraryDownloadJourneyHarness.buildServer(
            stateDirectory: self.stateDirectory,
            modelsDirectory: self.modelsDirectory,
            hubEndpoint: self.hubEndpoint,
            downloadCatalog: self.downloadCatalog,
            discoveryRefreshOutcome: discoveryRefreshOutcome)
    }

    /// Rebuilds the coordinator and REST server against the same state
    /// directory, models directory, and scripted hub — the restart seam.
    func restart() async throws {
        self.server.stop()
        self.server = try await LibraryDownloadJourneyHarness.buildServer(
            stateDirectory: self.stateDirectory,
            modelsDirectory: self.modelsDirectory,
            hubEndpoint: self.hubEndpoint,
            downloadCatalog: self.downloadCatalog,
            discoveryRefreshOutcome: self.discoveryRefreshOutcome)
    }

    private static func buildServer(
        stateDirectory: FilePath,
        modelsDirectory: FilePath,
        hubEndpoint: URL,
        downloadCatalog: DownloadCatalog,
        discoveryRefreshOutcome: DiscoveryRefreshOutcome
    ) async throws -> RestHttpServer {
        let downloadCoordinator: LibraryDownloadCoordinator = LibraryDownloadCoordinator(
            downloadCatalog: downloadCatalog,
            modelsDirectory: modelsDirectory,
            stateDirectory: stateDirectory,
            hubEndpoint: hubEndpoint,
            discoveryRefresh: {
                if discoveryRefreshOutcome == .fails {
                    throw LibraryDownloadJourneyFailure.pollTimeout
                }
            },
            availableCapacityBytes: { (_ volumeDirectory: FilePath) -> UInt64? in
                // Abundant scripted capacity: the disk-preflight contract has
                // its own suite; these journeys exercise the transfer, not
                // the volume, and an unanswered query now fails closed.
                return 1_000_000_000_000;
            })
        await downloadCoordinator.recoverStartupState()

        let recordStore: LibraryDownloadJobRecordStore = LibraryDownloadJobRecordStore(
            stateDirectory: stateDirectory)
        let publicationRegistry: LibraryValidatedPublicationRegistry = downloadCoordinator.validatedPublications
        let catalogContext: RestLibraryCatalogRouteContext = RestLibraryCatalogRouteContext(
            downloadCatalog: downloadCatalog,
            discoveredModelsProvider: { return [] },
            validatedPublicationsProvider: {
                return publicationRegistry.snapshot()
            },
            currentJobProvider: {
                guard let record: LibraryDownloadJobRecord = try? recordStore.load() else {
                    return nil
                }
                return LibraryDownloadJobSummary(
                    huggingfaceId: record.huggingfaceId,
                    stateName: record.state.rawValue)
            },
            destinationDirectoryProvider: { (huggingfaceId: String) -> String? in
                return downloadCoordinator.destinationDirectory(huggingfaceId: huggingfaceId).string
            })
        let downloadContext: RestLibraryDownloadRouteContext = RestLibraryDownloadRouteContext(
            coordinator: downloadCoordinator)

        let routeTable: RestRouteTable = RestEndpointRoutes.servingRouteTable(
            resolvedRuntimeConfig: try LibraryDownloadJourneyHarness.emptyResolvedConfig(
                stateDirectory: stateDirectory),
            workerHealthState: WorkerHealthState(),
            instancePaths: AstronomicalInstancePaths.forExplicitStateDirectory(
                stateDirectory,
                defaultBindAddress: SocketEndpoint.loopback(port: 0)),
            buildIdentity: ApplicationBuildIdentity(
                version: "0.0.0-test",
                buildNumber: 0,
                commit: "unknown",
                isDirty: false),
            libraryCatalogContext: catalogContext,
            libraryDownloadContext: downloadContext)
        return try RestHttpServer.start(
            bindEndpoint: SocketEndpoint.loopback(port: 0),
            routeTable: routeTable)
    }

    /// True when any file under the hub blob cache is still incomplete.
    func hasIncompleteCacheBlobs() -> Bool {
        let cacheRoot: String = self.stateDirectory.appending(component: "hub-blob-cache").string
        guard let enumerator: FileManager.DirectoryEnumerator = FileManager.default.enumerator(
            at: URL(fileURLWithPath: cacheRoot),
            includingPropertiesForKeys: nil)
        else {
            return false
        }
        while let enumeratedObject: Any = enumerator.nextObject() {
            if let cachedFileUrl: URL = enumeratedObject as? URL,
                cachedFileUrl.lastPathComponent.contains(".incomplete")
            {
                return true
            }
        }
        return false
    }

    /// The durable record document exactly as it sits on disk.
    func durableRecordOnDisk() -> LibraryDownloadJobRecord? {
        let recordStore: LibraryDownloadJobRecordStore = LibraryDownloadJobRecordStore(
            stateDirectory: self.stateDirectory)
        return try? recordStore.load()
    }

    func stop() {
        self.server.stop()
        self.scriptedHub.stop()
        try? FileManager.default.removeItem(atPath: self.temporaryRoot)
    }

    /// Polls the catalog until the entry reports ready; bounded well under
    /// the 120-second test ceiling.
    func pollUntilCatalogEntryReady() throws -> [String: Any] {
        for _ in 0 ..< 150 {
            let (statusCode, envelope): (Int, [String: Any]) = try self.getCatalogEnvelope()
            if statusCode == 200 {
                let entries: [[String: Any]] = envelope["entries"] as? [[String: Any]] ?? []
                if let firstEntry: [String: Any] = entries.first,
                    firstEntry["ready_on_this_mac"] as? Bool == true
                {
                    return firstEntry
                }
            }
            Thread.sleep(forTimeInterval: 0.1)
        }
        throw LibraryDownloadJourneyFailure.pollTimeout
    }

    func pollUntilDownloadState(_ wantedState: String) throws -> [String: Any] {
        for _ in 0 ..< 150 {
            let responseText: String? = RawLoopbackHttpClient.exchange(
                port: self.server.boundEndpoint.port,
                requestText: "GET /v1/library/download HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n")
            let (_, envelope): (Int, [String: Any]) = try LibraryDownloadJourneyHarness.decodeDownloadEnvelope(responseText)
            if envelope["state"] as? String == wantedState {
                return envelope
            }
            Thread.sleep(forTimeInterval: 0.1)
        }
        throw LibraryDownloadJourneyFailure.pollTimeout
    }

    /// Polls until the live REST projection reports in-flight bytes.
    func pollUntilLiveProgressObserved() throws -> [String: Any] {
        for _ in 0 ..< 150 {
            let responseText: String? = RawLoopbackHttpClient.exchange(
                port: self.server.boundEndpoint.port,
                requestText: "GET /v1/library/download HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n")
            let (_, envelope): (Int, [String: Any]) = try LibraryDownloadJourneyHarness.decodeDownloadEnvelope(responseText)
            let completedBytes: UInt64 = envelope["bytes_completed"] as? UInt64 ?? 0
            let totalBytes: UInt64 = envelope["bytes_total"] as? UInt64 ?? 0
            if completedBytes > 0 && totalBytes > 0 && completedBytes < totalBytes {
                return envelope
            }
            Thread.sleep(forTimeInterval: 0.1)
        }
        throw LibraryDownloadJourneyFailure.pollTimeout
    }

    func pollUntilAnyDownloadState() throws -> [String: Any] {
        for _ in 0 ..< 150 {
            let responseText: String? = RawLoopbackHttpClient.exchange(
                port: self.server.boundEndpoint.port,
                requestText: "GET /v1/library/download HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n")
            let (_, envelope): (Int, [String: Any]) = try LibraryDownloadJourneyHarness.decodeDownloadEnvelope(responseText)
            if (envelope["state"] as? String ?? "idle") != "idle" {
                return envelope
            }
            Thread.sleep(forTimeInterval: 0.1)
        }
        throw LibraryDownloadJourneyFailure.pollTimeout
    }

    private func getCatalogEnvelope() throws -> (Int, [String: Any]) {
        let responseText: String? = RawLoopbackHttpClient.exchange(
            port: self.server.boundEndpoint.port,
            requestText: "GET /v1/library/catalog HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n")
        return try LibraryDownloadJourneyHarness.decodeDownloadEnvelope(responseText)
    }

    private static func decodeDownloadEnvelope(_ responseText: String?) throws -> (Int, [String: Any]) {
        guard let unwrappedResponseText: String = responseText else {
            throw LibraryDownloadJourneyFailure.missingResponse
        }
        guard let statusToken: Substring = unwrappedResponseText.split(separator: " ", maxSplits: 2).dropFirst().first,
            let statusCode: Int = Int(statusToken)
        else {
            throw LibraryDownloadJourneyFailure.malformedStatusLine
        }
        guard let bodyStart: String.Index = unwrappedResponseText.range(of: "\r\n\r\n")?.upperBound else {
            throw LibraryDownloadJourneyFailure.missingResponseBody
        }
        let bodyJsonText: String = String(unwrappedResponseText[bodyStart...])
        guard let envelope: [String: Any] = try? JSONSerialization.jsonObject(with: Data(bodyJsonText.utf8)) as? [String: Any] else {
            throw LibraryDownloadJourneyFailure.nonJsonEnvelope
        }
        return (statusCode, envelope)
    }

    private static func emptyResolvedConfig(stateDirectory: FilePath) throws -> ResolvedRuntimeConfig {
        let instancePaths: AstronomicalInstancePaths = AstronomicalInstancePaths.forExplicitStateDirectory(
            stateDirectory,
            defaultBindAddress: SocketEndpoint.loopback(port: 0))
        try FileManager.default.createDirectory(
            atPath: instancePaths.stateDirectory.string,
            withIntermediateDirectories: true)
        let emptyConfig: AstronomicalConfig = try AstronomicalConfig.loadFromInstancePaths(instancePaths)
        return ResolvedRuntimeConfig(
            configurationGeneration: "0123456789abcdef0123456789abcdef",
            workerExecutablePath: FilePath(string: "/opt/astronomical/bin/astronomical-inference-worker"),
            discoveredModels: [],
            modelDiscoveryDiagnostics: [],
            configuredModelDirectories: [],
            modelPolicyCatalog: try ResolvedModelPolicyCatalog.resolve(
                userConfig: emptyConfig,
                discoveredModels: [],
                artifactContextWindows: [:]),
            unmatchedModelConfigIds: [],
            maximumMlxMemoryBytes: nil,
            performanceAttributionEnabled: false,
            completionAttributionEnabled: false,
            experimentalQwenThinkingChannelSeedEnabled: false,
            persistentPromptCacheEnabled: true,
            configuredPersistentPromptCacheEnabled: nil,
            configuredPromptCacheMaximumSizeBytes: 50_000_000_000,
            promptCacheConfig: PromptCacheConfig(
                rootDirectory: FilePath(string: "/state/prompt-cache"),
                maximumSizeBytes: 50_000_000_000),
            bindAddress: "127.0.0.1:0",
            bindEndpoint: SocketEndpoint.loopback(port: 0),
            loggingConfig: LoggingConfig(
                directory: FilePath(string: "/state/logs"),
                level: LogLevel.warn,
                retainedFiles: 7))
    }
}
