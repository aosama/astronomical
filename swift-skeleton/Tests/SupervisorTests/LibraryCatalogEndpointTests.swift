import Foundation

import Testing

import AstronomicalConfig
import IpcProtocol
import RestContract
import JourneyCategories

@testable import Supervisor

/**
 * Acceptance journeys for the Library catalog REST surface: the read-only
 * catalog endpoint answers the validated release catalog in authored order
 * with per-Mac readiness joined in, catalog mutation is rejected, and
 * unknown Library paths stay unmatched. Mirrors the Rust
 * rest_api/library_catalog.rs contract coverage.
 */
@Suite(.serialized, .tags(.hermeticJourney))
final class LibraryCatalogEndpointTests {

    private static let fictionalCatalogJson: String = """
        {
            "schema_version": 2,
            "entries": [
                {
                    "huggingface_id": "astronomical-test/example-qwen",
                    "revision": "0123456789abcdef0123456789abcdef01234567",
                    "display_name": "Example Qwen",
                    "family": "qwen3_5",
                    "approximate_size_bytes": 4000000000,
                    "public": true
                },
                {
                    "huggingface_id": "astronomical-test/example-k2",
                    "revision": "89abcdef0123456789abcdef0123456789abcdef",
                    "display_name": "Example K2",
                    "family": "k2_horizon_mova",
                    "approximate_size_bytes": 5000000000,
                    "public": true
                },
                {
                    "huggingface_id": "astronomical-test/example-embedder",
                    "revision": "fedcba9876543210fedcba9876543210fedcba98",
                    "display_name": "Example Embedder",
                    "family": "modernbert",
                    "approximate_size_bytes": 200000000,
                    "public": true,
                    "capabilities": {"supports_embeddings": true}
                }
            ]
        }
        """

    @Test
    func should_return_the_validated_catalog_in_authored_order_when_nothing_is_installed() throws {
        let downloadCatalog: DownloadCatalog = try DownloadCatalog.parseJson(
            LibraryCatalogEndpointTests.fictionalCatalogJson)
        let server: RestHttpServer = try self.startCatalogServer(downloadCatalog: downloadCatalog)

        let responseText: String? = RawLoopbackHttpClient.exchange(
            port: server.boundEndpoint.port,
            requestText: "GET /v1/library/catalog HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n")
        let unwrappedResponseText: String = try self.requireResponseText(responseText)
        #expect(unwrappedResponseText.contains("application/json"), "the catalog reply is JSON")
        let (statusCode, catalogDocument): (Int, Any) = try self.decodeJsonDocument(unwrappedResponseText)
        #expect(statusCode == 200)

        let expectedDocumentText: String = """
            {
                "schema_version": 2,
                "entries": [
                    {
                        "huggingface_id": "astronomical-test/example-qwen",
                        "revision": "0123456789abcdef0123456789abcdef01234567",
                        "display_name": "Example Qwen",
                        "family": "qwen3_5",
                        "approximate_size_bytes": 4000000000,
                        "public": true,
                        "ready_on_this_mac": false,
                        "download_state": null,
                        "capabilities": {
                            "supports_reasoning": false,
                            "supports_vision": false,
                            "supports_tool_calls": false,
                            "supports_image_generation": false,
                            "supports_embeddings": false
                        }
                    },
                    {
                        "huggingface_id": "astronomical-test/example-k2",
                        "revision": "89abcdef0123456789abcdef0123456789abcdef",
                        "display_name": "Example K2",
                        "family": "k2_horizon_mova",
                        "approximate_size_bytes": 5000000000,
                        "public": true,
                        "ready_on_this_mac": false,
                        "download_state": null,
                        "capabilities": {
                            "supports_reasoning": false,
                            "supports_vision": false,
                            "supports_tool_calls": false,
                            "supports_image_generation": false,
                            "supports_embeddings": false
                        }
                    },
                    {
                        "huggingface_id": "astronomical-test/example-embedder",
                        "revision": "fedcba9876543210fedcba9876543210fedcba98",
                        "display_name": "Example Embedder",
                        "family": "modernbert",
                        "approximate_size_bytes": 200000000,
                        "public": true,
                        "ready_on_this_mac": false,
                        "download_state": null,
                        "capabilities": {
                            "supports_reasoning": false,
                            "supports_vision": false,
                            "supports_tool_calls": false,
                            "supports_image_generation": false,
                            "supports_embeddings": true
                        }
                    }
                ]
            }
            """
        let expectedDocument: Any = try JSONSerialization.jsonObject(with: Data(expectedDocumentText.utf8))
        #expect(
            catalogDocument as? NSObject == expectedDocument as? NSObject,
            "the catalog document must equal the validated authored projection")
        server.stop()
    }

    @Test
    func should_reject_catalog_mutation_and_leave_unknown_library_paths_unmatched() throws {
        let downloadCatalog: DownloadCatalog = try DownloadCatalog.parseJson(
            LibraryCatalogEndpointTests.fictionalCatalogJson)
        let server: RestHttpServer = try self.startCatalogServer(downloadCatalog: downloadCatalog)

        let mutationResponseText: String? = RawLoopbackHttpClient.exchange(
            port: server.boundEndpoint.port,
            requestText: "POST /v1/library/catalog HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Length: 0\r\n\r\n")
        let (mutationStatusCode, _): (Int, Any) = try self.decodeJsonDocument(
            try self.requireResponseText(mutationResponseText))
        #expect(mutationStatusCode == 405, "the catalog is immutable")

        let unknownResponseText: String? = RawLoopbackHttpClient.exchange(
            port: server.boundEndpoint.port,
            requestText: "GET /v1/library/unknown HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n")
        let (unknownStatusCode, _): (Int, Any) = try self.decodeJsonDocument(
            try self.requireResponseText(unknownResponseText))
        #expect(unknownStatusCode == 404, "unknown Library paths stay unmatched")
        server.stop()
    }

    // MARK: - Journey helpers

    private func startCatalogServer(downloadCatalog: DownloadCatalog) throws -> RestHttpServer {
        let libraryCatalogContext: RestLibraryCatalogRouteContext = RestLibraryCatalogRouteContext(
            downloadCatalog: downloadCatalog,
            discoveredModelsProvider: { return Array(); },
            validatedPublicationsProvider: { return Set(); },
            currentJobProvider: { return nil; },
            destinationDirectoryProvider: { (_ huggingfaceId: String) -> String? in return nil; })
        let routeTable: RestRouteTable = RestEndpointRoutes.servingRouteTable(
            resolvedRuntimeConfig: try self.makeEmptyResolvedConfig(),
            workerHealthState: WorkerHealthState(),
            instancePaths: AstronomicalInstancePaths.forExplicitStateDirectory(
                FilePath(string: "/library-catalog-journey-state"),
                defaultBindAddress: SocketEndpoint.loopback(port: 0)),
            buildIdentity: ApplicationBuildIdentity(
                version: "0.0.0-test",
                buildNumber: 0,
                commit: "unknown",
                isDirty: false),
            libraryCatalogContext: libraryCatalogContext)
        return try RestHttpServer.start(
            bindEndpoint: SocketEndpoint.loopback(port: 0),
            routeTable: routeTable)
    }

    private func makeEmptyResolvedConfig() throws -> ResolvedRuntimeConfig {
        let temporaryStateDirectory: String = NSTemporaryDirectory() + "alibcat-\(UUID().uuidString.prefix(8))"
        try FileManager.default.createDirectory(atPath: temporaryStateDirectory, withIntermediateDirectories: true)
        self.temporaryRootPath = temporaryStateDirectory
        let instancePaths: AstronomicalInstancePaths = AstronomicalInstancePaths.forHomeDirectory(
            FilePath(string: temporaryStateDirectory),
            runtimeInstance: AstronomicalRuntimeInstance.development)
        try FileManager.default.createDirectory(
            atPath: instancePaths.stateDirectory.string,
            withIntermediateDirectories: true)
        let emptyConfig: AstronomicalConfig = try AstronomicalConfig.loadFromInstancePaths(instancePaths)
        return ResolvedRuntimeConfig(
            configurationGeneration: "0123456789abcdef0123456789abcdef",
            workerExecutablePath: FilePath(string: "/opt/astronomical/bin/astronomical-inference-worker"),
            discoveredModels: Array(),
            modelDiscoveryDiagnostics: Array(),
            configuredModelDirectories: Array(),
            modelPolicyCatalog: try ResolvedModelPolicyCatalog.resolve(
                userConfig: emptyConfig,
                discoveredModels: Array(),
                artifactContextWindows: Dictionary<String, UInt32>()),
            unmatchedModelConfigIds: Array<String>(),
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

    private func decodeJsonDocument(_ responseText: String) throws -> (Int, Any) {
        guard let statusToken: Substring = responseText.split(separator: " ", maxSplits: 2).dropFirst().first else {
            throw LibraryCatalogTestFailure.malformedStatusLine
        }
        guard let statusCode: Int = Int(statusToken) else {
            throw LibraryCatalogTestFailure.malformedStatusLine
        }
        guard let bodyStart: String.Index = responseText.range(of: "\r\n\r\n")?.upperBound else {
            throw LibraryCatalogTestFailure.missingResponseBody
        }
        let bodyJsonText: String = String(responseText[bodyStart...])
        guard !bodyJsonText.isEmpty else {
            throw LibraryCatalogTestFailure.missingResponseBody
        }
        let bodyDocument: Any = try JSONSerialization.jsonObject(with: Data(bodyJsonText.utf8))
        return (statusCode, bodyDocument)
    }

    private func requireResponseText(_ responseText: String?) throws -> String {
        guard let unwrappedResponseText: String = responseText else {
            throw LibraryCatalogTestFailure.missingResponse
        }
        return unwrappedResponseText
    }

    private var temporaryRootPath: String?

    deinit {
        if let temporaryRootPath: String = self.temporaryRootPath {
            try? FileManager.default.removeItem(atPath: temporaryRootPath)
        }
    }
}

private enum LibraryCatalogTestFailure: Error {
    case missingResponse
    case malformedStatusLine
    case missingResponseBody
}
