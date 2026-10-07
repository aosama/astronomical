import Foundation;

import Testing;

import AstronomicalConfig;
import IpcProtocol;
import RestContract;

@testable import Supervisor;

/**
 * The menu application's whole-document status contract, migrating
 * apps/supervisor/tests/rest_api/application/menu_status.rs: the production
 * /v1/status document stays aligned with the shared cross-application
 * fixtures, and the Development instance reports the state directory label
 * the menu depends on.
 */
@Suite(.serialized, .tags(.hermeticJourney))
final class MenuStatusContractTests {

    static let autoregressiveModelId: String = "fictional/autoregressive-model";
    static let fluxModelId: String = "fictional/image-model";
    static let autoregressiveGeneration: String =
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    static let fluxGeneration: String =
        "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

    @Test
    func should_keep_the_complete_autoregressive_menu_fixture_aligned_with_production_status() throws {
        let actualStatus: [String: Any] = try MenuStatusContractTests.productionStatusDocument(
            runtimeFeatureConfiguration: MenuStatusContractTests.autoregressiveRuntimeConfiguration());
        let expectedStatus: [String: Any] = try MenuStatusContractTests.fixtureDocument(
            fixtureFileName: "full-autoregressive-status.json");

        #expect(
            MenuStatusContractTests.documentsMatch(
                MenuStatusContractTests.normalizedBuildIdentity(actualStatus, expectedStatus),
                expectedStatus),
            Comment(rawValue: MenuStatusContractTests.firstDivergence(
                MenuStatusContractTests.normalizedBuildIdentity(actualStatus, expectedStatus),
                expectedStatus)));
    }

    @Test
    func should_keep_the_complete_flux_menu_fixture_aligned_with_production_status() throws {
        let actualStatus: [String: Any] = try MenuStatusContractTests.productionStatusDocument(
            runtimeFeatureConfiguration: MenuStatusContractTests.fluxRuntimeConfiguration());
        let expectedStatus: [String: Any] = try MenuStatusContractTests.fixtureDocument(
            fixtureFileName: "full-flux-status.json");

        #expect(
            MenuStatusContractTests.documentsMatch(
                MenuStatusContractTests.normalizedBuildIdentity(actualStatus, expectedStatus),
                expectedStatus),
            Comment(rawValue: MenuStatusContractTests.firstDivergence(
                MenuStatusContractTests.normalizedBuildIdentity(actualStatus, expectedStatus),
                expectedStatus)));
    }

    @Test
    func should_report_the_standard_development_state_directory_required_by_the_menu() throws {
        let statusJourney: StatusProgressJourney = try StatusProgressJourney.launch();
        defer { statusJourney.dispose() }

        let statusDocument: [String: Any] = try statusJourney.getStatusDocument();
        let menuFixture: [String: Any] = try MenuStatusContractTests.fixtureDocument(
            fixtureFileName: "full-autoregressive-status.json");
        let applicationDocument: [String: Any] = try MenuStatusContractTests.requireObject(
            statusDocument, field: "application");
        let fixtureApplication: [String: Any] = try MenuStatusContractTests.requireObject(
            menuFixture, field: "application");

        #expect(
            applicationDocument["state_directory"] as? String
                == fixtureApplication["state_directory"] as? String);
    }

    // MARK: Fixtures

    /// The runtime policy the autoregressive menu fixture acknowledges.
    static func autoregressiveRuntimeConfiguration() -> WorkerRuntimeFeatureConfiguration {
        return WorkerRuntimeFeatureConfiguration(
            configurationGeneration: MenuStatusContractTests.autoregressiveGeneration,
            persistentPromptCacheEnabled: true,
            promptCacheMaximumSizeBytes: 10_000_000_000,
            loadedModel: .autoregressive(WorkerLoadedAutoregressiveModelRuntimeConfiguration(
                modelId: MenuStatusContractTests.autoregressiveModelId,
                maximumContextTokens: 65_536,
                maximumOutputTokens: 8_192,
                chunking: WorkerChunkingConfiguration(
                    fixedPromptProcessingChunkSizeTokens: 2_048,
                    fixedSsdStreamingPromptProcessingChunkSizeTokens: 2_048,
                    fullAttentionKeyValueGrowthTokens: 256,
                    prefillGraphSubmissionLayerInterval: 0,
                    experimentalSsdPagingPrefillGraphSubmissionLayerInterval: 1,
                    experimentalSsdPagingGenerationGraphSubmissionLayerInterval: 3,
                    promptCacheBlockTokens: nil,
                    promptCacheCommonPrefixStrideBlocks: 4,
                    experimentalDecodeStageAttributionEnabled: false,
                    experimentalQuantizedKvCacheEnabled: false,
                    experimentalFusedMoeDecodeEnabled: false))));
    }

    /// The runtime policy the Flux menu fixture acknowledges.
    static func fluxRuntimeConfiguration() -> WorkerRuntimeFeatureConfiguration {
        return WorkerRuntimeFeatureConfiguration(
            configurationGeneration: MenuStatusContractTests.fluxGeneration,
            persistentPromptCacheEnabled: false,
            promptCacheMaximumSizeBytes: 0,
            loadedModel: .flux2Klein(WorkerFlux2KleinModelConfiguration(
                modelId: MenuStatusContractTests.fluxModelId,
                modelFamily: .flux2Klein,
                artifactRevision: "fictional-revision")));
    }

    /// One production status document over the fixture runtime policy.
    static func productionStatusDocument(
        runtimeFeatureConfiguration: WorkerRuntimeFeatureConfiguration
    ) throws -> [String: Any] {
        let readyModelId: String = runtimeFeatureConfiguration.loadedModel!.modelId();
        var healthSnapshot: WorkerHealthSnapshot = WorkerHealthSnapshot.readyWithModel(
            modelId: readyModelId,
            capabilities: RestChatJourneySupport.readyChatCapabilities());
        healthSnapshot.workerRuntimeFeatureConfiguration = runtimeFeatureConfiguration;
        let workerHealthState: WorkerHealthState = WorkerHealthState();
        workerHealthState.publish(healthSnapshot);
        let homeDirectoryUrl: URL = FileManager.default.temporaryDirectory
            .appendingPathComponent("astronomical-menu-contract-\(UUID().uuidString)", isDirectory: true);
        try FileManager.default.createDirectory(at: homeDirectoryUrl, withIntermediateDirectories: true);
        defer { try? FileManager.default.removeItem(at: homeDirectoryUrl) }
        let instancePaths: AstronomicalInstancePaths = AstronomicalInstancePaths.forHomeDirectory(
            FilePath(string: homeDirectoryUrl.path),
            runtimeInstance: AstronomicalRuntimeInstance.development);
        let resolvedRuntimeConfig: ResolvedRuntimeConfig = try RestChatJourneySupport.makeResolvedConfig();
        let routeTable: RestRouteTable = RestEndpointRoutes.servingRouteTable(
            resolvedRuntimeConfig: resolvedRuntimeConfig,
            workerHealthState: workerHealthState,
            instancePaths: instancePaths,
            buildIdentity: RestChatJourneySupport.journeyBuildIdentity());
        let routeOutcome: RestRouteOutcome = routeTable.outcome(method: "GET", path: "/v1/status");
        guard case .handler(let routeHandler) = routeOutcome else {
            throw MenuStatusContractFailure.statusRouteMissing;
        }
        let statusResponse: RestHttpResponse = try routeHandler(WorkerReplacementJourney.emptyRequest(
            method: "GET",
            path: "/v1/status"));
        return try ConfigReloadJourney.decodeObject(statusResponse);
    }

    /// Deep equality over the decoded documents; NSDictionary comparison
    /// recurses through nested arrays and objects.
    static func documentsMatch(
        _ actualStatus: [String: Any],
        _ expectedStatus: [String: Any]
    ) -> Bool {
        return (actualStatus as NSDictionary).isEqual(to: expectedStatus);
    }

    /// Reports the first divergent path between the two documents.
    static func firstDivergence(
        _ actualDocument: Any,
        _ expectedDocument: Any,
        path: String = ""
    ) -> String {
        switch (actualDocument, expectedDocument) {
        case let (actualObject as [String: Any], expectedObject as [String: Any]):
            for (key, expectedValue) in expectedObject.sorted(by: { $0.key < $1.key }) {
                guard let actualValue: Any = actualObject[key] else {
                    return "missing \(path).\(key)";
                }
                let divergence: String = firstDivergence(actualValue, expectedValue, path: "\(path).\(key)");
                if divergence != "" {
                    return divergence;
                }
            }
            for key in actualObject.keys where expectedObject[key] == nil {
                return "extra \(path).\(key)";
            }
            return "";
        case let (actualArray as [Any], expectedArray as [Any]):
            if actualArray.count != expectedArray.count {
                return "\(path) count \(actualArray.count) != \(expectedArray.count)";
            }
            for (index, expectedValue) in expectedArray.enumerated() {
                let divergence: String = firstDivergence(
                    actualArray[index], expectedValue, path: "\(path)[\(index)]");
                if divergence != "" {
                    return divergence;
                }
            }
            return "";
        default:
            if (actualDocument as AnyObject).isEqual(expectedDocument) {
                return "";
            }
            return "\(path): \(actualDocument) != \(expectedDocument)";
        }
    }

    /// Loads one shared menu fixture beside the menu application's own
    /// contract tests, so both applications prove the same document.
    static func fixtureDocument(fixtureFileName: String) throws -> [String: Any] {
        let journeyFileUrl: URL = URL(fileURLWithPath: #filePath);
        let repositoryRootUrl: URL = journeyFileUrl
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent();
        let fixtureUrl: URL = repositoryRootUrl
            .appendingPathComponent("apps/astronomical-menu/tests/AstronomicalMenuContractTests/Fixtures")
            .appendingPathComponent(fixtureFileName);
        let fixtureData: Data = try Data(contentsOf: fixtureUrl);
        return try JSONSerialization.jsonObject(with: fixtureData, options: []) as! [String: Any];
    }

    /// Copies the build-identity fields the fixture pins from the expected
    /// document into the actual one; everything else must already match.
    static func normalizedBuildIdentity(
        _ actualStatus: [String: Any],
        _ expectedStatus: [String: Any]
    ) -> [String: Any] {
        var normalizedStatus: [String: Any] = actualStatus;
        var actualApplication: [String: Any] = (actualStatus["application"] as? [String: Any]) ?? [:];
        let expectedApplication: [String: Any] = (expectedStatus["application"] as? [String: Any]) ?? [:];
        for environmentFieldName: String in [
            "version", "build_number", "commit", "is_dirty", "state_directory",
        ] {
            actualApplication[environmentFieldName] = expectedApplication[environmentFieldName];
        }
        normalizedStatus["application"] = actualApplication;
        return normalizedStatus;
    }

    private static func requireObject(
        _ document: [String: Any],
        field: String
    ) throws -> [String: Any] {
        guard let nestedObject: [String: Any] = document[field] as? [String: Any] else {
            throw MenuStatusContractFailure.objectMissing(field);
        }
        return nestedObject;
    }
}

/// Typed failures of the menu contract journeys.
enum MenuStatusContractFailure: Error {

    case statusRouteMissing;
    case objectMissing(String);
}
