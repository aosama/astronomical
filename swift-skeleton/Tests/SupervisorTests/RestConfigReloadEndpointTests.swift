import Foundation;

import Testing;

import AstronomicalConfig;
import IpcProtocol;
import RestContract;
import JourneyCategories;

@testable import Supervisor;

/**
 * Hermetic journeys for POST /v1/config/reload, migrating
 * apps/supervisor/tests/rest_api/config_reload/reload_validation.rs over a
 * resolver-driven config home without live worker control: invalid documents
 * fail without mutating live state, ambiguous model identities surface
 * path-safe feedback, a missing authored directory reloads with a
 * diagnostic, and a busy executor aborts with 409. Worker replacement and
 * the reload-side memory journeys land with the worker-replacement slice.
 */
@Suite(.serialized, .tags(.hermeticJourney))
final class RestConfigReloadEndpointTests {

    @Test
    func should_reject_invalid_config_reload_without_mutating_live_state() throws {
        let journey: ConfigReloadJourney = try ConfigReloadJourney.launch();
        defer { journey.dispose() }
        ConfigReloadJourney.writeRawConfigFile(journey.homeDirectoryUrl, rawConfigJson: "{ this is not valid json }");

        let reloadResponse: RestHttpResponse = try journey.postConfigReload();

        #expect(reloadResponse.statusCode == 400);
        let reloadDocument: [String: Any] = try ConfigReloadJourney.decodeObject(reloadResponse);
        #expect(reloadDocument["status"] as? String == "invalid_config");
        let statusDocument: [String: Any] = try journey.getStatusDocument();
        let configurationDocument: [String: Any] = try ConfigReloadJourney.requireConfigurationDocument(statusDocument);
        #expect(configurationDocument["is_effective"] as? Bool == false);
        #expect(configurationDocument["restart_required"] as? Bool == false);
        #expect(
            configurationDocument["validation_error"] as? String
                == "Configuration is invalid; correct the local configuration file and retry");
        #expect(journey.transitionState.currentReloadableConfig() == journey.initialResolvedConfig);
    }

    @Test
    func should_preserve_live_serving_and_expose_path_safe_duplicate_model_feedback() throws {
        let journey: ConfigReloadJourney = try ConfigReloadJourney.launch();
        defer { journey.dispose() }
        let firstRootUrl: URL = journey.homeDirectoryUrl.appendingPathComponent("first-root");
        let secondRootUrl: URL = journey.homeDirectoryUrl.appendingPathComponent("second-root");
        try ConfigReloadJourney.writeMinimalQwenModel(firstRootUrl.appendingPathComponent("ambiguous-model"));
        try ConfigReloadJourney.writeMinimalQwenModel(secondRootUrl.appendingPathComponent("ambiguous-model"));
        ConfigReloadJourney.writeConfigFile(
            journey.homeDirectoryUrl,
            configuredFieldsJson: "{\"runtime\":{\"model_directories\":"
                + "[\"\(firstRootUrl.path)\",\"\(secondRootUrl.path)\"]}}");

        let reloadResponse: RestHttpResponse = try journey.postConfigReload();

        #expect(reloadResponse.statusCode == 400);
        let reloadBodyText: String = String(decoding: reloadResponse.bodyBytes, as: UTF8.self);
        #expect(reloadBodyText.contains("ambiguous-model"));
        #expect(reloadBodyText.contains("entries 1, 2"));
        #expect(!reloadBodyText.contains(journey.homeDirectoryUrl.path),
            "the reload feedback must stay path-free");
        #expect(journey.transitionState.currentReloadableConfig() == journey.initialResolvedConfig);
        let statusDocument: [String: Any] = try journey.getStatusDocument();
        let configurationDocument: [String: Any] = try ConfigReloadJourney.requireConfigurationDocument(statusDocument);
        let diagnostics: Array<[String: Any]> = try ConfigReloadJourney.requireDiagnostics(configurationDocument);
        #expect(diagnostics.first?["model_id"] as? String == "ambiguous-model");
    }

    @Test
    func should_apply_reload_when_an_authored_model_directory_is_missing() throws {
        let journey: ConfigReloadJourney = try ConfigReloadJourney.launch();
        defer { journey.dispose() }
        let missingRootUrl: URL = journey.homeDirectoryUrl.appendingPathComponent("deleted-models");
        let availableRootUrl: URL = journey.homeDirectoryUrl.appendingPathComponent("available-models");
        try ConfigReloadJourney.writeMinimalQwenModel(availableRootUrl.appendingPathComponent("kept-model"));
        ConfigReloadJourney.writeConfigFile(
            journey.homeDirectoryUrl,
            configuredFieldsJson: "{\"runtime\":{\"model_directories\":"
                + "[\"\(missingRootUrl.path)\",\"\(availableRootUrl.path)\"]}}");

        let reloadResponse: RestHttpResponse = try journey.postConfigReload();

        #expect(reloadResponse.statusCode == 200);
        let reloadBodyText: String = String(decoding: reloadResponse.bodyBytes, as: UTF8.self);
        #expect(!reloadBodyText.contains(journey.homeDirectoryUrl.path),
            "the reload response must stay path-free");
        let statusResponse: RestHttpResponse = try journey.getStatusResponse();
        let statusBodyText: String = String(decoding: statusResponse.bodyBytes, as: UTF8.self);
        #expect(!statusBodyText.contains(journey.homeDirectoryUrl.path),
            "the status document must stay path-free");
        let statusDocument: [String: Any] = try ConfigReloadJourney.decodeObject(statusResponse);
        let configurationDocument: [String: Any] = try ConfigReloadJourney.requireConfigurationDocument(statusDocument);
        let diagnostics: Array<[String: Any]> = try ConfigReloadJourney.requireDiagnostics(configurationDocument);
        #expect(diagnostics.first?["code"] as? String == "unavailable_model_directory");
        #expect(diagnostics.first?["configured_root_numbers"] as? [Int] == [1]);
    }

    @Test
    func should_return_invalid_config_feedback_when_fixed_prompt_processing_tokens_are_zero() throws {
        let journey: ConfigReloadJourney = try ConfigReloadJourney.launch();
        defer { journey.dispose() }
        ConfigReloadJourney.writeConfigFile(
            journey.homeDirectoryUrl,
            configuredFieldsJson:
                "{ \"chunking\": { \"fixed_prompt_processing_chunk_size_tokens\": 0 } }");

        let reloadResponse: RestHttpResponse = try journey.postConfigReload();

        #expect(reloadResponse.statusCode == 400);
        let reloadDocument: [String: Any] = try ConfigReloadJourney.decodeObject(reloadResponse);
        #expect(reloadDocument["status"] as? String == "invalid_config");
        #expect(
            reloadDocument["message"] as? String
                == "Configuration is invalid; correct the local configuration file and retry");
        #expect(journey.transitionState.currentReloadableConfig() == journey.initialResolvedConfig);
    }

    @Test
    func should_return_busy_when_generation_is_active_during_config_reload() throws {
        let journey: ConfigReloadJourney = try ConfigReloadJourney.launch(
            generationActivityIdle: false);
        defer { journey.dispose() }
        ConfigReloadJourney.writeConfigFile(journey.homeDirectoryUrl, configuredFieldsJson: "{}");

        let reloadResponse: RestHttpResponse = try journey.postConfigReload();

        #expect(reloadResponse.statusCode == 409);
        let reloadDocument: [String: Any] = try ConfigReloadJourney.decodeObject(reloadResponse);
        #expect(reloadDocument["status"] as? String == "busy");
        #expect(journey.transitionState.currentReloadableConfig() == journey.initialResolvedConfig);
    }
}

/// One reload journey: a Development-shaped config home with its resolver,
/// a synthetic live snapshot, and the serving route table wired with the
/// reload context but no worker control.
final class ConfigReloadJourney {

    let transitionState: ConfigTransitionState;
    let initialResolvedConfig: ResolvedRuntimeConfig;
    let homeDirectoryUrl: URL;
    private let routeTable: RestRouteTable;

    private init(
        transitionState: ConfigTransitionState,
        initialResolvedConfig: ResolvedRuntimeConfig,
        homeDirectoryUrl: URL,
        routeTable: RestRouteTable
    ) {
        self.transitionState = transitionState;
        self.initialResolvedConfig = initialResolvedConfig;
        self.homeDirectoryUrl = homeDirectoryUrl;
        self.routeTable = routeTable;
    }

    static func launch(generationActivityIdle: Bool = true) throws -> ConfigReloadJourney {
        let homeDirectoryUrl: URL = FileManager.default.temporaryDirectory
            .appendingPathComponent("astronomical-reload-journey-\(UUID().uuidString)", isDirectory: true);
        try FileManager.default.createDirectory(at: homeDirectoryUrl, withIntermediateDirectories: true);
        writeRawConfigFile(homeDirectoryUrl, rawConfigJson: "{}");
        let instancePaths: AstronomicalInstancePaths = AstronomicalInstancePaths.forHomeDirectory(
            FilePath(string: homeDirectoryUrl.path),
            runtimeInstance: AstronomicalRuntimeInstance.development);
        let resolver: ResolvedRuntimeConfigResolver = ResolvedRuntimeConfigResolver(
            instancePaths: instancePaths,
            fallbackWorkerExecutablePath: try RestChatJourneySupport.makeResolvedConfig().workerExecutablePath);
        let initialResolvedConfig: ResolvedRuntimeConfig = try RestChatJourneySupport.makeResolvedConfig();
        let transitionState: ConfigTransitionState = ConfigTransitionState(
            reloadableConfig: initialResolvedConfig,
            configuredConfigSnapshot: initialResolvedConfig);
        let routeTable: RestRouteTable = RestEndpointRoutes.servingRouteTable(
            resolvedRuntimeConfig: initialResolvedConfig,
            workerHealthState: WorkerHealthState(),
            instancePaths: instancePaths,
            buildIdentity: RestChatJourneySupport.journeyBuildIdentity(),
            configReloadContext: RestConfigReloadRouteContext(
                transitionState: transitionState,
                runtimeConfigResolver: resolver,
                workerControl: nil,
                workerHealthState: WorkerHealthState(),
                generationActivityIdleProvider: { return generationActivityIdle }));
        return ConfigReloadJourney(
            transitionState: transitionState,
            initialResolvedConfig: initialResolvedConfig,
            homeDirectoryUrl: homeDirectoryUrl,
            routeTable: routeTable);
    }

    func dispose() -> Void {
        try? FileManager.default.removeItem(at: self.homeDirectoryUrl);
    }

    func postConfigReload() throws -> RestHttpResponse {
        let routeOutcome: RestRouteOutcome = self.routeTable.outcome(
            method: RestConfigReloadEndpoint.routeMethod,
            path: RestConfigReloadEndpoint.routePath);
        guard case .handler(let routeHandler) = routeOutcome else {
            throw ConfigReloadJourneyFailure.reloadRouteMissing;
        }
        return try routeHandler(ConfigReloadJourney.emptyRequest(
            method: RestConfigReloadEndpoint.routeMethod,
            path: RestConfigReloadEndpoint.routePath));
    }

    func getStatusResponse() throws -> RestHttpResponse {
        let routeOutcome: RestRouteOutcome = self.routeTable.outcome(method: "GET", path: "/v1/status");
        guard case .handler(let routeHandler) = routeOutcome else {
            throw ConfigReloadJourneyFailure.statusRouteMissing;
        }
        return try routeHandler(ConfigReloadJourney.emptyRequest(
            method: "GET",
            path: "/v1/status"));
    }

    func getStatusDocument() throws -> [String: Any] {
        return try ConfigReloadJourney.decodeObject(try self.getStatusResponse());
    }

    // MARK: Fixture helpers

    /// Writes raw config bytes into the instance config file.
    static func writeRawConfigFile(_ homeDirectoryUrl: URL, rawConfigJson: String) -> Void {
        let configFileUrl: URL = homeDirectoryUrl.appendingPathComponent(".astronomical-dev/config.json");
        try? FileManager.default.createDirectory(
            at: configFileUrl.deletingLastPathComponent(),
            withIntermediateDirectories: true);
        try? Data(rawConfigJson.utf8).write(to: configFileUrl);
    }

    /// Merges the configured fields over the minimal v1 document and writes
    /// the instance config file, mirroring the Rust support's
    /// write_config_file helper.
    static func writeConfigFile(_ homeDirectoryUrl: URL, configuredFieldsJson: String) -> Void {
        let configuredFields: [String: Any];
        if let parsedFields: [String: Any] = (try? JSONSerialization.jsonObject(
            with: Data(configuredFieldsJson.utf8), options: [])) as? [String: Any] {
            configuredFields = parsedFields;
        } else {
            configuredFields = [:];
        }
        var configDocument: [String: Any] = [
            "$schema": "./astronomical-config.schema.json",
            "schema_version": 1,
            "runtime": ["model_directories": Array<String>()],
        ];
        for (fieldName, fieldValue) in configuredFields {
            configDocument[fieldName] = fieldValue;
        }
        let configDocumentBytes: Data;
        if JSONSerialization.isValidJSONObject(configDocument) {
            configDocumentBytes = (try? JSONSerialization.data(
                withJSONObject: configDocument,
                options: [.prettyPrinted, .sortedKeys])) ?? Data();
        } else {
            configDocumentBytes = Data();
        }
        let configFileUrl: URL = homeDirectoryUrl.appendingPathComponent(".astronomical-dev/config.json");
        try? FileManager.default.createDirectory(
            at: configFileUrl.deletingLastPathComponent(),
            withIntermediateDirectories: true);
        try? configDocumentBytes.write(to: configFileUrl);
    }

    /// Writes one minimal fictional Qwen-family model directory, mirroring
    /// the Rust support's write_minimal_qwen_model fixture.
    static func writeMinimalQwenModel(_ modelDirectoryUrl: URL) throws -> Void {
        let modelShardBytes: Data = Data("fictional-shard".utf8);
        try FileManager.default.createDirectory(at: modelDirectoryUrl, withIntermediateDirectories: true);
        try "{\"model_type\":\"qwen3_5_moe\",\"text_config\":{\"max_position_embeddings\":262144}}"
            .write(to: modelDirectoryUrl.appendingPathComponent("config.json"), atomically: true, encoding: .utf8);
        try modelShardBytes.write(to: modelDirectoryUrl.appendingPathComponent("model-00001.safetensors"));
        let modelIndexDocument: String =
            "{\"metadata\":{\"total_size\":\(modelShardBytes.count)},\"weight_map\":"
            + "{\"model.embed_tokens.weight\":\"model-00001.safetensors\"}}";
        try modelIndexDocument.write(
            to: modelDirectoryUrl.appendingPathComponent("model.safetensors.index.json"),
            atomically: true,
            encoding: .utf8);
        try "{\"version\":1,\"model\":{\"type\":\"BPE\"}}".write(
            to: modelDirectoryUrl.appendingPathComponent("tokenizer.json"),
            atomically: true,
            encoding: .utf8);
    }

    private static func emptyRequest(method: String, path: String) -> RestHttpRequest {
        return RestHttpRequest(
            method: method,
            path: path,
            requestTarget: path,
            headersByLowercasedName: [:],
            bodyBytes: Data());
    }

    static func decodeObject(_ response: RestHttpResponse) throws -> [String: Any] {
        let responseDocument: Any = try JSONSerialization.jsonObject(
            with: response.bodyBytes,
            options: []);
        guard let responseDocumentObject = responseDocument as? [String: Any] else {
            throw ConfigReloadJourneyFailure.responseNotAnObject(response.statusCode);
        }
        return responseDocumentObject;
    }

    static func requireConfigurationDocument(_ statusDocument: [String: Any]) throws -> [String: Any] {
        guard let configurationDocument = statusDocument["configuration"] as? [String: Any] else {
            throw ConfigReloadJourneyFailure.configurationSectionMissing;
        }
        return configurationDocument;
    }

    static func requireDiagnostics(_ configurationDocument: [String: Any]) throws -> Array<[String: Any]> {
        guard let diagnostics = configurationDocument["model_discovery_diagnostics"] as? Array<[String: Any]> else {
            throw ConfigReloadJourneyFailure.diagnosticsMissing;
        }
        return diagnostics;
    }
}

/// Typed failures of the reload journey plumbing.
enum ConfigReloadJourneyFailure: Error {

    case reloadRouteMissing;
    case statusRouteMissing;
    case responseNotAnObject(Int);
    case configurationSectionMissing;
    case diagnosticsMissing;
}
