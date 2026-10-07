import Foundation;

import IpcProtocol;
import RestContract;

/**
 * The internal reload response consumed by Astronomical's menu application,
 * the Swift port of apps/supervisor/src/config_reload_response.rs.
 */
struct RestConfigReloadResponse {

    let statusText: String;
    let messageText: String;
    let workerRestartCompleted: Bool;
    let restApiRestartRequired: Bool;
    let restartRequiredFields: Array<String>;
    let reloadedFields: Array<String>;
    let discoveredModelCount: Int;
    let workerRuntimeFeatureConfiguration: WorkerRuntimeFeatureConfiguration?;
    var candidateGeneration: String?;
    var effectiveGeneration: String?;

    static func reloaded(
        reloadedFields: Array<String>,
        discoveredModelCount: Int
    ) -> RestConfigReloadResponse {
        return RestConfigReloadResponse(
            statusText: "reloaded",
            messageText: "Config reloaded",
            workerRestartCompleted: false,
            restApiRestartRequired: false,
            restartRequiredFields: Array<String>(),
            reloadedFields: reloadedFields,
            discoveredModelCount: discoveredModelCount,
            workerRuntimeFeatureConfiguration: nil);
    }

    static func restartRequired(
        reloadedFields: Array<String>,
        restartRequiredFields: Array<String>,
        discoveredModelCount: Int
    ) -> RestConfigReloadResponse {
        return RestConfigReloadResponse(
            statusText: "restart_required",
            messageText: "Config is valid, but a full server restart is required",
            workerRestartCompleted: false,
            restApiRestartRequired: true,
            restartRequiredFields: restartRequiredFields,
            reloadedFields: reloadedFields,
            discoveredModelCount: discoveredModelCount,
            workerRuntimeFeatureConfiguration: nil);
    }

    static func workerRestartCompleted(
        reloadedFields: Array<String>,
        discoveredModelCount: Int,
        acknowledgedConfiguration: WorkerRuntimeFeatureConfiguration
    ) -> RestConfigReloadResponse {
        return RestConfigReloadResponse(
            statusText: "reloaded",
            messageText: "Config reloaded and applied by the worker",
            workerRestartCompleted: true,
            restApiRestartRequired: false,
            restartRequiredFields: Array<String>(),
            reloadedFields: reloadedFields,
            discoveredModelCount: discoveredModelCount,
            workerRuntimeFeatureConfiguration: acknowledgedConfiguration);
    }

    static func invalidConfig(_ validationError: String) -> RestConfigReloadResponse {
        return RestConfigReloadResponse.failure(
            status: "invalid_config",
            message: validationError,
            discoveredModelCount: 0);
    }

    static func busy() -> RestConfigReloadResponse {
        return RestConfigReloadResponse.failure(
            status: "busy",
            message: "A generation is active or queued; reload aborted",
            discoveredModelCount: 0);
    }

    static func failed(
        _ message: String,
        discoveredModelCount: Int
    ) -> RestConfigReloadResponse {
        return RestConfigReloadResponse.failure(
            status: "failed",
            message: message,
            discoveredModelCount: discoveredModelCount);
    }

    func withGenerations(candidate: String, effective: String?) -> RestConfigReloadResponse {
        var generationalResponse: RestConfigReloadResponse = self;
        generationalResponse.candidateGeneration = candidate;
        generationalResponse.effectiveGeneration = effective;
        return generationalResponse;
    }

    func wireValue() -> JsonWireValue {
        var reloadDocument: JsonWireObject = JsonWireObject(entries: Array<(key: String, value: JsonWireValue)>());
        reloadDocument.appendEntry(key: "status", value: .string(self.statusText));
        reloadDocument.appendEntry(key: "message", value: .string(self.messageText));
        reloadDocument.appendEntry(key: "worker_restart_completed", value: .boolean(self.workerRestartCompleted));
        reloadDocument.appendEntry(key: "rest_api_restart_required", value: .boolean(self.restApiRestartRequired));
        reloadDocument.appendEntry(
            key: "restart_required_fields",
            value: .array(self.restartRequiredFields.map({ (fieldName: String) -> JsonWireValue in
                return .string(fieldName);
            })));
        reloadDocument.appendEntry(
            key: "reloaded_fields",
            value: .array(self.reloadedFields.map({ (fieldName: String) -> JsonWireValue in
                return .string(fieldName);
            })));
        reloadDocument.appendEntry(key: "discovered_model_count", value: .unsignedInteger(UInt64(self.discoveredModelCount)));
        reloadDocument.appendEntry(
            key: "worker_runtime_feature_configuration",
            value: self.workerRuntimeFeatureConfiguration?.wireValue() ?? .null);
        reloadDocument.appendEntry(
            key: "candidate_generation",
            value: self.candidateGeneration.map({ (candidateGeneration: String) -> JsonWireValue in
                return .string(candidateGeneration);
            }) ?? .null);
        reloadDocument.appendEntry(
            key: "effective_generation",
            value: self.effectiveGeneration.map({ (effectiveGeneration: String) -> JsonWireValue in
                return .string(effectiveGeneration);
            }) ?? .null);
        return .object(reloadDocument);
    }

    private static func failure(
        status: String,
        message: String,
        discoveredModelCount: Int
    ) -> RestConfigReloadResponse {
        return RestConfigReloadResponse(
            statusText: status,
            messageText: message,
            workerRestartCompleted: false,
            restApiRestartRequired: false,
            restartRequiredFields: Array<String>(),
            reloadedFields: Array<String>(),
            discoveredModelCount: discoveredModelCount,
            workerRuntimeFeatureConfiguration: nil);
    }
}
