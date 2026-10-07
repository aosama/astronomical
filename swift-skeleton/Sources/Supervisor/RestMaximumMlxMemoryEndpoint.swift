import Foundation;

import AstronomicalConfig;
import IpcProtocol;
import RestContract;

/**
 * The context the daemon hands the memory-limit route: the live supervisor,
 * the config resolver, and the shared transition state, or nil when no
 * worker control exists and the route must advertise itself as absent.
 */
public struct RestMaximumMlxMemoryRouteContext: @unchecked Sendable {

    let workerControl: WorkerSupervisor;
    let runtimeConfigResolver: ResolvedRuntimeConfigResolver;
    let transitionState: ConfigTransitionState;

    public init(
        workerControl: WorkerSupervisor,
        runtimeConfigResolver: ResolvedRuntimeConfigResolver,
        transitionState: ConfigTransitionState
    ) {
        self.workerControl = workerControl;
        self.runtimeConfigResolver = runtimeConfigResolver;
        self.transitionState = transitionState;
    }
}

/// Updates the optional MLX memory ceiling end to end, migrating
/// apps/supervisor/src/maximum_mlx_memory_endpoint.rs:
/// PUT /v1/config/maximum-mlx-memory persists the candidate document
/// atomically, applies or queues the live ceiling on the worker, and answers
/// 200 applied, 202 queued, 400 for an invalid or rejected ceiling, 409
/// while another memory transition or unrelated config change is in flight,
/// 404 without live control, and 500 when control or persistence fails.
public enum RestMaximumMlxMemoryEndpoint {

    public static let routeMethod: String = "PUT";
    public static let routePath: String = "/v1/config/maximum-mlx-memory";

    public static func handle(
        _ request: RestHttpRequest,
        memoryContext: RestMaximumMlxMemoryRouteContext?
    ) -> RestHttpResponse {
        guard let memoryContext = memoryContext else {
            return RestHttpResponse.text(statusCode: 404, body: "live worker control is unavailable");
        }
        let requestedMaximumMlxMemoryGb: UInt64?;
        do {
            requestedMaximumMlxMemoryGb = try RestMaximumMlxMemoryEndpoint.decodeRequestedMaximumMlxMemoryGb(
                request.bodyBytes);
        } catch {
            return RestHttpResponse.text(statusCode: 400, body: "invalid maximum-mlx-memory request body");
        }
        return memoryContext.transitionState.withTransitionGuard({ () -> RestHttpResponse in
            return RestMaximumMlxMemoryEndpoint.updateMaximumMlxMemory(
                requestedMaximumMlxMemoryGb,
                memoryContext: memoryContext);
        });
    }

    /// Decodes `{"maximum_mlx_memory_gb": <u64>|null}` and rejects unknown
    /// fields, mirroring the serde deny_unknown_fields request contract.
    private static func decodeRequestedMaximumMlxMemoryGb(_ bodyBytes: Data) throws -> UInt64? {
        let requestDocument: Any = try JSONSerialization.jsonObject(with: bodyBytes, options: []);
        guard let requestObject: [String: Any] = requestDocument as? [String: Any] else {
            throw RestMaximumMlxMemoryFailure.requestBodyNotAnObject;
        }
        for requestFieldName: String in requestObject.keys {
            if requestFieldName != "maximum_mlx_memory_gb" {
                throw RestMaximumMlxMemoryFailure.unknownRequestField(requestFieldName);
            }
        }
        guard let rawRequestedMaximumMlxMemoryGb: Any = requestObject["maximum_mlx_memory_gb"] else {
            return nil;
        }
        if rawRequestedMaximumMlxMemoryGb is NSNull {
            return nil;
        }
        guard let requestedMaximumMlxMemoryGb = rawRequestedMaximumMlxMemoryGb as? UInt64 else {
            throw RestMaximumMlxMemoryFailure.requestedValueNotAnUnsignedInteger;
        }
        return requestedMaximumMlxMemoryGb;
    }

    private static func updateMaximumMlxMemory(
        _ requestedMaximumMlxMemoryGb: UInt64?,
        memoryContext: RestMaximumMlxMemoryRouteContext
    ) -> RestHttpResponse {
        if memoryContext.transitionState.currentPendingMemoryConfigGeneration() != nil {
            return RestMaximumMlxMemoryEndpoint.memoryDocumentResponse(
                statusCode: 409,
                requestedMaximumMlxMemoryGb: requestedMaximumMlxMemoryGb,
                workerHealthSnapshot: memoryContext.workerControl.workerHealthSnapshot(),
                message: "a queued MLX memory setting is still awaiting worker acknowledgement");
        }
        let workerHealthSnapshot: WorkerHealthSnapshot = memoryContext.workerControl.workerHealthSnapshot();
        let requestedMlxMemoryCeilingBytes: UInt64;
        if let requestedMaximumMlxMemoryGb = requestedMaximumMlxMemoryGb {
            do {
                requestedMlxMemoryCeilingBytes = try MaximumMlxMemory.maximumMlxMemoryGbToBytes(
                    requestedMaximumMlxMemoryGb);
            } catch {
                return RestMaximumMlxMemoryEndpoint.invalidRequestResponse(
                    requestedMaximumMlxMemoryGb: requestedMaximumMlxMemoryGb,
                    workerHealthSnapshot: workerHealthSnapshot,
                    message: String(describing: error));
            }
        } else {
            requestedMlxMemoryCeilingBytes = workerHealthSnapshot.machineMlxMemoryCeilingBytes;
        }
        if workerHealthSnapshot.machineMlxMemoryCeilingBytes == 0
            || requestedMlxMemoryCeilingBytes > workerHealthSnapshot.machineMlxMemoryCeilingBytes
            || requestedMlxMemoryCeilingBytes < workerHealthSnapshot.minimumMlxMemoryCeilingBytes {
            return RestMaximumMlxMemoryEndpoint.invalidRequestResponse(
                requestedMaximumMlxMemoryGb: requestedMaximumMlxMemoryGb,
                workerHealthSnapshot: workerHealthSnapshot,
                message: "requested MLX memory ceiling is outside the worker's reported bounds");
        }

        let priorResolvedConfig: ResolvedRuntimeConfig = memoryContext.transitionState.currentReloadableConfig();
        let configUpdate: MaximumMlxMemoryConfigUpdate;
        do {
            configUpdate = try MaximumMlxMemory.prepareMaximumMlxMemoryGbUpdate(
                stateDirectory: memoryContext.runtimeConfigResolver.stateDirectory,
                maximumMlxMemoryGb: requestedMaximumMlxMemoryGb);
        } catch {
            return RestMaximumMlxMemoryEndpoint.memoryDocumentResponse(
                statusCode: 500,
                requestedMaximumMlxMemoryGb: requestedMaximumMlxMemoryGb,
                workerHealthSnapshot: workerHealthSnapshot,
                message: "could not prepare the MLX memory setting; inspect local diagnostics");
        }
        let candidateResolvedConfig: ResolvedRuntimeConfig;
        do {
            let candidateConfig: AstronomicalConfig = try AstronomicalConfig.loadFromV1Bytes(
                instancePaths: memoryContext.runtimeConfigResolver.resolvedInstancePaths,
                configBytes: configUpdate.candidateConfigBytes);
            candidateResolvedConfig = try memoryContext.runtimeConfigResolver.resolve(userConfig: candidateConfig);
        } catch {
            return RestMaximumMlxMemoryEndpoint.memoryDocumentResponse(
                statusCode: 500,
                requestedMaximumMlxMemoryGb: requestedMaximumMlxMemoryGb,
                workerHealthSnapshot: workerHealthSnapshot,
                message: "prepared memory setting could not be resolved; inspect local diagnostics");
        }
        if !RestMaximumMlxMemoryEndpoint.memoryIsTheOnlyCandidateChange(
            priorResolvedConfig: priorResolvedConfig,
            candidateResolvedConfig: candidateResolvedConfig) {
            return RestMaximumMlxMemoryEndpoint.memoryDocumentResponse(
                statusCode: 409,
                requestedMaximumMlxMemoryGb: requestedMaximumMlxMemoryGb,
                workerHealthSnapshot: workerHealthSnapshot,
                message: "other configuration changes are pending; reload the complete configuration first");
        }
        do {
            try MaximumMlxMemory.commitMaximumMlxMemoryGbUpdate(
                stateDirectory: memoryContext.runtimeConfigResolver.stateDirectory,
                configUpdate: configUpdate);
        } catch let commitError as AstronomicalConfigError {
            let conflictStatusCode: Int;
            let failureMessage: String;
            if case .configChangedDuringUpdate = commitError {
                conflictStatusCode = 409;
                failureMessage = "configuration changed during the memory update; retry after reloading";
            } else {
                conflictStatusCode = 500;
                failureMessage = "could not prepare the MLX memory setting; inspect local diagnostics";
            }
            return RestMaximumMlxMemoryEndpoint.memoryDocumentResponse(
                statusCode: conflictStatusCode,
                requestedMaximumMlxMemoryGb: requestedMaximumMlxMemoryGb,
                workerHealthSnapshot: workerHealthSnapshot,
                message: failureMessage);
        } catch {
            return RestMaximumMlxMemoryEndpoint.memoryDocumentResponse(
                statusCode: 500,
                requestedMaximumMlxMemoryGb: requestedMaximumMlxMemoryGb,
                workerHealthSnapshot: workerHealthSnapshot,
                message: "could not prepare the MLX memory setting; inspect local diagnostics");
        }
        let candidateConfigurationGeneration: String = candidateResolvedConfig.configurationGeneration;
        memoryContext.transitionState.setPendingMemoryConfigGeneration(candidateConfigurationGeneration);
        memoryContext.workerControl.stageMemoryConfigurationGeneration(candidateConfigurationGeneration);
        let updateOutcomeResult: Result<MlxMemoryLimitUpdateOutcome, Error>;
        do {
            updateOutcomeResult = .success(try memoryContext.workerControl.updateMlxMemoryLimit(
                requestedMlxMemoryCeilingBytes,
                configurationGeneration: candidateConfigurationGeneration));
        } catch {
            updateOutcomeResult = .failure(error);
        }
        switch (updateOutcomeResult) {
        case .success(let updateOutcome):
            switch (updateOutcome) {
            case .applied, .queued:
                return RestMaximumMlxMemoryEndpoint.finishAppliedUpdate(
                    updateOutcome,
                    requestedMaximumMlxMemoryGb: requestedMaximumMlxMemoryGb,
                    memoryContext: memoryContext,
                    candidateResolvedConfig: candidateResolvedConfig,
                    candidateConfigBytes: configUpdate.candidateConfigBytes,
                    priorResolvedConfig: priorResolvedConfig);
            case .rejected:
                memoryContext.transitionState.setPendingMemoryConfigGeneration(nil);
                memoryContext.workerControl.recordMemoryConfigurationGeneration(
                    candidateConfigurationGeneration, .rejected);
                MaximumMlxMemoryTransaction.retainRejectedPersistedConfig(
                    transitionState: memoryContext.transitionState,
                    resolver: memoryContext.runtimeConfigResolver);
                let rejectedWorkerHealthSnapshot: WorkerHealthSnapshot = memoryContext.workerControl.workerHealthSnapshot();
                return RestMaximumMlxMemoryEndpoint.invalidRequestResponse(
                    requestedMaximumMlxMemoryGb: requestedMaximumMlxMemoryGb,
                    workerHealthSnapshot: rejectedWorkerHealthSnapshot,
                    message: rejectedWorkerHealthSnapshot.mlxMemoryLimitError
                        ?? "worker rejected the requested MLX memory ceiling");
            }
        case .failure:
            memoryContext.transitionState.setPendingMemoryConfigGeneration(nil);
            memoryContext.workerControl.recordMemoryConfigurationGeneration(
                candidateConfigurationGeneration, .rejected);
            MaximumMlxMemoryTransaction.retainRejectedPersistedConfig(
                transitionState: memoryContext.transitionState,
                resolver: memoryContext.runtimeConfigResolver);
            return RestMaximumMlxMemoryEndpoint.memoryDocumentResponse(
                statusCode: 500,
                requestedMaximumMlxMemoryGb: requestedMaximumMlxMemoryGb,
                workerHealthSnapshot: memoryContext.workerControl.workerHealthSnapshot(),
                message: "worker control failed");
        }
    }

    private static func finishAppliedUpdate(
        _ updateOutcome: MlxMemoryLimitUpdateOutcome,
        requestedMaximumMlxMemoryGb: UInt64?,
        memoryContext: RestMaximumMlxMemoryRouteContext,
        candidateResolvedConfig: ResolvedRuntimeConfig,
        candidateConfigBytes: Data,
        priorResolvedConfig: ResolvedRuntimeConfig
    ) -> RestHttpResponse {
        let candidateConfigurationGeneration: String = candidateResolvedConfig.configurationGeneration;
        memoryContext.workerControl.recordMemoryConfigurationGeneration(
            candidateConfigurationGeneration, updateOutcome);
        MaximumMlxMemoryTransaction.commitAppliedConfigSnapshots(
            memoryContext.transitionState,
            resolver: memoryContext.runtimeConfigResolver,
            candidateResolvedConfig: candidateResolvedConfig,
            candidateConfigBytes: candidateConfigBytes);
        if updateOutcome == .queued {
            let reconcileSupervisor: WorkerSupervisor = memoryContext.workerControl;
            let reconcileTransitionState: ConfigTransitionState = memoryContext.transitionState;
            let reconcileResolver: ResolvedRuntimeConfigResolver = memoryContext.runtimeConfigResolver;
            let reconcileThread: Thread = Thread {
                MaximumMlxMemoryTransaction.reconcileQueuedMemoryConfig(
                    supervisor: reconcileSupervisor,
                    transitionState: reconcileTransitionState,
                    resolver: reconcileResolver,
                    candidateResolvedConfig: candidateResolvedConfig,
                    candidateConfigBytes: candidateConfigBytes,
                    priorResolvedConfig: priorResolvedConfig);
            };
            reconcileThread.name = "maximum-mlx-memory-reconcile";
            reconcileThread.start();
        } else {
            memoryContext.transitionState.setPendingMemoryConfigGeneration(nil);
        }
        let updatedWorkerHealthSnapshot: WorkerHealthSnapshot = memoryContext.workerControl.workerHealthSnapshot();
        let isQueued: Bool = updatedWorkerHealthSnapshot.pendingMlxMemoryCeilingBytes != nil;
        let responseStatusCode: Int = isQueued ? 202 : 200;
        let responseMessage: String = isQueued
            ? "MLX memory setting persisted and queued until generation finalizes"
            : "MLX memory setting persisted and applied";
        return RestMaximumMlxMemoryEndpoint.memoryDocumentResponse(
            statusCode: responseStatusCode,
            requestedMaximumMlxMemoryGb: requestedMaximumMlxMemoryGb,
            workerHealthSnapshot: updatedWorkerHealthSnapshot,
            message: responseMessage);
    }

    /// The candidate may only move the memory setting; any other resolved
    /// difference demands the full reload first.
    private static func memoryIsTheOnlyCandidateChange(
        priorResolvedConfig: ResolvedRuntimeConfig,
        candidateResolvedConfig: ResolvedRuntimeConfig
    ) -> Bool {
        guard case .noWorkerRestart(let reloadedFields, _) = ConfigReloadDiff.compare(
            current: priorResolvedConfig,
            candidate: candidateResolvedConfig) else {
            return false;
        }
        return reloadedFields.allSatisfy({ (fieldName: String) -> Bool in
            return fieldName == "maximum_mlx_memory_gb";
        });
    }

    private static func invalidRequestResponse(
        requestedMaximumMlxMemoryGb: UInt64?,
        workerHealthSnapshot: WorkerHealthSnapshot,
        message: String
    ) -> RestHttpResponse {
        return RestMaximumMlxMemoryEndpoint.memoryDocumentResponse(
            statusCode: 400,
            requestedMaximumMlxMemoryGb: requestedMaximumMlxMemoryGb,
            workerHealthSnapshot: workerHealthSnapshot,
            message: message);
    }

    private static func memoryDocumentResponse(
        statusCode: Int,
        requestedMaximumMlxMemoryGb: UInt64?,
        workerHealthSnapshot: WorkerHealthSnapshot,
        message: String
    ) -> RestHttpResponse {
        guard let memoryResponse: RestHttpResponse = try? RestHttpResponse.json(
            statusCode: statusCode,
            wireValue: RestMaximumMlxMemoryEndpoint.memoryDocumentWireValue(
                requestedMaximumMlxMemoryGb: requestedMaximumMlxMemoryGb,
                workerHealthSnapshot: workerHealthSnapshot,
                message: message)) else {
            return RestHttpResponse.text(statusCode: statusCode, body: message);
        }
        return memoryResponse;
    }

    private static func memoryDocumentWireValue(
        requestedMaximumMlxMemoryGb: UInt64?,
        workerHealthSnapshot: WorkerHealthSnapshot,
        message: String
    ) -> JsonWireValue {
        var memoryDocument: JsonWireObject = JsonWireObject(entries: Array<(key: String, value: JsonWireValue)>());
        memoryDocument.appendEntry(
            key: "configured_maximum_mlx_memory_gb",
            value: requestedMaximumMlxMemoryGb.map({ (configuredMaximumMlxMemoryGb: UInt64) -> JsonWireValue in
                return .unsignedInteger(configuredMaximumMlxMemoryGb);
            }) ?? .null);
        memoryDocument.appendEntry(
            key: "effective_mlx_memory_ceiling_bytes",
            value: .unsignedInteger(workerHealthSnapshot.effectiveMlxMemoryCeilingBytes));
        memoryDocument.appendEntry(
            key: "machine_mlx_memory_ceiling_bytes",
            value: .unsignedInteger(workerHealthSnapshot.machineMlxMemoryCeilingBytes));
        memoryDocument.appendEntry(
            key: "minimum_mlx_memory_ceiling_bytes",
            value: .unsignedInteger(workerHealthSnapshot.minimumMlxMemoryCeilingBytes));
        memoryDocument.appendEntry(
            key: "pending_mlx_memory_ceiling_bytes",
            value: workerHealthSnapshot.pendingMlxMemoryCeilingBytes.map({ (pendingMlxMemoryCeilingBytes: UInt64) -> JsonWireValue in
                return .unsignedInteger(pendingMlxMemoryCeilingBytes);
            }) ?? .null);
        memoryDocument.appendEntry(key: "message", value: .string(message));
        return .object(memoryDocument);
    }
}

/// Typed failures of the memory-limit request decode.
enum RestMaximumMlxMemoryFailure: Error {

    case requestBodyNotAnObject;
    case unknownRequestField(String);
    case requestedValueNotAnUnsignedInteger;
}
