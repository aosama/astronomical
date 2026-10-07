import Foundation;

import AstronomicalConfig;
import IpcProtocol;
import RestContract;

/**
 * The endpoint routes the daemon serves at startup. Mirrors the always-on
 * spine of the Rust application router: the health liveness probe and the
 * readiness probe driven by the worker health state. Generation and model
 * endpoints attach as their slices land.
 */
public enum RestEndpointRoutes {

    public static func foundationRouteTable(
        readinessProvider: @escaping @Sendable () -> WorkerHealthStatus
    ) -> RestRouteTable {
        var routeTable: RestRouteTable = RestRouteTable();
        routeTable.register(
            method: "GET",
            path: "/health",
            handler: { (_ request: RestHttpRequest) -> RestHttpResponse in
                return RestHttpResponse.text(statusCode: 200, body: "ok");
            });
        routeTable.register(
            method: "GET",
            path: "/ready",
            handler: { (_ request: RestHttpRequest) -> RestHttpResponse in
                let workerHealthStatus: WorkerHealthStatus = readinessProvider();
                let readinessStatusCode: Int = workerHealthStatus.isReady() ? 200 : 503;
                return RestHttpResponse.text(statusCode: readinessStatusCode, body: workerHealthStatus.readinessText());
            });
        return routeTable;
    }

    /// The serving route table: the foundation probes plus the model
    /// advertisement, cache statistics, and instance status endpoints built
    /// from the resolved runtime configuration and the worker health state.
    /// Until the config-reload slice lands, one resolved snapshot is both the
    /// configured and the resolved view; the status contract already keeps
    /// them apart so reload only rewires this call.
    public static func servingRouteTable(
        resolvedRuntimeConfig: ResolvedRuntimeConfig,
        workerHealthState: WorkerHealthState,
        instancePaths: AstronomicalInstancePaths,
        buildIdentity: ApplicationBuildIdentity,
        configurationValidationError: String? = nil,
        chatContext: RestChatRouteContext? = nil,
        responsesContext: RestResponsesRouteContext? = nil,
        embeddingsContext: RestEmbeddingsRouteContext? = nil,
        imageContext: RestImageGenerationRouteContext? = nil,
        cacheClearContext: RestCacheClearRouteContext? = nil,
        shutdownController: ShutdownController? = nil,
        memoryContext: RestMaximumMlxMemoryRouteContext? = nil,
        configReloadContext: RestConfigReloadRouteContext? = nil
    ) -> RestRouteTable {
        var routeTable: RestRouteTable = RestEndpointRoutes.foundationRouteTable(readinessProvider: {
            return workerHealthState.currentSnapshot().status;
        });
        // With reload support the listing and routing surfaces read the live
        // reloadable snapshot, so one reloaded discovery snapshot reaches
        // both, exactly like ApplicationState::discovered_models_snapshot.
        let baseResolvedRuntimeConfig: ResolvedRuntimeConfig = resolvedRuntimeConfig;
        var liveResolvedRuntimeConfigProvider: @Sendable () -> ResolvedRuntimeConfig =
            { return baseResolvedRuntimeConfig; };
        if let reloadContext: RestConfigReloadRouteContext = configReloadContext {
            liveResolvedRuntimeConfigProvider = {
                return reloadContext.transitionState.currentReloadableConfig();
            };
        }
        let servingResolvedRuntimeConfigProvider: @Sendable () -> ResolvedRuntimeConfig =
            liveResolvedRuntimeConfigProvider;
        routeTable.register(
            method: "GET",
            path: "/v1/status",
            handler: { (_ request: RestHttpRequest) -> RestHttpResponse in
                // With a reload context the status triple follows the live
                // transition state; without one there is no accepted
                // persisted snapshot, so the configured view stays null —
                // exactly like the Rust plain application builders.
                let liveConfiguredRuntimeConfig: ResolvedRuntimeConfig? = configReloadContext.flatMap(
                    { (reloadContext: RestConfigReloadRouteContext) -> ResolvedRuntimeConfig? in
                        return reloadContext.transitionState.currentConfiguredConfigSnapshot()
                            ?? reloadContext.transitionState.currentReloadableConfig();
                    });
                let liveResolvedRuntimeConfig: ResolvedRuntimeConfig? = configReloadContext.map(
                    { (reloadContext: RestConfigReloadRouteContext) -> ResolvedRuntimeConfig in
                        return reloadContext.transitionState.currentReloadableConfig();
                    });
                let liveValidationError: String? = configReloadContext.flatMap(
                    { (reloadContext: RestConfigReloadRouteContext) -> String? in
                        return reloadContext.transitionState.currentConfigurationValidationError();
                    }) ?? configurationValidationError;
                return try RestStatusResponse.statusResponse(
                    configuredRuntimeConfig: liveConfiguredRuntimeConfig,
                    resolvedRuntimeConfig: liveResolvedRuntimeConfig,
                    workerHealthSnapshot: workerHealthState.currentSnapshot(),
                    configurationValidationError: liveValidationError,
                    instancePaths: instancePaths,
                    buildIdentity: buildIdentity);            });
        routeTable.register(
            method: "GET",
            path: "/v1/models",
            handler: { (_ request: RestHttpRequest) -> RestHttpResponse in
                return try RestEndpointRoutes.advertiseModelsResponse(
                    resolvedRuntimeConfig: servingResolvedRuntimeConfigProvider(),
                    workerHealthState: workerHealthState);
            });
        routeTable.registerPrefix(
            method: "GET",
            pathPrefix: "/v1/models/",
            handler: { (request: RestHttpRequest) -> RestHttpResponse in
                let requestedModelId: String = String(request.path.dropFirst("/v1/models/".count));
                let advertisedModels: Array<OpenAiModel> = try RestEndpointRoutes.advertiseModels(
                    resolvedRuntimeConfig: servingResolvedRuntimeConfigProvider(),
                    workerHealthState: workerHealthState);
                let resolvedModelId: String = RestAdvertisedModels.resolveRequestedModelId(
                    requestedModelId: requestedModelId,
                    advertisedModels: advertisedModels);
                for advertisedModel: OpenAiModel in advertisedModels {
                    if advertisedModel.id() == resolvedModelId {
                        return try RestHttpResponse.json(statusCode: 200, wireValue: advertisedModel.wireValue());
                    }
                }
                return try RestEndpointFailure(
                    statusCode: 404,
                    message: "the requested model \(requestedModelId) is not advertised by this instance")
                    .envelopeResponse();
            });
        if let chatContext = chatContext {
            routeTable.register(
                method: RestChatCompletionEndpoint.routeMethod,
                path: RestChatCompletionEndpoint.routePath,
                handler: { (request: RestHttpRequest) -> RestHttpResponse in
                    return RestChatCompletionEndpoint.handle(request, chatContext: chatContext);
                });
        }
        if let responsesContext = responsesContext {
            routeTable.register(
                method: RestResponsesEndpoint.routeMethod,
                path: RestResponsesEndpoint.routePath,
                handler: { (request: RestHttpRequest) -> RestHttpResponse in
                    return RestResponsesEndpoint.handle(request, responsesContext: responsesContext);
                });
        }
        if let embeddingsContext = embeddingsContext {
            routeTable.register(
                method: RestEmbeddingsEndpoint.routeMethod,
                path: RestEmbeddingsEndpoint.routePath,
                handler: { (request: RestHttpRequest) -> RestHttpResponse in
                    return RestEmbeddingsEndpoint.handle(
                        request,
                        embeddingsContext: embeddingsContext);
                });
        }
        if let imageContext = imageContext {
            routeTable.register(
                method: RestImageGenerationEndpoint.routeMethod,
                path: RestImageGenerationEndpoint.routePath,
                handler: { (request: RestHttpRequest) -> RestHttpResponse in
                    return RestImageGenerationEndpoint.handle(
                        request,
                        imageContext: imageContext);
                });
        }
        routeTable.register(
            method: "GET",
            path: "/v1/cache/stats",
            handler: { (_ request: RestHttpRequest) -> RestHttpResponse in
                return try RestEndpointRoutes.cacheStatsResponse(
                    workerHealthState: workerHealthState);
            });
        if let cacheClearContext = cacheClearContext {
            routeTable.register(
                method: RestCacheClearEndpoint.routeMethod,
                path: RestCacheClearEndpoint.routePath,
                handler: { (request: RestHttpRequest) -> RestHttpResponse in
                    return try RestCacheClearEndpoint.handle(
                        request,
                        cacheClearContext: cacheClearContext);
                });
        }
        if let shutdownController = shutdownController {
            routeTable.register(
                method: RestShutdownControlEndpoint.routeMethod,
                path: RestShutdownControlEndpoint.routePath,
                handler: { (request: RestHttpRequest) -> RestHttpResponse in
                    return try RestShutdownControlEndpoint.handle(
                        request,
                        shutdownController: shutdownController);
                });
        }
        if let memoryContext = memoryContext {
            routeTable.register(
                method: RestMaximumMlxMemoryEndpoint.routeMethod,
                path: RestMaximumMlxMemoryEndpoint.routePath,
                handler: { (request: RestHttpRequest) -> RestHttpResponse in
                    return RestMaximumMlxMemoryEndpoint.handle(request, memoryContext: memoryContext);
                });
        }
        if let configReloadContext = configReloadContext {
            routeTable.register(
                method: RestConfigReloadEndpoint.routeMethod,
                path: RestConfigReloadEndpoint.routePath,
                handler: { (request: RestHttpRequest) -> RestHttpResponse in
                    return RestConfigReloadEndpoint.handle(request, reloadContext: configReloadContext);
                });
        }
        return routeTable;
    }

    private static func advertiseModels(
        resolvedRuntimeConfig: ResolvedRuntimeConfig,
        workerHealthState: WorkerHealthState
    ) throws -> Array<OpenAiModel> {
        return try RestAdvertisedModels.advertise(
            discoveredModels: resolvedRuntimeConfig.discoveredModels,
            workerHealthSnapshot: workerHealthState.currentSnapshot(),
            createdAtIndexedSeconds: UInt64(Date().timeIntervalSince1970));
    }

    private static func advertiseModelsResponse(
        resolvedRuntimeConfig: ResolvedRuntimeConfig,
        workerHealthState: WorkerHealthState
    ) throws -> RestHttpResponse {
        do {
            let advertisedModels: Array<OpenAiModel> = try RestEndpointRoutes.advertiseModels(
                resolvedRuntimeConfig: resolvedRuntimeConfig,
                workerHealthState: workerHealthState);
            return try RestHttpResponse.json(
                statusCode: 200,
                wireValue: OpenAiModelList.fromModels(models: advertisedModels).wireValue());
        } catch let advertisementFailure as OpenAiModelValidationError {
            return try RestEndpointFailure(
                statusCode: 500,
                message: "the discovered model metadata failed validation: \(advertisementFailure)")
                .envelopeResponse();
        }
    }

    private static func cacheStatsResponse(
        workerHealthState: WorkerHealthState
    ) throws -> RestHttpResponse {
        // Every statistic is worker-reported: the supervisor forwards the
        // latest persistent prompt-cache observation, zeroed until the
        // worker publishes one, exactly like the Rust cache_stats endpoint.
        let persistentPromptCacheStats: WorkerPersistentPromptCacheStats? = workerHealthState
            .currentSnapshot()
            .persistentPromptCacheStats;
        let hits: UInt64 = persistentPromptCacheStats?.persistentPromptCacheHits ?? 0;
        let misses: UInt64 = persistentPromptCacheStats?.persistentPromptCacheMisses ?? 0;
        let queryCount: UInt64 = hits &+ misses;
        let hitRate: Double;
        if queryCount == 0 {
            hitRate = 0;
        } else {
            hitRate = (Double(hits) / Double(queryCount) * 10_000).rounded() / 10_000;
        }
        var statsObject: JsonWireObject = JsonWireObject(entries: Array());
        statsObject.appendEntry(key: "persistent_prompt_cache_hits", value: .unsignedInteger(hits));
        statsObject.appendEntry(key: "persistent_prompt_cache_misses", value: .unsignedInteger(misses));
        statsObject.appendEntry(
            key: "persistent_prompt_cache_tokens_saved",
            value: .unsignedInteger(persistentPromptCacheStats?.persistentPromptCacheTokensSaved ?? 0));
        statsObject.appendEntry(
            key: "persistent_prompt_cache_block_token_count",
            value: .unsignedInteger(persistentPromptCacheStats?.persistentPromptCacheBlockTokenCount ?? 0));
        statsObject.appendEntry(
            key: "persistent_prompt_cache_sequence_state_block_count",
            value: .unsignedInteger(persistentPromptCacheStats?.persistentPromptCacheSequenceStateBlockCount ?? 0));
        statsObject.appendEntry(
            key: "persistent_prompt_cache_boundary_state_snapshot_count",
            value: .unsignedInteger(persistentPromptCacheStats?.persistentPromptCacheBoundaryStateSnapshotCount ?? 0));
        statsObject.appendEntry(
            key: "persistent_prompt_cache_visual_embedding_count",
            value: .unsignedInteger(persistentPromptCacheStats?.persistentPromptCacheVisualEmbeddingCount ?? 0));
        statsObject.appendEntry(
            key: "persistent_prompt_cache_total_size_bytes",
            value: .unsignedInteger(persistentPromptCacheStats?.persistentPromptCacheTotalSizeBytes ?? 0));
        statsObject.appendEntry(
            key: "persistent_prompt_cache_visual_embedding_total_size_bytes",
            value: .unsignedInteger(persistentPromptCacheStats?.persistentPromptCacheVisualEmbeddingTotalSizeBytes ?? 0));
        statsObject.appendEntry(
            key: "persistent_prompt_cache_maximum_size_bytes",
            value: .unsignedInteger(persistentPromptCacheStats?.persistentPromptCacheMaximumSizeBytes ?? 0));
        statsObject.appendEntry(key: "persistent_prompt_cache_hit_rate", value: .double(hitRate));
        statsObject.appendEntry(
            key: "persistent_prompt_cache_visual_embedding_hits",
            value: .unsignedInteger(persistentPromptCacheStats?.persistentPromptCacheVisualEmbeddingHits ?? 0));
        statsObject.appendEntry(
            key: "persistent_prompt_cache_visual_embedding_misses",
            value: .unsignedInteger(persistentPromptCacheStats?.persistentPromptCacheVisualEmbeddingMisses ?? 0));
        statsObject.appendEntry(
            key: "persistent_prompt_cache_visual_embedding_rows_loaded",
            value: .unsignedInteger(persistentPromptCacheStats?.persistentPromptCacheVisualEmbeddingRowsLoaded ?? 0));
        statsObject.appendEntry(
            key: "persistent_prompt_cache_partial_tail_hits",
            value: .unsignedInteger(persistentPromptCacheStats?.persistentPromptCachePartialTailHits ?? 0));
        let pendingCacheClear: PendingPromptCacheClear? = workerHealthState
            .currentSnapshot()
            .pendingPromptCacheClear;
        statsObject.appendEntry(
            key: "pending_cache_clear",
            value: pendingCacheClear.map({ (pendingClear: PendingPromptCacheClear) -> JsonWireValue in
                var pendingClearObject: JsonWireObject = JsonWireObject(entries: Array());
                pendingClearObject.appendEntry(
                    key: "model_id",
                    value: pendingClear.modelId.map({ (pendingModelId: String) -> JsonWireValue in
                        return .string(pendingModelId);
                    }) ?? .null);
                return .object(pendingClearObject);
            }) ?? .null);
        return try RestHttpResponse.json(statusCode: 200, wireValue: .object(statsObject));
    }
}
