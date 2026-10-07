import Foundation;

import AstronomicalConfig;
import IpcProtocol;

/**
 * Assembles the GET /v1/status document, porting apps/supervisor/src/status_endpoint.rs.
 *
 * Sections whose sources have not landed stay out of the document entirely
 * rather than answered with placeholders: activity, serving_session,
 * persistent_prompt_cache worker-event statistics, and per-request progress
 * join when the worker session machinery (E2) and the cache statistics pump
 * (E4) publish them. The cache zero-state lives on /v1/cache/stats meanwhile.
 */
public enum RestStatusResponse {

    public static func statusResponse(
        configuredRuntimeConfig: ResolvedRuntimeConfig?,
        resolvedRuntimeConfig: ResolvedRuntimeConfig,
        workerHealthSnapshot: WorkerHealthSnapshot,
        configurationValidationError: String?,
        instancePaths: AstronomicalInstancePaths,
        buildIdentity: ApplicationBuildIdentity
    ) throws -> RestHttpResponse {
        var statusObject: JsonWireObject = JsonWireObject(entries: Array<(key: String, value: JsonWireValue)>());
        statusObject.appendEntry(
            key: "application",
            value: RestStatusResponse.applicationSection(
                instancePaths: instancePaths,
                buildIdentity: buildIdentity));
        statusObject.appendEntry(key: "status", value: .string(workerHealthSnapshot.status.readinessText()));
        statusObject.appendEntry(key: "activity", value: .string(workerHealthSnapshot.activity.activityText()));
        if let activeRequestProgress: ActiveRequestProgress = workerHealthSnapshot.activeRequestProgress {
            statusObject.appendEntry(
                key: "progress",
                value: RestStatusResponse.progressSection(activeRequestProgress));
        }
        statusObject.appendEntry(
            key: "worker_runtime_feature_configuration_applied",
            value: .boolean(workerHealthSnapshot.workerRuntimeFeatureConfiguration != nil));
        statusObject.appendEntry(
            key: "worker_runtime_feature_configuration",
            value: workerHealthSnapshot.workerRuntimeFeatureConfiguration?.wireValue() ?? .null);
        let configurationSummary: ConfigurationStatusSummary = ConfigurationStatusSummary.fromParts(
            configuredRuntimeConfig: configuredRuntimeConfig,
            resolvedRuntimeConfig: resolvedRuntimeConfig,
            workerHealthSnapshot: workerHealthSnapshot,
            configurationValidationError: configurationValidationError);
        statusObject.appendEntry(key: "configuration", value: configurationSummary.wireValue());
        statusObject.appendEntry(key: "configured_generation", value: RestStatusResponse.optionalTextWireValue(configurationSummary.configuredGeneration));
        statusObject.appendEntry(key: "resolved_generation", value: RestStatusResponse.optionalTextWireValue(configurationSummary.resolvedGeneration));
        statusObject.appendEntry(key: "effective_generation", value: RestStatusResponse.optionalTextWireValue(configurationSummary.effectiveGeneration));
        if let readyModelId: String = workerHealthSnapshot.readyModelId {
            statusObject.appendEntry(key: "ready_model_id", value: .string(readyModelId));
            statusObject.appendEntry(
                key: "ready_model_size_bytes",
                value: RestStatusResponse.readyModelSizeWireValue(
                    readyModelId: readyModelId,
                    configuredRuntimeConfig: configuredRuntimeConfig,
                    resolvedRuntimeConfig: resolvedRuntimeConfig));
        }
        let expertMemoryModeWireValue: JsonWireValue;
        if let expertMemoryMode: ExpertMemoryMode = workerHealthSnapshot.expertMemoryMode {
            expertMemoryModeWireValue = .string(expertMemoryMode.wireName);
        } else {
            expertMemoryModeWireValue = .null;
        }
        statusObject.appendEntry(key: "expert_memory_mode", value: expertMemoryModeWireValue);
        statusObject.appendEntry(
            key: "expert_residency",
            value: workerHealthSnapshot.expertResidency?.wireValue() ?? .null);
        statusObject.appendEntry(
            key: "mlx_memory_snapshot",
            value: workerHealthSnapshot.latestMlxMemorySnapshot?.wireValue() ?? .null);
        statusObject.appendEntry(key: "mlx_memory_ceiling_bytes", value: .unsignedInteger(workerHealthSnapshot.effectiveMlxMemoryCeilingBytes));
        statusObject.appendEntry(key: "machine_mlx_memory_ceiling_bytes", value: .unsignedInteger(workerHealthSnapshot.machineMlxMemoryCeilingBytes));
        statusObject.appendEntry(key: "minimum_mlx_memory_ceiling_bytes", value: .unsignedInteger(workerHealthSnapshot.minimumMlxMemoryCeilingBytes));
        statusObject.appendEntry(
            key: "pending_mlx_memory_ceiling_bytes",
            value: RestStatusResponse.optionalBytesWireValue(workerHealthSnapshot.pendingMlxMemoryCeilingBytes));
        statusObject.appendEntry(
            key: "mlx_memory_limit_error",
            value: RestStatusResponse.optionalTextWireValue(workerHealthSnapshot.mlxMemoryLimitError));
        statusObject.appendEntry(
            key: "configured_maximum_mlx_memory_gb",
            value: RestStatusResponse.configuredMaximumMlxMemoryGbWireValue(configuredRuntimeConfig?.maximumMlxMemoryBytes));
        return try RestHttpResponse.json(statusCode: 200, wireValue: .object(statusObject));
    }

    /// Serializes the active request's progress observation, mirroring the
    /// Rust status endpoint: prefill and preparation report live elapsed
    /// time that keeps advancing between worker frames; generation collapses
    /// into the shared processed/total tokens shape.
    private static func progressSection(
        _ activeRequestProgress: ActiveRequestProgress
    ) -> JsonWireValue {
        var progressObject: JsonWireObject = JsonWireObject(entries: Array<(key: String, value: JsonWireValue)>());
        switch (activeRequestProgress) {
        case let .prefill(
            promptProcessingPhase,
            processedTokens,
            totalTokens,
            requestStartedAt,
            elapsedMillis,
            completedPrefillChunkTokens):
            let liveElapsedMillis: UInt64 = max(
                elapsedMillis,
                UInt64((Date().timeIntervalSince(requestStartedAt) * 1000).rounded()));
            progressObject.appendEntry(key: "phase", value: .string(promptProcessingPhase.wireName));
            progressObject.appendEntry(key: "processed_tokens", value: .unsignedInteger(UInt64(processedTokens)));
            progressObject.appendEntry(key: "total_tokens", value: .unsignedInteger(UInt64(totalTokens)));
            progressObject.appendEntry(key: "elapsed_ms", value: .unsignedInteger(liveElapsedMillis));
            if let completedPrefillChunkTokens = completedPrefillChunkTokens {
                progressObject.appendEntry(
                    key: "completed_prefill_chunk_tokens",
                    value: .unsignedInteger(UInt64(completedPrefillChunkTokens)));
            }
        case let .generationPreparation(
            requestStartedAt,
            preparationStartedAt,
            totalLayerCount,
            residentExpertCount,
            residentExpertPayloadBytes):
            progressObject.appendEntry(key: "phase", value: .string("generation_preparation"));
            progressObject.appendEntry(key: "processed_tokens", value: .unsignedInteger(0));
            progressObject.appendEntry(key: "total_tokens", value: .unsignedInteger(1));
            progressObject.appendEntry(
                key: "elapsed_ms",
                value: .unsignedInteger(UInt64((Date().timeIntervalSince(preparationStartedAt) * 1000).rounded())));
            progressObject.appendEntry(
                key: "request_elapsed_ms",
                value: .unsignedInteger(UInt64((Date().timeIntervalSince(requestStartedAt) * 1000).rounded())));
            progressObject.appendEntry(key: "total_layer_count", value: .unsignedInteger(UInt64(totalLayerCount)));
            progressObject.appendEntry(key: "resident_expert_count", value: .unsignedInteger(UInt64(residentExpertCount)));
            progressObject.appendEntry(key: "resident_expert_payload_bytes", value: .unsignedInteger(residentExpertPayloadBytes));
        case let .generation(
            generatedTokenCount,
            maximumOutputTokens,
            elapsedMillis):
            progressObject.appendEntry(key: "phase", value: .string("generation"));
            progressObject.appendEntry(key: "processed_tokens", value: .unsignedInteger(UInt64(generatedTokenCount)));
            progressObject.appendEntry(key: "total_tokens", value: .unsignedInteger(UInt64(maximumOutputTokens)));
            progressObject.appendEntry(key: "elapsed_ms", value: .unsignedInteger(elapsedMillis));
        case let .imageGeneration(
            phase,
            completedSteps,
            totalSteps,
            elapsedMillis):
            progressObject.appendEntry(key: "phase", value: .string(phase.wireName));
            progressObject.appendEntry(key: "completed_steps", value: .unsignedInteger(UInt64(completedSteps)));
            progressObject.appendEntry(key: "total_steps", value: .unsignedInteger(UInt64(totalSteps)));
            progressObject.appendEntry(key: "elapsed_ms", value: .unsignedInteger(elapsedMillis));
        }
        return .object(progressObject);
    }

    private static func applicationSection(
        instancePaths: AstronomicalInstancePaths,
        buildIdentity: ApplicationBuildIdentity
    ) -> JsonWireValue {
        // The Rust endpoint defaults to Development when the instance was not
        // resolved, keeping custom test state readable in status answers.
        let runtimeInstance: AstronomicalRuntimeInstance = instancePaths.runtimeInstance ?? AstronomicalRuntimeInstance.development;
        var applicationObject: JsonWireObject = JsonWireObject(entries: Array<(key: String, value: JsonWireValue)>());
        applicationObject.appendEntry(key: "version", value: .string(buildIdentity.version));
        applicationObject.appendEntry(key: "build_number", value: .unsignedInteger(buildIdentity.buildNumber));
        applicationObject.appendEntry(key: "commit", value: .string(buildIdentity.commit));
        applicationObject.appendEntry(key: "is_dirty", value: .boolean(buildIdentity.isDirty));
        applicationObject.appendEntry(key: "channel", value: .string(runtimeInstance.rawInstanceName));
        applicationObject.appendEntry(key: "channel_display_name", value: .string(runtimeInstance.displayName));
        applicationObject.appendEntry(key: "state_directory", value: .string(RestStatusResponse.stateDirectoryLabel(instancePaths, runtimeInstance)));
        return .object(applicationObject);
    }

    private static func stateDirectoryLabel(
        _ instancePaths: AstronomicalInstancePaths,
        _ runtimeInstance: AstronomicalRuntimeInstance
    ) -> String {
        switch (instancePaths.isStandardStateDirectory, runtimeInstance) {
        case (true, AstronomicalRuntimeInstance.stable):
            return "~/.astronomical";
        case (true, AstronomicalRuntimeInstance.development):
            return "~/.astronomical-dev";
        case (false, _):
            return "custom";
        }
    }

    private static func readyModelSizeWireValue(
        readyModelId: String,
        configuredRuntimeConfig: ResolvedRuntimeConfig?,
        resolvedRuntimeConfig: ResolvedRuntimeConfig
    ) -> JsonWireValue {
        // The live resolved snapshot wins; the configured snapshot answers
        // when the reload path has not published the model yet.
        let discoveredModelSizes: Array<UInt64?> = [
            resolvedRuntimeConfig.discoveredModels.first { (discoveredModel: DiscoveryDiscoveredModel) -> Bool in
                return discoveredModel.modelId == readyModelId;
            }?.modelSizeBytes,
            configuredRuntimeConfig?.discoveredModels.first { (discoveredModel: DiscoveryDiscoveredModel) -> Bool in
                return discoveredModel.modelId == readyModelId;
            }?.modelSizeBytes,
        ];
        for discoveredModelSize: UInt64? in discoveredModelSizes {
            if let presentModelSize: UInt64 = discoveredModelSize {
                return .unsignedInteger(presentModelSize);
            }
        }
        return .null;
    }

    private static func configuredMaximumMlxMemoryGbWireValue(_ configuredMaximumBytes: UInt64?) -> JsonWireValue {
        guard let configuredMaximumBytes: UInt64 = configuredMaximumBytes else {
            return .null;
        }
        return .unsignedInteger(configuredMaximumBytes / 1_000_000_000);
    }

    private static func optionalBytesWireValue(_ optionalBytes: UInt64?) -> JsonWireValue {
        guard let presentBytes: UInt64 = optionalBytes else {
            return .null;
        }
        return .unsignedInteger(presentBytes);
    }

    private static func optionalTextWireValue(_ optionalText: String?) -> JsonWireValue {
        guard let presentText: String = optionalText else {
            return .null;
        }
        return .string(presentText);
    }
}
