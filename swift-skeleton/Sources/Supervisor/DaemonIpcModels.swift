import Foundation

import AstronomicalConfig;
import IpcProtocol;

/**
 * Everything the daemon IPC model-lifecycle verbs need: the widest supervisor
 * executor for embeddings and health, the live resolved configuration for
 * discovery, the bundled release catalog, and the Library download
 * coordinator when the daemon owns Library state. Mirrors the model half of
 * the Rust `DaemonIpcGenerationContext`.
 */
public struct DaemonIpcModelsContext: @unchecked Sendable {

    /// Executor backed by the single resident local worker; carries the
    /// embeddings start method and worker health.
    public let embeddingsExecutor: any EmbeddingsExecuting;
    /// Live model discovery, read fresh per request; `nil` when config
    /// reloading is not wired.
    public let liveResolvedRuntimeConfigProvider: @Sendable () -> ResolvedRuntimeConfig?;
    /// Bundled release catalog of downloadable models.
    public let downloadCatalog: DownloadCatalog;
    /// Library download coordinator, present when the daemon owns Library state.
    public let libraryDownloadCoordinator: LibraryDownloadCoordinator?;
    /// Instance paths used to read and persist user configuration.
    public let instancePaths: AstronomicalInstancePaths;

    public init(
        embeddingsExecutor: any EmbeddingsExecuting,
        liveResolvedRuntimeConfigProvider: @escaping @Sendable () -> ResolvedRuntimeConfig?,
        downloadCatalog: DownloadCatalog,
        libraryDownloadCoordinator: LibraryDownloadCoordinator?,
        instancePaths: AstronomicalInstancePaths
    ) {
        self.embeddingsExecutor = embeddingsExecutor;
        self.liveResolvedRuntimeConfigProvider = liveResolvedRuntimeConfigProvider;
        self.downloadCatalog = downloadCatalog;
        self.libraryDownloadCoordinator = libraryDownloadCoordinator;
        self.instancePaths = instancePaths;
    }

    /// Discovered models from the live config, empty when no config is wired.
    func liveDiscoveredModels() -> Array<DiscoveryDiscoveredModel> {
        return self.liveResolvedRuntimeConfigProvider()?.discoveredModels ?? [];
    }

    /// Performance-attribution preference of the live configuration.
    func performanceAttributionEnabled() -> Bool {
        return self.liveResolvedRuntimeConfigProvider()?.performanceAttributionEnabled ?? false;
    }
}

/**
 * Daemon IPC handlers for the model-lifecycle CLI verbs, porting
 * apps/supervisor/src/daemon_ipc_models.rs: installed model listing, catalog
 * projection, download control, and default-model setting.
 */
enum DaemonIpcModels {

    /// Lists the models discovered on this machine with their resident marker.
    static func handleModelsList(
        modelsContext: DaemonIpcModelsContext,
        streamingResponseWriter: StreamingResponseWriter
    ) throws -> Void {
        let attributionStart: ContinuousClock.Instant? = DaemonIpcPerformanceAttribution.startedOperation(
            operationName: "daemon_ipc_models_list",
            performanceAttributionEnabled: modelsContext.performanceAttributionEnabled());
        defer {
            DaemonIpcPerformanceAttribution.finishedOperation(
                operationName: "daemon_ipc_models_list",
                operationStart: attributionStart,
                operationOutcome: "success",
                performanceAttributionEnabled: modelsContext.performanceAttributionEnabled());
        }
        let healthSnapshot: WorkerHealthSnapshot = modelsContext.embeddingsExecutor.workerHealthSnapshot();
        let listedModels: Array<DaemonListedModel> = DaemonIpcModels.buildListedModels(
            healthSnapshot: healthSnapshot,
            discoveredModels: modelsContext.liveDiscoveredModels()
        );
        try DaemonIpcModels.sendTerminalResponse(
            streamingResponseWriter,
            .modelsList(models: listedModels)
        );
    }

    private static func buildListedModels(
        healthSnapshot: WorkerHealthSnapshot,
        discoveredModels: Array<DiscoveryDiscoveredModel>
    ) -> Array<DaemonListedModel> {
        var listedModels: Array<DaemonListedModel> = [];
        listedModels.reserveCapacity(discoveredModels.count);
        for discoveredModel: DiscoveryDiscoveredModel in discoveredModels {
            let contextWindow: UInt32?;
            if case let .chat(chatCapabilities) = discoveredModel.capabilities {
                contextWindow = chatCapabilities.contextWindowTokens;
            } else {
                contextWindow = nil;
            }
            let supportsEmbeddings: Bool;
            if case .embeddings = discoveredModel.capabilities {
                supportsEmbeddings = true;
            } else {
                supportsEmbeddings = false;
            }
            listedModels.append(DaemonListedModel(
                modelId: discoveredModel.modelId,
                family: discoveredModel.modelFamily.rawValue,
                contextWindow: contextWindow,
                supportsEmbeddings: supportsEmbeddings,
                isResident: healthSnapshot.readyModelId == discoveredModel.modelId,
                sizeBytes: discoveredModel.modelSizeBytes
            ));
        }
        return listedModels;
    }

    /// Lists the release download catalog with local readiness per entry.
    static func handleCatalog(
        modelsContext: DaemonIpcModelsContext,
        streamingResponseWriter: StreamingResponseWriter
    ) throws -> Void {
        let attributionStart: ContinuousClock.Instant? = DaemonIpcPerformanceAttribution.startedOperation(
            operationName: "daemon_ipc_catalog",
            performanceAttributionEnabled: modelsContext.performanceAttributionEnabled());
        defer {
            DaemonIpcPerformanceAttribution.finishedOperation(
                operationName: "daemon_ipc_catalog",
                operationStart: attributionStart,
                operationOutcome: "success",
                performanceAttributionEnabled: modelsContext.performanceAttributionEnabled());
        }
        let currentJob: LibraryDownloadJobRecord? = try DaemonIpcModels.activeDownloadJob(modelsContext: modelsContext);
        let validatedPublications: Set<String> = DaemonIpcModels.activeValidatedPublications(
            modelsContext: modelsContext
        );
        let currentJobSummary: LibraryDownloadJobSummary? = currentJob.map { (jobRecord: LibraryDownloadJobRecord) -> LibraryDownloadJobSummary in
            return LibraryDownloadJobSummary(
                huggingfaceId: jobRecord.huggingfaceId,
                stateName: jobRecord.state.rawValue
            );
        };
        let projections: Array<LibraryCatalogEntryProjection> = LibraryCatalogProjection.projectCatalogEntries(
            downloadCatalog: modelsContext.downloadCatalog,
            discoveredModels: modelsContext.liveDiscoveredModels(),
            validatedPublications: validatedPublications,
            currentJob: currentJobSummary
        );
        let entries: Array<DaemonCatalogEntry> = projections.map { (projection: LibraryCatalogEntryProjection) -> DaemonCatalogEntry in
            let catalogEntry: DownloadCatalogEntry = projection.catalogEntry;
            let capabilities: DownloadCatalogCapabilities = projection.capabilities;
            return DaemonCatalogEntry(
                huggingfaceId: catalogEntry.huggingfaceId,
                displayName: catalogEntry.displayName,
                family: catalogEntry.family.rawValue,
                approximateSizeBytes: catalogEntry.approximateSizeBytes,
                readyOnThisMac: projection.readyOnThisMac,
                requestableModelId: projection.requestableModelId,
                downloadState: projection.downloadState,
                contextWindow: capabilities.contextWindow,
                supportsReasoning: capabilities.supportsReasoning,
                supportsVision: capabilities.supportsVision,
                supportsToolCalls: capabilities.supportsToolCalls,
                supportsImageGeneration: capabilities.supportsImageGeneration,
                supportsEmbeddings: capabilities.supportsEmbeddings
            );
        };
        try DaemonIpcModels.sendTerminalResponse(streamingResponseWriter, .catalog(entries: entries));
    }

    /// Starts a download for the requested model, or resumes a paused job for
    /// the same catalog entry. Rejects when a different download is active.
    static func handleDownloadStart(
        modelId requestedModelId: String,
        modelsContext: DaemonIpcModelsContext,
        streamingResponseWriter: StreamingResponseWriter
    ) throws -> Void {
        guard let downloadCoordinator: LibraryDownloadCoordinator = modelsContext.libraryDownloadCoordinator else {
            return try DaemonIpcModels.sendTerminalResponse(
                streamingResponseWriter,
                .requestRejected(reason: "the daemon has no Library download coordinator wired")
            );
        }
        guard let catalogEntry: DownloadCatalogEntry = DaemonIpcModels.resolveCatalogEntry(
            downloadCatalog: modelsContext.downloadCatalog,
            requestedModelId: requestedModelId
        ) else {
            let suggestedModelIds: Array<String> = ModelIdentity.nearModelMatches(
                requestedModelId: requestedModelId,
                candidateModelIds: DaemonIpcModels.catalogModelIdCandidates(
                    downloadCatalog: modelsContext.downloadCatalog
                )
            );
            return try DaemonIpcModels.sendTerminalResponse(
                streamingResponseWriter,
                .requestRejected(reason: DaemonIpcChat.unknownModelRejectionReason(
                    requestedModelId: requestedModelId,
                    suggestedModelIds: suggestedModelIds,
                    baseMessage: "the model is not in the release catalog; run "
                        + "`astronomical models supported` to list downloadable models"
                ))
            );
        }
        let huggingfaceId: String = catalogEntry.huggingfaceId;
        let attributionStart: ContinuousClock.Instant? = DaemonIpcPerformanceAttribution.startedOperation(
            operationName: "daemon_ipc_download_start",
            performanceAttributionEnabled: modelsContext.performanceAttributionEnabled());
        var attributionOutcomeText: String = "success";
        defer {
            DaemonIpcPerformanceAttribution.finishedOperation(
                operationName: "daemon_ipc_download_start",
                operationStart: attributionStart,
                operationOutcome: attributionOutcomeText,
                performanceAttributionEnabled: modelsContext.performanceAttributionEnabled());
        }
        do {
            try DaemonIpcModels.startOrResumeDownload(
                downloadCoordinator: downloadCoordinator,
                huggingfaceId: huggingfaceId
            );
        } catch let rejectionError {
            attributionOutcomeText = "failure";
            return try DaemonIpcModels.sendTerminalResponse(
                streamingResponseWriter,
                .requestRejected(reason: String(describing: rejectionError))
            );
        }
        try DaemonIpcModels.sendTerminalResponse(
            streamingResponseWriter,
            .downloadStarted(huggingfaceId: huggingfaceId)
        );
    }

    /// Resolves a requestable model id or a full huggingface id to a catalog entry.
    private static func resolveCatalogEntry(
        downloadCatalog: DownloadCatalog,
        requestedModelId: String
    ) -> DownloadCatalogEntry? {
        let requestedLeafModelId: String = ModelIdentity.leafModelId(modelId: requestedModelId);
        return downloadCatalog.entries.first { (catalogEntry: DownloadCatalogEntry) -> Bool in
            return catalogEntry.huggingfaceId == requestedModelId
                || ModelIdentity.leafModelId(modelId: catalogEntry.huggingfaceId) == requestedLeafModelId;
        };
    }

    private static func catalogModelIdCandidates(downloadCatalog: DownloadCatalog) -> Array<String> {
        return downloadCatalog.entries.map { (catalogEntry: DownloadCatalogEntry) -> String in
            return LibraryCatalogProjection.requestableModelIdFromHuggingfaceId(
                catalogEntry.huggingfaceId
            );
        };
    }

    private static func startOrResumeDownload(
        downloadCoordinator: LibraryDownloadCoordinator,
        huggingfaceId: String
    ) throws -> Void {
        if let activeJob: LibraryDownloadJobRecord = try DaemonIpcModels.bridgedValue(
            operation: { return await downloadCoordinator.currentJob(); }
        ) {
            if (activeJob.huggingfaceId == huggingfaceId) {
                do {
                    try DaemonIpcModels.bridgedValue(
                        operation: { try await downloadCoordinator.resume(); }
                    );
                } catch let resumeError {
                    throw DaemonIpcDownloadStartError.resumeFailure(
                        description: String(describing: resumeError)
                    );
                }
                return;
            }
            throw DaemonIpcDownloadStartError.anotherDownloadActive(
                activeHuggingfaceId: activeJob.huggingfaceId,
                requestedHuggingfaceId: huggingfaceId
            );
        }
        do {
            try DaemonIpcModels.bridgedValue(
                operation: { try await downloadCoordinator.start(huggingfaceId: huggingfaceId); }
            );
        } catch let startError {
            throw DaemonIpcDownloadStartError.startFailure(
                description: String(describing: startError)
            );
        }
    }

    /// Reports the active library download job, if any.
    static func handleDownloadStatus(
        modelsContext: DaemonIpcModelsContext,
        streamingResponseWriter: StreamingResponseWriter
    ) throws -> Void {
        let attributionStart: ContinuousClock.Instant? = DaemonIpcPerformanceAttribution.startedOperation(
            operationName: "daemon_ipc_download_status",
            performanceAttributionEnabled: modelsContext.performanceAttributionEnabled());
        defer {
            DaemonIpcPerformanceAttribution.finishedOperation(
                operationName: "daemon_ipc_download_status",
                operationStart: attributionStart,
                operationOutcome: "success",
                performanceAttributionEnabled: modelsContext.performanceAttributionEnabled());
        }
        let downloadJob: DaemonDownloadJob? = try DaemonIpcModels.activeDownloadJob(
            modelsContext: modelsContext
        ).map { (jobRecord: LibraryDownloadJobRecord) -> DaemonDownloadJob in
            return DaemonIpcModels.downloadJobToWire(jobRecord);
        };
        try DaemonIpcModels.sendTerminalResponse(
            streamingResponseWriter,
            .downloadStatus(job: downloadJob)
        );
    }

    private static func downloadJobToWire(_ downloadJob: LibraryDownloadJobRecord) -> DaemonDownloadJob {
        return DaemonDownloadJob(
            huggingfaceId: downloadJob.huggingfaceId,
            state: downloadJob.state.rawValue,
            bytesCompleted: downloadJob.bytesCompleted,
            bytesTotal: downloadJob.bytesTotal,
            error: downloadJob.errorCode
        );
    }

    /// Persists the default model after validating it against the catalog or
    /// the discovered models, so manual installs can also become the default.
    static func handleDefaultModelSet(
        modelId requestedModelId: String,
        modelsContext: DaemonIpcModelsContext,
        streamingResponseWriter: StreamingResponseWriter
    ) throws -> Void {
        let attributionStart: ContinuousClock.Instant? = DaemonIpcPerformanceAttribution.startedOperation(
            operationName: "daemon_ipc_default_model_set",
            performanceAttributionEnabled: modelsContext.performanceAttributionEnabled());
        var attributionOutcomeText: String = "success";
        defer {
            DaemonIpcPerformanceAttribution.finishedOperation(
                operationName: "daemon_ipc_default_model_set",
                operationStart: attributionStart,
                operationOutcome: attributionOutcomeText,
                performanceAttributionEnabled: modelsContext.performanceAttributionEnabled());
        }
        let normalizedModelId: String? = DaemonIpcModels.normalizeDefaultModelId(
            downloadCatalog: modelsContext.downloadCatalog,
            discoveredModels: modelsContext.liveDiscoveredModels(),
            requestedModelId: requestedModelId
        );
        guard let normalizedDefaultModelId: String = normalizedModelId else {
            attributionOutcomeText = "failure";
            let candidates: Array<String> = DaemonIpcModels.catalogModelIdCandidates(
                downloadCatalog: modelsContext.downloadCatalog
            );
            let suggestedModelIds: Array<String> = ModelIdentity.nearModelMatches(
                requestedModelId: requestedModelId,
                candidateModelIds: candidates
            );
            return try DaemonIpcModels.sendTerminalResponse(
                streamingResponseWriter,
                .requestRejected(reason: DaemonIpcChat.unknownModelRejectionReason(
                    requestedModelId: requestedModelId,
                    suggestedModelIds: suggestedModelIds,
                    baseMessage: "the model is neither in the release catalog nor installed on "
                        + "this machine; set a model from `astronomical models supported` or "
                        + "`astronomical models list`"
                ))
            );
        }
        do {
            _ = try DefaultModel.writeDefaultModel(
                stateDirectory: modelsContext.instancePaths.stateDirectory,
                defaultModel: normalizedDefaultModelId
            );
        } catch let persistError {
            attributionOutcomeText = "failure";
            return try DaemonIpcModels.sendTerminalResponse(
                streamingResponseWriter,
                .requestRejected(reason: String(describing: persistError))
            );
        }
        try DaemonIpcModels.sendTerminalResponse(
            streamingResponseWriter,
            .defaultModelSet(defaultModelId: normalizedDefaultModelId)
        );
    }

    /// Maps the requested id to the canonical requestable id: the catalog
    /// entry's requestable id when the id names a catalog entry, the
    /// discovered model's id when it names a discovered model, otherwise `nil`.
    private static func normalizeDefaultModelId(
        downloadCatalog: DownloadCatalog,
        discoveredModels: Array<DiscoveryDiscoveredModel>,
        requestedModelId: String
    ) -> String? {
        if let catalogEntry: DownloadCatalogEntry = DaemonIpcModels.resolveCatalogEntry(
            downloadCatalog: downloadCatalog,
            requestedModelId: requestedModelId
        ) {
            return LibraryCatalogProjection.requestableModelIdFromHuggingfaceId(
                catalogEntry.huggingfaceId
            );
        }
        let knownModelIds: Array<String> = discoveredModels.map { (discoveredModel: DiscoveryDiscoveredModel) -> String in
            return discoveredModel.modelId;
        };
        let resolvedModelId: String = ModelIdentity.resolveModelId(
            requestedModelId: requestedModelId,
            knownModelIds: knownModelIds
        );
        return knownModelIds.contains(resolvedModelId) ? resolvedModelId : nil;
    }

    private static func sendTerminalResponse(
        _ streamingResponseWriter: StreamingResponseWriter,
        _ daemonResponse: DaemonResponse
    ) throws -> Void {
        try streamingResponseWriter.sendResponse(daemonResponse);
        try streamingResponseWriter.close();
    }

    /// The active download job, or `nil` when no coordinator is wired or no
    /// job is in flight.
    private static func activeDownloadJob(
        modelsContext: DaemonIpcModelsContext
    ) throws -> LibraryDownloadJobRecord? {
        guard let downloadCoordinator: LibraryDownloadCoordinator = modelsContext.libraryDownloadCoordinator else {
            return nil;
        }
        return try DaemonIpcModels.bridgedValue(
            operation: { return await downloadCoordinator.currentJob(); }
        );
    }

    private static func activeValidatedPublications(
        modelsContext: DaemonIpcModelsContext
    ) -> Set<String> {
        guard let downloadCoordinator: LibraryDownloadCoordinator = modelsContext.libraryDownloadCoordinator else {
            return [];
        }
        return downloadCoordinator.validatedPublications.snapshot();
    }

    /**
     * Bridges one actor-isolated operation onto the serving thread: the
     * per-connection daemon IPC threads are synchronous, so the operation
     * runs on the concurrent executor and the caller waits on a semaphore.
     */
    private static func bridgedValue<Outcome>(
        operation: @escaping @Sendable () async throws -> Outcome
    ) throws -> Outcome {
        let outcomeBox: DaemonIpcOutcomeBox<Outcome> = DaemonIpcOutcomeBox<Outcome>();
        let completionSemaphore: DispatchSemaphore = DispatchSemaphore(value: 0);
        Task<Void, Never> {
            do {
                outcomeBox.record(.success(try await operation()));
            } catch {
                outcomeBox.record(.failure(error));
            }
            completionSemaphore.signal();
        }
        completionSemaphore.wait();
        guard let outcome: Result<Outcome, Error> = outcomeBox.take() else {
            throw DaemonIpcModelsBridgeError.outcomeLost;
        }
        return try outcome.get();
    }
}

/// One semaphore-handed outcome box for the sync-to-actor bridge.
private final class DaemonIpcOutcomeBox<Outcome>: @unchecked Sendable {

    private let stateLock: NSLock = NSLock();
    private var outcome: Result<Outcome, Error>?;

    func record(_ outcome: Result<Outcome, Error>) {
        self.stateLock.lock();
        self.outcome = outcome;
        self.stateLock.unlock();
    }

    func take() -> Result<Outcome, Error>? {
        self.stateLock.lock();
        let currentOutcome: Result<Outcome, Error>? = self.outcome;
        self.stateLock.unlock();
        return currentOutcome;
    }
}

/// One-line rejection reasons for `download_start`, mirroring the Rust texts.
enum DaemonIpcDownloadStartError: Error, CustomStringConvertible {

    case anotherDownloadActive(activeHuggingfaceId: String, requestedHuggingfaceId: String);
    case resumeFailure(description: String);
    case startFailure(description: String);

    var description: String {
        switch (self) {
        case let .anotherDownloadActive(activeHuggingfaceId, requestedHuggingfaceId):
            return "a download of \(activeHuggingfaceId) is already active; wait for it to "
                + "finish or cancel it before downloading \(requestedHuggingfaceId)";
        case let .resumeFailure(description):
            return "the paused download could not be resumed: \(description)";
        case let .startFailure(description):
            return "the download could not be started: \(description)";
        }
    }
}

enum DaemonIpcModelsBridgeError: Error {

    case outcomeLost;
}
