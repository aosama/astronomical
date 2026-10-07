import Foundation

import AstronomicalConfig;
import IpcProtocol;

/// Failures of model resolution and preparation. `modelUnavailable` exits 2
/// (usage: the user asked for something this machine cannot serve); the rest
/// exit 1 as transient daemon problems.
public enum ModelLifecycleError: Error, CustomStringConvertible {

    case daemonNotRunning
    case workerNotReady
    case daemonStoppedResponding
    case modelUnavailable(reason: String)
    case downloadFailed(reason: String)

    public var description: String {
        switch (self) {
        case .daemonNotRunning:
            return "Astronomical isn't running — start it, then retry."
        case .workerNotReady:
            return "The Astronomical worker is not ready yet — retry in a moment."
        case .daemonStoppedResponding:
            return "The daemon stopped responding."
        case let .modelUnavailable(reason):
            return reason
        case let .downloadFailed(reason):
            return "the model download failed: \(reason)"
        }
    }

    static func from(_ probeError: DaemonProbeError) -> ModelLifecycleError {
        switch (probeError) {
        case .daemonNotRunning:
            return .daemonNotRunning
        case .daemonStoppedResponding:
            return .daemonStoppedResponding
        case let .daemonRejected(reason):
            return .downloadFailed(reason: reason)
        }
    }
}

/// The capability the verb needs from its model. Checked before download so
/// the CLI never fetches a model that cannot serve the request.
public enum RequiredCapability: Equatable {

    case chat
    case embeddings

    /// The capability word and the article that reads correctly before it.
    var capabilityText: (article: String, capability: String) {
        switch (self) {
        case .chat:
            return ("a", "chat")
        case .embeddings:
            return ("an", "embeddings")
        }
    }

    func describesInstalledModel(_ listedModel: DaemonListedModel) -> Bool {
        switch (self) {
        case .chat:
            return listedModel.contextWindow != nil;
        case .embeddings:
            return listedModel.supportsEmbeddings;
        }
    }

    func describesCatalogEntry(_ catalogEntry: DaemonCatalogEntry) -> Bool {
        switch (self) {
        case .chat:
            return catalogEntry.contextWindow != nil;
        case .embeddings:
            return catalogEntry.supportsEmbeddings;
        }
    }
}

/**
 * Shared model lifecycle for the one-shot CLI verbs, porting
 * model_lifecycle.rs: resolves which model the user wants (`--model` flag,
 * then the daemon's effective default, then the built-in default), checks
 * the requested capability before touching the network, and when the model
 * is not on this Mac yet starts the daemon download and waits for readiness
 * with live stderr progress.
 */
public struct ModelLifecycle {

    /// Bound for the whole download-wait stage: local disk writes are slow
    /// but finite; generous enough for multi-GB models, short enough that a
    /// wedged download cannot hang the calling script forever.
    public static let downloadWaitStageBoundSeconds: Double = 120;
    /// Wait between download status polls.
    public static let downloadPollIntervalSeconds: Double = 2;

    /// Instance sockets to try, most preferred first.
    public let candidateSocketPaths: Array<String>;
    /// Bound for the whole download-wait stage, in seconds.
    public let downloadStageBoundSeconds: Double;
    /// Wait between download status polls, in seconds.
    public let downloadPollIntervalSeconds: Double;

    public init(
        candidateSocketPaths: Array<String>,
        downloadStageBoundSeconds: Double = ModelLifecycle.downloadWaitStageBoundSeconds,
        downloadPollIntervalSeconds: Double = ModelLifecycle.downloadPollIntervalSeconds
    ) {
        self.candidateSocketPaths = candidateSocketPaths;
        self.downloadStageBoundSeconds = downloadStageBoundSeconds;
        self.downloadPollIntervalSeconds = downloadPollIntervalSeconds;
    }

    /// Worker state, resident model, and effective default model.
    public func statusSnapshot() throws -> DaemonStatusSnapshot {
        do {
            return try DaemonProbe.statusSnapshot(candidateSocketPaths: self.candidateSocketPaths);
        } catch let probeError as DaemonProbeError {
            throw ModelLifecycleError.from(probeError);
        }
    }

    /// Resolves the requested model, verifies the capability, and waits for a
    /// not-yet-present model to finish downloading. Returns the model id to
    /// send on the generation request.
    public func prepareModelId(
        requestedModelId: String?,
        requiredCapability: RequiredCapability,
        progress: (String) -> Void
    ) throws -> String {
        let statusSnapshot: DaemonStatusSnapshot = try self.statusSnapshot();
        if (statusSnapshot.workerStatus == .unavailable) {
            throw ModelLifecycleError.workerNotReady;
        }
        let installedModels: Array<DaemonListedModel>;
        do {
            installedModels = try DaemonProbe.modelsList(candidateSocketPaths: self.candidateSocketPaths);
        } catch let probeError as DaemonProbeError {
            throw ModelLifecycleError.from(probeError);
        }
        let resolvedModelId: String = ModelLifecycle.resolveRequestedModelId(
            requestedModelId: requestedModelId,
            statusSnapshot: statusSnapshot
        );
        let installedAndCapable: Bool = installedModels.contains { (listedModel: DaemonListedModel) -> Bool in
            return listedModel.modelId == resolvedModelId
                && requiredCapability.describesInstalledModel(listedModel);
        };
        if (installedAndCapable) {
            return resolvedModelId;
        }
        // Installed but incapable: say so plainly instead of pretending the
        // model is missing.
        let capabilityText: (article: String, capability: String) = requiredCapability.capabilityText;
        if installedModels.contains(where: { (listedModel: DaemonListedModel) -> Bool in
            return listedModel.modelId == resolvedModelId;
        }) {
            var reason: String = "model \(resolvedModelId) is installed on this Mac but is not "
                + "\(capabilityText.article) \(capabilityText.capability) model — it cannot serve "
                + "this request; run `astronomical models supported` to see what it does";
            if (requestedModelId == nil) {
                reason += "; pass --model <id> to pick one explicitly instead of the default";
            }
            throw ModelLifecycleError.modelUnavailable(reason: reason);
        }
        let catalogEntries: Array<DaemonCatalogEntry>;
        do {
            catalogEntries = try DaemonProbe.catalog(candidateSocketPaths: self.candidateSocketPaths);
        } catch let probeError as DaemonProbeError {
            throw ModelLifecycleError.from(probeError);
        }
        guard let catalogEntry: DaemonCatalogEntry = catalogEntries.first(where: { (candidateEntry: DaemonCatalogEntry) -> Bool in
            return ModelLifecycle.catalogEntryMatches(candidateEntry, requestedModelId: resolvedModelId);
        }) else {
            throw ModelLifecycleError.modelUnavailable(reason: ModelLifecycle.unknownModelReason(
                requestedModelId: resolvedModelId,
                installedModels: installedModels,
                requiredCapability: requiredCapability
            ));
        }
        if !requiredCapability.describesCatalogEntry(catalogEntry) {
            throw ModelLifecycleError.modelUnavailable(reason: "model \(resolvedModelId) is not "
                + "\(capabilityText.article) \(capabilityText.capability) model — it cannot serve "
                + "this request; run `astronomical models supported` to see what it does");
        }
        if !catalogEntry.readyOnThisMac {
            try self.waitForDownload(modelId: resolvedModelId, progress: progress);
        }
        return resolvedModelId;
    }

    /// Downloads a catalog entry regardless of capability — the
    /// `models download` verb — and waits until it is ready.
    public func ensureDownloaded(
        requestedModelId: String,
        progress: (String) -> Void
    ) throws -> String {
        let catalogEntries: Array<DaemonCatalogEntry>;
        do {
            catalogEntries = try DaemonProbe.catalog(candidateSocketPaths: self.candidateSocketPaths);
        } catch let probeError as DaemonProbeError {
            throw ModelLifecycleError.from(probeError);
        }
        guard let matchingEntry: DaemonCatalogEntry = catalogEntries.first(where: { (candidateEntry: DaemonCatalogEntry) -> Bool in
            return ModelLifecycle.catalogEntryMatches(candidateEntry, requestedModelId: requestedModelId);
        }) else {
            throw ModelLifecycleError.modelUnavailable(reason: "model \(requestedModelId) is not in "
                + "the release catalog; run `astronomical models supported` to list downloadable "
                + "models");
        }
        if !matchingEntry.readyOnThisMac {
            try self.waitForDownload(modelId: requestedModelId, progress: progress);
        }
        return requestedModelId;
    }

    /// Flag > daemon effective default > built-in default.
    private static func resolveRequestedModelId(
        requestedModelId: String?,
        statusSnapshot: DaemonStatusSnapshot
    ) -> String {
        if let requestedModelId: String = requestedModelId {
            return requestedModelId;
        }
        if let defaultModelId: String = statusSnapshot.defaultModelId {
            return defaultModelId;
        }
        return DefaultModel.builtinDefaultModelId;
    }

    /// Starts the download (or resumes a matching paused job) and polls the
    /// job state until the catalog says the model is ready, the job reports
    /// a failure, or the stage bound expires.
    private func waitForDownload(
        modelId: String,
        progress: (String) -> Void
    ) throws -> Void {
        do {
            try DaemonProbe.downloadStart(modelId: modelId, candidateSocketPaths: self.candidateSocketPaths);
        } catch let probeError as DaemonProbeError {
            throw ModelLifecycleError.from(probeError);
        }
        let downloadDeadline: Date = Date().addingTimeInterval(self.downloadStageBoundSeconds);
        progress("downloading \(modelId) …");
        while (true) {
            Thread.sleep(forTimeInterval: self.downloadPollIntervalSeconds);
            let activeJob: DaemonDownloadJob?;
            do {
                activeJob = try DaemonProbe.downloadStatus(candidateSocketPaths: self.candidateSocketPaths);
            } catch let probeError as DaemonProbeError {
                throw ModelLifecycleError.from(probeError);
            }
            if let activeJob: DaemonDownloadJob = activeJob {
                if let downloadError: String = activeJob.error {
                    throw ModelLifecycleError.downloadFailed(
                        reason: "\(activeJob.huggingfaceId): \(downloadError)");
                }
                progress(ModelLifecycle.renderDownloadProgress(activeJob));
                if (Date() >= downloadDeadline) {
                    throw ModelLifecycleError.downloadFailed(
                        reason: ModelLifecycle.downloadTimeoutReason(modelId));
                }
                continue;
            }
            // The daemon deletes the job file when a download succeeds, so a
            // vanished job is only believed once the catalog agrees the model
            // is ready; a vanished job with no catalog download state means
            // it never really ran.
            let catalogEntries: Array<DaemonCatalogEntry>;
            do {
                catalogEntries = try DaemonProbe.catalog(candidateSocketPaths: self.candidateSocketPaths);
            } catch let probeError as DaemonProbeError {
                throw ModelLifecycleError.from(probeError);
            }
            let matchingEntry: DaemonCatalogEntry? = catalogEntries.first { (candidateEntry: DaemonCatalogEntry) -> Bool in
                return ModelLifecycle.catalogEntryMatches(candidateEntry, requestedModelId: modelId);
            };
            switch (matchingEntry) {
            case let .some(catalogEntry) where catalogEntry.readyOnThisMac:
                progress("\(modelId) is ready");
                return;
            case let .some(catalogEntry) where catalogEntry.downloadState != nil:
                // Job file was not visible this poll but the catalog still
                // tracks state: keep waiting.
                break;
            default:
                throw ModelLifecycleError.downloadFailed(reason: "the download of \(modelId) did "
                    + "not complete; run `astronomical models download \(modelId)` to retry");
            }
            if (Date() >= downloadDeadline) {
                throw ModelLifecycleError.downloadFailed(
                    reason: ModelLifecycle.downloadTimeoutReason(modelId));
            }
        }
    }

    private static func downloadTimeoutReason(_ modelId: String) -> String {
        return "the download of \(modelId) did not finish in time; run `astronomical status` "
            + "to check on it";
    }

    /// Mirrors the daemon's catalog entry matching: the full huggingface id,
    /// or the leaf derived from it against the request's leaf. The leaf is
    /// derived here rather than taken from the wire's `requestableModelId`,
    /// which stays `nil` until the entry is ready on this Mac.
    static func catalogEntryMatches(
        _ catalogEntry: DaemonCatalogEntry,
        requestedModelId: String
    ) -> Bool {
        return catalogEntry.huggingfaceId == requestedModelId
            || ModelIdentity.leafModelId(modelId: catalogEntry.huggingfaceId)
                == ModelIdentity.leafModelId(modelId: requestedModelId);
    }

    /// Usage-grade rejection for a model neither installed nor downloadable:
    /// near matches from what the machine actually has, plus the pointer to
    /// the full downloadable list.
    static func unknownModelReason(
        requestedModelId: String,
        installedModels: Array<DaemonListedModel>,
        requiredCapability: RequiredCapability
    ) -> String {
        let installedIds: Array<String> = installedModels.map { (listedModel: DaemonListedModel) -> String in
            return listedModel.modelId;
        };
        let nearMatches: Array<String> = ModelIdentity.nearModelMatches(
            requestedModelId: requestedModelId,
            candidateModelIds: installedIds
        );
        let capabilityText: (article: String, capability: String) = requiredCapability.capabilityText;
        var reason: String = "model \(requestedModelId) is not installed on this Mac and is not "
            + "in the release catalog for \(capabilityText.capability) models";
        if !nearMatches.isEmpty {
            reason += " (did you mean: \(nearMatches.joined(separator: ", "))?)";
        }
        reason += "; run `astronomical models supported` to list what can be downloaded";
        return reason;
    }

    /// One live-progress line: the terminal re-uses the line while the
    /// download runs. Sizes render in decimal SI gigabytes.
    static func renderDownloadProgress(_ downloadJob: DaemonDownloadJob) -> String {
        if (downloadJob.bytesTotal == 0) {
            return "\(downloadJob.huggingfaceId): \(downloadJob.state)";
        }
        let completedGigabytes: String = CliFormatting.formatGigabytes(downloadJob.bytesCompleted);
        let totalGigabytes: String = CliFormatting.formatGigabytes(downloadJob.bytesTotal);
        let percentage: Double = 100.0 * Double(downloadJob.bytesCompleted) / Double(downloadJob.bytesTotal);
        return "\(downloadJob.huggingfaceId): \(downloadJob.state) \(completedGigabytes) GB / "
            + "\(totalGigabytes) GB (\(Int(percentage.rounded()))%)";
    }
}
