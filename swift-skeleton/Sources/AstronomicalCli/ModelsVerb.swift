import Foundation

import IpcProtocol;

/**
 * Collaborators the models verb needs, injected so tests can stub them.
 */
public struct ModelsDependencies {

    /// Instance sockets to try, most preferred first.
    public let candidateSocketPaths: Array<String>;
    /// Where the report data goes.
    public let stdout: TextOutputWriter;
    /// Where download progress goes.
    public let stderr: TextOutputWriter;
    /// Bound for the whole download-wait stage.
    public let downloadStageBoundSeconds: Double;
    /// Wait between download status polls.
    public let downloadPollIntervalSeconds: Double;

    public init(
        candidateSocketPaths: Array<String>,
        stdout: TextOutputWriter,
        stderr: TextOutputWriter,
        downloadStageBoundSeconds: Double = ModelLifecycle.downloadWaitStageBoundSeconds,
        downloadPollIntervalSeconds: Double = ModelLifecycle.downloadPollIntervalSeconds
    ) {
        self.candidateSocketPaths = candidateSocketPaths;
        self.stdout = stdout;
        self.stderr = stderr;
        self.downloadStageBoundSeconds = downloadStageBoundSeconds;
        self.downloadPollIntervalSeconds = downloadPollIntervalSeconds;
    }
}

/**
 * The `astronomical models` verb, porting models_command.rs: list installed
 * models, show the release catalog, show or persist the effective default
 * model, and drive a download with live progress. Report data goes to
 * stdout; download progress goes to stderr.
 */
public enum ModelsVerb {

    /// Runs one models subcommand against the resident daemon.
    public static func run(
        modelsCommand: ModelsSubcommand,
        modelsDependencies: ModelsDependencies
    ) throws -> Void {
        switch (modelsCommand) {
        case .list:
            let installedModels: Array<DaemonListedModel>;
            do {
                installedModels = try DaemonProbe.modelsList(
                    candidateSocketPaths: modelsDependencies.candidateSocketPaths);
            } catch let probeError as DaemonProbeError {
                throw ModelsVerbError.from(probeError);
            }
            try ModelsVerb.writeInstalledModels(
                modelsDependencies.stdout,
                installedModels: installedModels
            );
        case .supported:
            let catalogEntries: Array<DaemonCatalogEntry>;
            do {
                catalogEntries = try DaemonProbe.catalog(
                    candidateSocketPaths: modelsDependencies.candidateSocketPaths);
            } catch let probeError as DaemonProbeError {
                throw ModelsVerbError.from(probeError);
            }
            try ModelsVerb.writeCatalog(
                modelsDependencies.stdout,
                catalogEntries: catalogEntries
            );
        case let .default(modelId):
            try ModelsVerb.runDefaultSubcommand(
                modelId: modelId,
                modelsDependencies: modelsDependencies
            );
        case let .download(modelId):
            try ModelsVerb.runDownloadSubcommand(
                modelId: modelId,
                modelsDependencies: modelsDependencies
            );
        }
    }

    private static func runDefaultSubcommand(
        modelId: String?,
        modelsDependencies: ModelsDependencies
    ) throws -> Void {
        let modelLifecycle: ModelLifecycle = ModelLifecycle(
            candidateSocketPaths: modelsDependencies.candidateSocketPaths,
            downloadStageBoundSeconds: modelsDependencies.downloadStageBoundSeconds,
            downloadPollIntervalSeconds: modelsDependencies.downloadPollIntervalSeconds
        );
        if let requestedModelId: String = modelId {
            // Persist the choice only once the model can actually serve: a
            // default pointing at a model this Mac lacks is fetched first,
            // and an id outside the catalog is rejected here.
            let resolvedModelId: String;
            do {
                resolvedModelId = try modelLifecycle.ensureDownloaded(
                    requestedModelId: requestedModelId,
                    progress: { (progressLine: String) -> Void in
                        _ = modelsDependencies.stderr.write("\r\(progressLine)");
                    }
                );
            } catch let lifecycleError as ModelLifecycleError {
                throw ModelsVerbError.from(lifecycleError);
            }
            let persistedModelId: String;
            do {
                persistedModelId = try DaemonProbe.defaultModelSet(
                    modelId: resolvedModelId,
                    candidateSocketPaths: modelsDependencies.candidateSocketPaths);
            } catch let probeError as DaemonProbeError {
                throw ModelsVerbError.from(probeError);
            }
            if !modelsDependencies.stdout.write("default model: \(persistedModelId)\n") {
                throw ModelsVerbError.stdoutUnwritable(cause: "standard output is closed");
            }
            return;
        }
        let statusSnapshot: DaemonStatusSnapshot;
        do {
            statusSnapshot = try DaemonProbe.statusSnapshot(
                candidateSocketPaths: modelsDependencies.candidateSocketPaths);
        } catch let probeError as DaemonProbeError {
            throw ModelsVerbError.from(probeError);
        }
        let effectiveDefaultModelId: String = statusSnapshot.defaultModelId ?? "none";
        if !modelsDependencies.stdout.write("default model: \(effectiveDefaultModelId)\n") {
            throw ModelsVerbError.stdoutUnwritable(cause: "standard output is closed");
        }
    }

    private static func runDownloadSubcommand(
        modelId: String,
        modelsDependencies: ModelsDependencies
    ) throws -> Void {
        let modelLifecycle: ModelLifecycle = ModelLifecycle(
            candidateSocketPaths: modelsDependencies.candidateSocketPaths,
            downloadStageBoundSeconds: modelsDependencies.downloadStageBoundSeconds,
            downloadPollIntervalSeconds: modelsDependencies.downloadPollIntervalSeconds
        );
        let resolvedModelId: String;
        do {
            resolvedModelId = try modelLifecycle.ensureDownloaded(
                requestedModelId: modelId,
                progress: { (progressLine: String) -> Void in
                    _ = modelsDependencies.stderr.write("\r\(progressLine)");
                }
            );
        } catch let lifecycleError as ModelLifecycleError {
            throw ModelsVerbError.from(lifecycleError);
        }
        _ = modelsDependencies.stderr.write("\n\(resolvedModelId) is available\n");
    }

    /// One aligned line per installed model; `*` marks the resident one.
    static func writeInstalledModels(
        _ stdout: TextOutputWriter,
        installedModels: Array<DaemonListedModel>
    ) throws -> Void {
        let modelWidth: Int = installedModels.map { (listedModel: DaemonListedModel) -> Int in
            return listedModel.modelId.count;
        }.max() ?? "MODEL".count;
        var report: String = "";
        report += ModelsVerb.padded("", width: 8) + " "
            + ModelsVerb.padded("MODEL", width: modelWidth) + " "
            + ModelsVerb.padded("FAMILY", width: 10) + " "
            + ModelsVerb.padded("CONTEXT", width: 10) + " "
            + "SIZE\n";
        for listedModel: DaemonListedModel in installedModels {
            let marker: String = listedModel.isResident ? "*" : "";
            let contextText: String = listedModel.contextWindow.map { (contextWindow: UInt32) -> String in
                return String(contextWindow);
            } ?? "";
            report += ModelsVerb.padded(marker, width: 8) + " "
                + ModelsVerb.padded(listedModel.modelId, width: modelWidth) + " "
                + ModelsVerb.padded(listedModel.family, width: 10) + " "
                + ModelsVerb.padded(contextText, width: 10) + " "
                + CliFormatting.formatGigabytes(listedModel.sizeBytes) + "\n";
        }
        if !stdout.write(report) {
            throw ModelsVerbError.stdoutUnwritable(cause: "standard output is closed");
        }
    }

    /// One aligned line per catalog entry; STATE is `ready`, the active
    /// download state, or `not on this Mac`.
    static func writeCatalog(
        _ stdout: TextOutputWriter,
        catalogEntries: Array<DaemonCatalogEntry>
    ) throws -> Void {
        let modelWidth: Int = catalogEntries.map { (catalogEntry: DaemonCatalogEntry) -> Int in
            let entryModelId: String = catalogEntry.requestableModelId ?? catalogEntry.huggingfaceId;
            return entryModelId.count;
        }.max() ?? "MODEL".count;
        var report: String = "";
        report += ModelsVerb.padded("MODEL", width: modelWidth) + " "
            + ModelsVerb.padded("FAMILY", width: 10) + " "
            + ModelsVerb.padded("SIZE", width: 8) + " "
            + "STATE\n";
        for catalogEntry: DaemonCatalogEntry in catalogEntries {
            let entryModelId: String = catalogEntry.requestableModelId ?? catalogEntry.huggingfaceId;
            let entryState: String;
            if (catalogEntry.readyOnThisMac) {
                entryState = "ready";
            } else if let downloadState: String = catalogEntry.downloadState {
                entryState = downloadState;
            } else {
                entryState = "not on this Mac";
            }
            report += ModelsVerb.padded(entryModelId, width: modelWidth) + " "
                + ModelsVerb.padded(catalogEntry.family, width: 10) + " "
                + ModelsVerb.padded(CliFormatting.formatGigabytes(catalogEntry.approximateSizeBytes), width: 8) + " "
                + entryState + "\n";
        }
        if !stdout.write(report) {
            throw ModelsVerbError.stdoutUnwritable(cause: "standard output is closed");
        }
    }

    /// Left-aligns text in a field, mirroring Rust's `{:<width$}` formatting.
    private static func padded(_ text: String, width: Int) -> String {
        if (text.count >= width) {
            return text;
        }
        return text + String(repeating: " ", count: width - text.count);
    }
}
