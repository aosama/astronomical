import Foundation;

import AstronomicalConfig;

/**
 * Owns snapshot commit and deferred rollback for the narrow live-memory
 * mutation, the Swift port of apps/supervisor/src/maximum_mlx_memory_transaction.rs.
 * The live reloadable snapshot follows the candidate immediately; the
 * configured snapshot always re-reads the persisted document so an external
 * editor never loses its newer intent.
 */
enum MaximumMlxMemoryTransaction {

    /// Applies the candidate to the live snapshot and refreshes the
    /// configured snapshot from the freshly committed document.
    static func commitAppliedConfigSnapshots(
        _ transitionState: ConfigTransitionState,
        resolver: ResolvedRuntimeConfigResolver,
        candidateResolvedConfig: ResolvedRuntimeConfig,
        candidateConfigBytes: Data
    ) -> Void {
        transitionState.replaceReloadableConfig(candidateResolvedConfig);
        if let configuredResolvedConfig: ResolvedRuntimeConfig = MaximumMlxMemoryTransaction.newerConfiguredResolvedConfig(
            resolver: resolver,
            candidateResolvedConfig: candidateResolvedConfig,
            candidateConfigBytes: candidateConfigBytes)
        {
            transitionState.replaceConfiguredConfigSnapshot(configuredResolvedConfig);
        }
    }

    /// Resolves the configured view after the commit: the candidate when the
    /// file still holds exactly the committed bytes, otherwise a fresh load
    /// of whatever now owns the file.
    private static func newerConfiguredResolvedConfig(
        resolver: ResolvedRuntimeConfigResolver,
        candidateResolvedConfig: ResolvedRuntimeConfig,
        candidateConfigBytes: Data
    ) -> ResolvedRuntimeConfig? {
        let currentFileMatchesCandidate: Bool = (try? Data(
            contentsOf: URL(filePath: resolver.stateDirectory.appending(component: "config.json").string)
        )) == candidateConfigBytes;
        if currentFileMatchesCandidate {
            return candidateResolvedConfig;
        }
        return try? resolver.load();
    }

    /// Follows one queued memory raise until the worker finalizes it, then
    /// either keeps the applied snapshots or rolls the live snapshot back to
    /// the prior configuration when the deferred application was rejected.
    /// Runs on its own thread exactly as the Rust tokio::spawn does.
    static func reconcileQueuedMemoryConfig(
        supervisor: WorkerSupervisor,
        transitionState: ConfigTransitionState,
        resolver: ResolvedRuntimeConfigResolver,
        candidateResolvedConfig: ResolvedRuntimeConfig,
        candidateConfigBytes: Data,
        priorResolvedConfig: ResolvedRuntimeConfig?
    ) -> Void {
        while true {
            let workerHealthSnapshot: WorkerHealthSnapshot = supervisor.workerHealthSnapshot();
            if workerHealthSnapshot.pendingMlxMemoryCeilingBytes != nil {
                Thread.sleep(forTimeInterval: 0.025);
                continue;
            }
            transitionState.withTransitionGuard({ () -> Void in
                if transitionState.currentPendingMemoryConfigGeneration()
                    != candidateResolvedConfig.configurationGeneration {
                    return;
                }
                let healthAfterFinalization: WorkerHealthSnapshot = supervisor.workerHealthSnapshot();
                let wasApplied: Bool = healthAfterFinalization.workerRuntimeFeatureConfiguration?
                    .configurationGeneration == candidateResolvedConfig.configurationGeneration
                    && healthAfterFinalization.mlxMemoryLimitError == nil;
                if wasApplied {
                    if let newerConfiguredResolvedConfig: ResolvedRuntimeConfig = MaximumMlxMemoryTransaction.newerConfiguredResolvedConfig(
                        resolver: resolver,
                        candidateResolvedConfig: candidateResolvedConfig,
                        candidateConfigBytes: candidateConfigBytes)
                    {
                        transitionState.replaceConfiguredConfigSnapshot(newerConfiguredResolvedConfig);
                    }
                } else {
                    MaximumMlxMemoryTransaction.reconcileRejectedCandidate(
                        transitionState: transitionState,
                        resolver: resolver,
                        priorResolvedConfig: priorResolvedConfig);
                }
                transitionState.setPendingMemoryConfigGeneration(nil);
            });
            return;
        }
    }

    /// Restores the prior live configuration after a rejection. Persisted
    /// intent may have been edited outside the process; keeping the file
    /// intact is safer than a non-atomic read-then-restore that could
    /// overwrite a concurrent editor.
    private static func reconcileRejectedCandidate(
        transitionState: ConfigTransitionState,
        resolver: ResolvedRuntimeConfigResolver,
        priorResolvedConfig: ResolvedRuntimeConfig?
    ) -> Void {
        if let priorResolvedConfig = priorResolvedConfig {
            transitionState.replaceReloadableConfig(priorResolvedConfig);
        }
        MaximumMlxMemoryTransaction.refreshNewerConfiguredSnapshot(
            transitionState: transitionState,
            resolver: resolver);
    }

    /// Keeps a rejected candidate from clobbering the configured view:
    /// whatever the persisted document now holds becomes the snapshot.
    static func retainRejectedPersistedConfig(
        transitionState: ConfigTransitionState,
        resolver: ResolvedRuntimeConfigResolver
    ) -> Void {
        MaximumMlxMemoryTransaction.refreshNewerConfiguredSnapshot(
            transitionState: transitionState,
            resolver: resolver);
    }

    private static func refreshNewerConfiguredSnapshot(
        transitionState: ConfigTransitionState,
        resolver: ResolvedRuntimeConfigResolver
    ) -> Void {
        if let configuredResolvedConfig: ResolvedRuntimeConfig = try? resolver.load() {
            transitionState.replaceConfiguredConfigSnapshot(configuredResolvedConfig);
        }
    }
}
