import Foundation;

/**
 * Reconciles a live memory limit queued by config reload after the active
 * generation finalizes, the Swift port of
 * apps/supervisor/src/queued_memory_reload.rs: a rejected deferred raise
 * rolls the live snapshot back to the prior configuration; an applied one
 * keeps the reload.
 */
enum QueuedMemoryReload {

    static func reconcileReloadedMemoryConfig(
        supervisor: WorkerSupervisor,
        transitionState: ConfigTransitionState,
        effectiveMemoryGeneration: String,
        priorResolvedConfig: ResolvedRuntimeConfig
    ) -> Void {
        while true {
            let workerHealthSnapshot: WorkerHealthSnapshot = supervisor.workerHealthSnapshot();
            if workerHealthSnapshot.pendingMlxMemoryCeilingBytes != nil {
                Thread.sleep(forTimeInterval: 0.025);
                continue;
            }
            transitionState.withTransitionGuard({ () -> Void in
                if transitionState.currentPendingMemoryConfigGeneration() != effectiveMemoryGeneration {
                    return;
                }
                let healthAfterFinalization: WorkerHealthSnapshot = supervisor.workerHealthSnapshot();
                let wasApplied: Bool = healthAfterFinalization.workerRuntimeFeatureConfiguration?
                    .configurationGeneration == effectiveMemoryGeneration
                    && healthAfterFinalization.mlxMemoryLimitError == nil;
                if !wasApplied {
                    transitionState.replaceReloadableConfig(priorResolvedConfig);
                }
                transitionState.setPendingMemoryConfigGeneration(nil);
            });
            return;
        }
    }
}
