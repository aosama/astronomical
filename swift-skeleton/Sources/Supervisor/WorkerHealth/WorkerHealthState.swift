import Foundation;

import IpcProtocol;

/// Lock-guarded owner of the supervisor's current worker health snapshot.
///
/// Replaces the `Arc<RwLock<WorkerHealthSnapshot>>` of
/// apps/supervisor/src/worker_health.rs: the worker event thread publishes
/// snapshots and field updates while the daemon IPC thread reads them for the
/// status verb, so every access crosses threads under one lock.
public final class WorkerHealthState {

    private let stateLock: NSLock;
    private var snapshot: WorkerHealthSnapshot;
    /// Whether the current worker process already sent its one lifecycle
    /// event (Ready or Idle). The Rust worker loop keeps this as a loop-local
    /// `is_ready` flag; it lives beside the snapshot here because the Swift
    /// shell drives the loop from several call sites.
    private var lifecycleAcknowledged: Bool;

    public init() {
        self.stateLock = NSLock();
        self.snapshot = WorkerHealthSnapshot.unavailable(.loading);
        self.lifecycleAcknowledged = false;
    }

    /// A copy of the current snapshot.
    public func currentSnapshot() -> WorkerHealthSnapshot {
        self.stateLock.lock();
        defer { self.stateLock.unlock(); }
        return self.snapshot;
    }

    /// The status-verb view of the current snapshot.
    public func daemonStatusReport() -> DaemonStatusReport {
        self.stateLock.lock();
        defer { self.stateLock.unlock(); }
        return DaemonStatusReport(
            workerStatus: self.snapshot.status.daemonWorkerStatus(),
            readyModelId: self.snapshot.readyModelId);
    }

    /// Whether the worker has acknowledged its startup runtime policy.
    public func hasRuntimeFeatureConfiguration() -> Bool {
        self.stateLock.lock();
        defer { self.stateLock.unlock(); }
        return self.snapshot.workerRuntimeFeatureConfiguration != nil;
    }

    /// Whether the current worker already sent its one lifecycle event.
    public func hasAcknowledgedLifecycle() -> Bool {
        self.stateLock.lock();
        defer { self.stateLock.unlock(); }
        return self.lifecycleAcknowledged;
    }

    /// Records that the current worker sent its one lifecycle event.
    public func markLifecycleAcknowledged() {
        self.stateLock.lock();
        defer { self.stateLock.unlock(); }
        self.lifecycleAcknowledged = true;
    }

    /// Replaces the whole snapshot, as the Ready and Idle events do.
    public func publish(_ replacementSnapshot: WorkerHealthSnapshot) {
        self.stateLock.lock();
        defer { self.stateLock.unlock(); }
        self.snapshot = replacementSnapshot;
    }

    /// Mutates selected fields under the lock, as the field-level Rust
    /// publisher helpers do.
    public func apply(_ transform: (inout WorkerHealthSnapshot) throws -> Void) throws {
        self.stateLock.lock();
        defer { self.stateLock.unlock(); }
        var mutableSnapshot: WorkerHealthSnapshot = self.snapshot;
        try transform(&mutableSnapshot);
        self.snapshot = mutableSnapshot;
    }
}
