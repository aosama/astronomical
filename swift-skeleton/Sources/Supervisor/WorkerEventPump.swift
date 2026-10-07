import Foundation;

import IpcProtocol;

/// Delivers worker events to a waiter that must not block forever.
///
/// The Rust supervisor wraps its event reads in `tokio::time::timeout`, which
/// cancels the read future on deadline. A synchronous pipe read cannot be
/// cancelled, so this pump drains the worker's event stream on its own
/// thread and lets the waiter give up after a bounded wait while the pump
/// thread stays parked in the read until the worker closes its output —
/// the same observable semantics with the read parked instead of cancelled.
public final class WorkerEventPump: @unchecked Sendable {

    private let workerProcess: WorkerProcess;
    private let pumpCondition: NSCondition;
    private var pendingEvents: Array<WorkerEvent>;
    private var isEventStreamFinished: Bool;
    private var streamFailure: Error?;
    /// Incremented on every relaunch resume so a drain thread parked on the
    /// replaced process's pipe can never publish into the new generation.
    private var drainGeneration: Int;

    public init(workerProcess: WorkerProcess) {
        self.workerProcess = workerProcess;
        self.pumpCondition = NSCondition();
        self.pendingEvents = Array<WorkerEvent>();
        self.isEventStreamFinished = false;
        self.streamFailure = nil;
        self.drainGeneration = 1;
        let initialGeneration: Int = self.drainGeneration;
        let drainThread: Thread = Thread(block: { self.drainUntilClosed(generation: initialGeneration); });
        drainThread.name = "astronomicald-worker-event-pump";
        drainThread.start();
    }

    /// Returns the next event, throws the stream failure when the reader hit
    /// one, throws the composed process-exit diagnostics (exit status,
    /// lifetime, stderr tail) when the worker closed its output, and returns
    /// `nil` only when the bounded wait expired.
    public func nextEvent(within maximumWait: TimeInterval) throws -> WorkerEvent? {
        self.pumpCondition.lock();
        defer { self.pumpCondition.unlock(); }
        let waitDeadline: Date = Date().addingTimeInterval(maximumWait);
        while self.pendingEvents.isEmpty && !self.isEventStreamFinished && Date() < waitDeadline {
            self.pumpCondition.wait(until: waitDeadline);
        }
        if !self.pendingEvents.isEmpty {
            return self.pendingEvents.removeFirst();
        }
        if let readFailure: Error = self.streamFailure {
            throw readFailure;
        }
        if self.isEventStreamFinished {
            throw WorkerControlError.workerEventStreamClosed;
        }
        return nil;
    }

    /// Clears the latched stream end and starts a fresh drain thread after
    /// the owning process relaunched; stale events and failures from the
    /// replaced process are discarded, exactly like the Rust loop's fresh
    /// event reader.
    public func resumeAfterRelaunch() -> Void {
        self.pumpCondition.lock();
        defer { self.pumpCondition.unlock(); }
        self.drainGeneration += 1;
        let resumedGeneration: Int = self.drainGeneration;
        self.pendingEvents.removeAll();
        self.isEventStreamFinished = false;
        self.streamFailure = nil;
        let drainThread: Thread = Thread(block: { self.drainUntilClosed(generation: resumedGeneration); });
        drainThread.name = "astronomicald-worker-event-pump";
        drainThread.start();
    }

    private func drainUntilClosed(generation: Int) -> Void {
        while true {
            self.pumpCondition.lock();
            let isSuperseded: Bool = generation != self.drainGeneration;
            self.pumpCondition.unlock();
            if isSuperseded {
                return;
            }
            do {
                // A nil return is the clean frame-boundary EOF; the stream
                // end carries the process diagnostics the way the Rust
                // worker's next_event composes its exit error. Anything
                // thrown is a framing or decoding failure on this stream.
                guard let workerEvent: WorkerEvent = try self.workerProcess.nextEvent() else {
                    self.finishStream(
                        failure: self.workerProcess.workerProcessExitError(),
                        generation: generation);
                    return;
                }
                self.enqueue(workerEvent)
            } catch let readFailure {
                self.finishStream(failure: readFailure, generation: generation)
                return
            }
        }
    }

    private func enqueue(_ workerEvent: WorkerEvent) -> Void {
        self.pumpCondition.lock();
        defer { self.pumpCondition.unlock(); }
        self.pendingEvents.append(workerEvent);
        self.pumpCondition.signal();
    }

    private func finishStream(failure: Error?, generation: Int) -> Void {
        self.pumpCondition.lock();
        defer { self.pumpCondition.unlock(); }
        // A thread parked on the replaced process's pipe must never latch
        // its stream end into the relaunched generation.
        if generation != self.drainGeneration {
            return;
        }
        self.isEventStreamFinished = true;
        self.streamFailure = failure;
        self.pumpCondition.broadcast();
    }
}
