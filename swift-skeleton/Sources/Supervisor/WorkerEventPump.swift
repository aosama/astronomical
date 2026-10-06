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

    public init(workerProcess: WorkerProcess) {
        self.workerProcess = workerProcess;
        self.pumpCondition = NSCondition();
        self.pendingEvents = Array<WorkerEvent>();
        self.isEventStreamFinished = false;
        let drainThread: Thread = Thread(block: { self.drainUntilClosed(); });
        drainThread.name = "astronomicald-worker-event-pump";
        drainThread.start();
    }

    /// Returns the next event, throws the stream failure when the reader hit
    /// one, throws `workerEventStreamClosed` when the worker closed its
    /// output, and returns `nil` only when the bounded wait expired.
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

    private func drainUntilClosed() -> Void {
        while true {
            do {
                // A nil return is the clean frame-boundary EOF; anything
                // thrown is a framing or decoding failure on this stream.
                guard let workerEvent: WorkerEvent = try self.workerProcess.nextEvent() else {
                    self.finishStream(failure: nil);
                    return;
                }
                self.enqueue(workerEvent);
            } catch let readFailure {
                self.finishStream(failure: readFailure);
                return;
            }
        }
    }

    private func enqueue(_ workerEvent: WorkerEvent) -> Void {
        self.pumpCondition.lock();
        defer { self.pumpCondition.unlock(); }
        self.pendingEvents.append(workerEvent);
        self.pumpCondition.signal();
    }

    private func finishStream(failure: Error?) -> Void {
        self.pumpCondition.lock();
        defer { self.pumpCondition.unlock(); }
        self.isEventStreamFinished = true;
        self.streamFailure = failure;
        self.pumpCondition.broadcast();
    }
}
