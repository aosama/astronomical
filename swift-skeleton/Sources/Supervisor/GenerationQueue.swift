import Foundation;

import IpcProtocol;

/// How many requests can wait in the admission queue while one request owns
/// the worker. The request after an active slot plus this many waiters is
/// rejected immediately with capacityUnavailable, which the REST surface maps
/// to HTTP 429.
public enum GenerationQueueDepth {

    public static let maximumWaiterCount: Int = 8;
}

/// The bounded FIFO admission path shared by chat, image, and embeddings
/// requests, migrating apps/supervisor/src/worker_generation_admission.rs.
///
/// The Rust handle uses a queue semaphore (eight permits) plus an active
/// semaphore (one permit); this synchronous port issues monotonic admission
/// tickets on the supervisor's condition lock instead. A ticket holder owns
/// the worker in arrival order, `wait()` parks later tickets, and the
/// `finishAdmissionSlot()` release advances the served ticket so the next
/// waiter wakes. Abandoned tickets (a waiter that observed shutdown or a
/// contained worker) are skipped so the queue always drains forward.
extension WorkerSupervisor {

    /// Reserves one admission ticket and blocks the caller until its turn
    /// owns the worker. Refusal paths never leave a ticket outstanding:
    /// capacity rejection happens before ticketing, and a waiter that
    /// observes shutdown or a dead worker abandons its ticket and cascades
    /// the queue forward before throwing.
    func admitGenerationSlot() throws -> Void {
        self.stateLock.lock();
        if self.isShutdownRequested || self.workerProcess == nil {
            self.stateLock.unlock();
            throw GenerationStartError.workerUnavailable;
        }
        let outstandingAdmissionTicketCount: Int = self.issuedAdmissionTicketCount - self.servedAdmissionTicket;
        if outstandingAdmissionTicketCount > GenerationQueueDepth.maximumWaiterCount {
            self.stateLock.unlock();
            throw GenerationStartError.capacityUnavailable;
        }
        let admissionTicket: Int = self.issuedAdmissionTicketCount;
        self.issuedAdmissionTicketCount += 1;
        while admissionTicket != self.servedAdmissionTicket {
            self.skipAbandonedHeadTicketsWhileLocked();
            if self.isShutdownRequested || self.workerProcess == nil {
                self.abandonedAdmissionTickets.insert(admissionTicket);
                self.stateLock.broadcast();
                self.stateLock.unlock();
                throw GenerationStartError.workerUnavailable;
            }
            self.stateLock.wait();
        }
        self.skipAbandonedHeadTicketsWhileLocked();
        if self.isShutdownRequested || self.workerProcess == nil {
            // The turn arrived after the world ended: hand the slot back
            // exactly as a finished generation would. The caller has no
            // deferred release yet because admission never returned.
            self.servedAdmissionTicket += 1;
            self.skipAbandonedHeadTicketsWhileLocked();
            self.stateLock.broadcast();
            self.stateLock.unlock();
            throw GenerationStartError.workerUnavailable;
        }
        self.stateLock.unlock();
    }

    /// Releases the active slot after one generation finished: any memory
    /// raise queued behind the generation is applied first so the next
    /// admission observes the new ceiling, then the served ticket advances
    /// and the next waiter wakes.
    func finishAdmissionSlot() -> Void {
        self.applyPendingMlxMemoryLimitUpdateAfterFinalization();
        self.stateLock.lock();
        self.servedAdmissionTicket += 1;
        self.skipAbandonedHeadTicketsWhileLocked();
        self.stateLock.broadcast();
        self.stateLock.unlock();
    }

    /// Advances past abandoned head tickets; the caller holds the lock.
    private func skipAbandonedHeadTicketsWhileLocked() -> Void {
        while self.abandonedAdmissionTickets.contains(self.servedAdmissionTicket) {
            self.abandonedAdmissionTickets.remove(self.servedAdmissionTicket);
            self.servedAdmissionTicket += 1;
        }
    }

    /// The issued admission tickets that have not reached the worker yet:
    /// the active request plus every queued waiter. Journeys observe queue
    /// fill through this count so their expectations never depend on thread
    /// scheduling order.
    var outstandingAdmissionTicketCount: Int {
        self.stateLock.lock();
        defer { self.stateLock.unlock(); }
        return self.issuedAdmissionTicketCount - self.servedAdmissionTicket;
    }
}
