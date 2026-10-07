import Foundation;

import IpcProtocol;

/// The thread-safe abandonment signal one streaming client controls. Kept
/// separate from the public handle so the supervisor's generation thread can
/// poll the signal without extending the client's ownership: releasing the
/// handle trips the signal through its deinit even while the generation is
/// still running.
final class ChatGenerationAbandonmentFlag: @unchecked Sendable {

    private let abandonmentLock: NSLock;
    private var isAbandoned: Bool;

    init() {
        self.abandonmentLock = NSLock();
        self.isAbandoned = false;
    }

    func abandon() -> Void {
        self.abandonmentLock.lock();
        self.isAbandoned = true;
        self.abandonmentLock.unlock();
    }

    func clientAbandoned() -> Bool {
        self.abandonmentLock.lock();
        defer { self.abandonmentLock.unlock(); }
        return self.isAbandoned;
    }
}

/// The sentinel that diverts one streaming generation into its cancellation
/// path; only the streaming surface throws and catches it.
enum ChatGenerationClientAbandonment: Error {

    case abandonedByClient;
}

/// The client sink wrapped for cross-thread delivery; the closure itself is
/// not Sendable, this box is the single transfer point.
final class ChatGenerationEventSink: @unchecked Sendable {

    private let deliverEvent: (ChatGenerationStreamEvent) -> Void;

    init(deliverEvent: @escaping (ChatGenerationStreamEvent) -> Void) {
        self.deliverEvent = deliverEvent;
    }

    func deliver(_ streamEvent: ChatGenerationStreamEvent) -> Void {
        self.deliverEvent(streamEvent);
    }
}

/// The client-owned handle of one streaming chat generation, the Swift
/// counterpart of the Rust mpsc stream receiver: the generation delivers its
/// events through the sink while the client holds the handle, and releasing
/// (or explicitly abandoning) it is the disconnect the supervisor must
/// cancel through.
public final class ChatGenerationStreamHandle: @unchecked Sendable {

    private let abandonmentFlag: ChatGenerationAbandonmentFlag;

    init(abandonmentFlag: ChatGenerationAbandonmentFlag) {
        self.abandonmentFlag = abandonmentFlag;
    }

    /// Marks the client as gone; the in-flight generation observes this
    /// within its next poll tick and runs the bounded cancellation path.
    public func abandon() -> Void {
        self.abandonmentFlag.abandon();
    }

    deinit {
        self.abandon();
    }
}

/// Everything one streaming generation thread needs, boxed for the one
/// cross-thread handoff from the admitting caller: the command is a value
/// type this module never mutates after admission.
final class ChatGenerationStreamDispatch: @unchecked Sendable {

    let generationCommand: ChatGenerationCommand;
    let abandonmentFlag: ChatGenerationAbandonmentFlag;
    let eventSink: ChatGenerationEventSink;

    init(
        generationCommand: ChatGenerationCommand,
        abandonmentFlag: ChatGenerationAbandonmentFlag,
        eventSink: ChatGenerationEventSink
    ) {
        self.generationCommand = generationCommand;
        self.abandonmentFlag = abandonmentFlag;
        self.eventSink = eventSink;
    }
}

/// The streaming chat surface of the supervisor, the disconnect-aware
/// counterpart of startChatGeneration: one dedicated execution thread owns
/// the request after admission, forwards every public stream event to the
/// sink as it is produced, and — when the client abandons the handle —
/// cancels the request and replaces the worker when the worker cannot
/// acknowledge that cancellation.
extension WorkerSupervisor {

    /// Starts one chat generation whose events stream to the sink while the
    /// returned handle is alive. Throws GenerationStartError exactly when
    /// startChatGeneration would, before any thread starts.
    public func startChatGenerationStream(
        _ generationCommand: ChatGenerationCommand,
        onEvent: @escaping (ChatGenerationStreamEvent) -> Void
    ) throws -> ChatGenerationStreamHandle {
        try self.admitGenerationSlot();
        let abandonmentFlag: ChatGenerationAbandonmentFlag = ChatGenerationAbandonmentFlag();
        let streamHandle: ChatGenerationStreamHandle = ChatGenerationStreamHandle(
            abandonmentFlag: abandonmentFlag);
        let streamDispatch: ChatGenerationStreamDispatch = ChatGenerationStreamDispatch(
            generationCommand: generationCommand,
            abandonmentFlag: abandonmentFlag,
            eventSink: ChatGenerationEventSink(deliverEvent: onEvent));
        let supervisor: WorkerSupervisor = self;
        let generationThread: Thread = Thread(block: { () -> Void in
            supervisor.runStreamingGeneration(streamDispatch);
        });
        generationThread.name = "astronomical-chat-generation-stream";
        generationThread.start();
        return streamHandle;
    }

    /// The streaming generation body: identical startup, swap, and dispatch
    /// to runGeneration, but events forward through the sink and a client
    /// abandonment diverts into the cancellation path instead of surfacing
    /// as an error to a caller that no longer exists.
    private func runStreamingGeneration(
        _ streamDispatch: ChatGenerationStreamDispatch
    ) -> Void {
        let generationCommand: ChatGenerationCommand = streamDispatch.generationCommand;
        let abandonmentFlag: ChatGenerationAbandonmentFlag = streamDispatch.abandonmentFlag;
        defer { self.finishAdmissionSlot(); }
        self.stateLock.lock();
        let workerProcess: WorkerProcess? = self.workerProcess;
        let eventPump: WorkerEventPump? = self.eventPump;
        self.stateLock.unlock();
        guard let workerProcess = workerProcess, let eventPump = eventPump else {
            return;
        }
        do {
            try WorkerStartupRuntime.waitForStartupRuntimeConfiguration(
                workerProcess: workerProcess,
                eventPump: eventPump,
                healthState: self.healthState,
                modelLoadTimeout: self.modelLoadTimeout);
            try WorkerGenerate.prepareResidentModel(
                targetModelId: generationCommand.model,
                workerProcess: workerProcess,
                eventPump: eventPump,
                healthState: self.healthState,
                modelPolicyCatalog: self.modelPolicyCatalog,
                modelLoadTimeout: self.modelLoadTimeout,
                containment: { (controlError: Error) -> Void in
                    self.containWorkerFailure(controlError: controlError);
                });
            try workerProcess.sendCommand(.generate(generationCommand));
            self.publishServingActivity(.promptProcessing, progress: nil);
            _ = try self.collectGenerationEvents(
                generationCommand.requestId,
                requestStartedAt: Date(),
                maximumOutputTokens: generationCommand.settings.maxOutputTokens,
                eventPump: eventPump,
                onStreamEvent: { (streamEvent: ChatGenerationStreamEvent) -> Void in
                    streamDispatch.eventSink.deliver(streamEvent);
                },
                isClientAbandoned: { () -> Bool in
                    return abandonmentFlag.clientAbandoned();
                });
        } catch ChatGenerationClientAbandonment.abandonedByClient {
            self.cancelActiveGeneration(
                requestId: generationCommand.requestId,
                workerProcess: workerProcess,
                eventPump: eventPump,
                expectsImageFinalization: false);
        } catch let controlError {
            self.stateLock.lock();
            let isShutdownRequested: Bool = self.isShutdownRequested;
            self.stateLock.unlock();
            if !isShutdownRequested {
                // The client still owns the handle, so the terminal word
                // belongs to it before the worker is taken down: the Rust
                // loop delivers Error(WorkerUnavailable) to the stream and
                // terminates the worker in the same breach.
                streamDispatch.eventSink.deliver(.streamError(.workerUnavailable));
                self.containWorkerFailure(controlError: controlError);
            }
        }
    }
}
