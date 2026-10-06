import Foundation;

import AstronomicalConfig;
import IpcProtocol;

/// The generation-facing face the daemon serves chat through.
///
/// Mirrors apps/supervisor/src/chat_generation_executor.rs's
/// ChatGenerationExecutor trait: the daemon IPC surface and the REST surface
/// both see this boundary, never the worker process directly. The synchronous
/// Swift serving model returns the ordered stream events of one request
/// instead of a channel; the terminal event is always present unless the
/// start itself failed.
public protocol ChatGenerationExecuting: Sendable {

    /// Starts one bounded chat generation and returns its ordered stream
    /// events, ending with the terminal event. Throws GenerationStartError
    /// when the request cannot start.
    func startChatGeneration(
        _ generationCommand: ChatGenerationCommand
    ) throws -> Array<ChatGenerationStreamEvent>;

    /// One consistent worker health snapshot for gating and status.
    func workerHealthSnapshot() -> WorkerHealthSnapshot;
}

/// Everything the daemon IPC chat verb needs beyond the shared health
/// provider: the executor, the live resolved configuration, and the instance
/// paths the optional thinking-channel seed reads from.
public struct DaemonIpcChatContext: @unchecked Sendable {

    let chatExecutor: any ChatGenerationExecuting;
    let resolvedRuntimeConfig: ResolvedRuntimeConfig;
    let instancePaths: AstronomicalInstancePaths;

    public init(
        chatExecutor: any ChatGenerationExecuting,
        resolvedRuntimeConfig: ResolvedRuntimeConfig,
        instancePaths: AstronomicalInstancePaths
    ) {
        self.chatExecutor = chatExecutor;
        self.resolvedRuntimeConfig = resolvedRuntimeConfig;
        self.instancePaths = instancePaths;
    }
}

/// Supervisor-local monotonic request identifiers.
///
/// Mirrors apps/supervisor/src/application.rs's allocate_chat_request_id:
/// one counter per daemon process, incremented under a lock, exhausted only
/// at the UInt64 ceiling.
public final class ChatRequestIdAllocator: @unchecked Sendable {

    private let allocatorLock: NSLock;
    private var nextRequestId: UInt64;

    public init() {
        self.allocatorLock = NSLock();
        self.nextRequestId = 1;
    }

    /// Returns the next identifier, or nil once the numeric space is spent.
    public func allocate() -> UInt64? {
        self.allocatorLock.lock();
        defer { self.allocatorLock.unlock(); }
        guard self.nextRequestId < UInt64.max else {
            return nil;
        }
        let allocatedRequestId: UInt64 = self.nextRequestId;
        self.nextRequestId += 1;
        return allocatedRequestId;
    }
}
