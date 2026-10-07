import Foundation;

import IpcProtocol;
import RestContract;

/// Shared graceful-shutdown signal for the daemon, migrating
/// apps/supervisor/src/shutdown_control.rs's ShutdownController: the first
/// caller triggers the request, later callers are idempotent, and the
/// request persists for observers that attach after it fired — the existing
/// SIGINT/SIGTERM path stays untouched beside it.
public final class ShutdownController: @unchecked Sendable {

    private let stateLock: NSLock;
    private var isShutdownRequestedState: Bool;
    private var shutdownHandlers: Array<@Sendable () -> Void>;

    public init() {
        self.stateLock = NSLock();
        self.isShutdownRequestedState = false;
        self.shutdownHandlers = Array<@Sendable () -> Void>();
    }

    /// Whether a shutdown was already requested.
    public var isShutdownRequested: Bool {
        self.stateLock.lock();
        defer { self.stateLock.unlock(); }
        return self.isShutdownRequestedState;
    }

    /// Observes the shutdown signal; a handler attached after the request
    /// fired invokes immediately, so the request is never lost.
    public func subscribe(_ shutdownHandler: @escaping @Sendable () -> Void) -> Void {
        self.stateLock.lock();
        if self.isShutdownRequestedState {
            self.stateLock.unlock();
            shutdownHandler();
            return;
        }
        self.shutdownHandlers.append(shutdownHandler);
        self.stateLock.unlock();
    }

    /// Requests shutdown. Returns whether this was the first caller to
    /// trigger it.
    @discardableResult
    public func requestShutdown() -> Bool {
        self.stateLock.lock();
        if self.isShutdownRequestedState {
            self.stateLock.unlock();
            return false;
        }
        self.isShutdownRequestedState = true;
        let requestedHandlers: Array<@Sendable () -> Void> = self.shutdownHandlers;
        self.shutdownHandlers = Array<@Sendable () -> Void>();
        self.stateLock.unlock();
        for requestedHandler: @Sendable () -> Void in requestedHandlers {
            requestedHandler();
        }
        return true;
    }
}

/// Triggers the daemon's existing graceful-shutdown path, migrating the
/// /v1/control/shutdown arm of shutdown_control.rs: POST answers 202 with
/// the shutting-down document, the route is absent (404) without a
/// controller, and the route table answers other methods with 405.
public enum RestShutdownControlEndpoint {

    public static let routeMethod: String = "POST";
    public static let routePath: String = "/v1/control/shutdown";

    public static func handle(
        _ request: RestHttpRequest,
        shutdownController: ShutdownController?
    ) throws -> RestHttpResponse {
        guard let shutdownController = shutdownController else {
            return RestHttpResponse.text(statusCode: 404, body: "shutdown not supported");
        }
        shutdownController.requestShutdown();
        var shutdownDocument: JsonWireObject = JsonWireObject(entries: Array<(key: String, value: JsonWireValue)>());
        shutdownDocument.appendEntry(key: "status", value: .string("shutting_down"));
        shutdownDocument.appendEntry(
            key: "message",
            value: .string("Astronomical daemon is shutting down"));
        return try RestHttpResponse.json(statusCode: 202, wireValue: .object(shutdownDocument));
    }
}
