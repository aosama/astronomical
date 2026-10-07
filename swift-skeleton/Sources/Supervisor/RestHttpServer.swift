import Foundation;

import AstronomicalConfig;

/**
 * The loopback HTTP/1.1 listener the daemon's REST surface answers on.
 *
 * Hand-rolled on the same POSIX + dedicated-thread idiom as the daemon IPC
 * service, deliberately without an HTTP framework dependency: the surface is
 * closed loopback with no external consumers, so the transport owns a small
 * accepted grammar (request line, headers, Content-Length bodies within a
 * cap) and rejects everything else with the shared failure envelope. Every
 * connection serves exactly one request and closes; a per-connection queue
 * dispatch keeps one stalled client from blocking the accept loop.
 */
public final class RestHttpServer: @unchecked Sendable {

    public static let DEFAULT_MAXIMUM_REQUEST_BODY_BYTES: UInt = 8_388_608;
    private static let LISTEN_BACKLOG: Int32 = 16;
    private static let SHUTDOWN_JOIN_SECONDS: Double = 5;
    private static let ACCEPT_ERROR_RETRY_MICROSECONDS: UInt32 = 10_000;

    private let listenerFileDescriptor: Int32;
    private let boundEndpointValue: SocketEndpoint;
    private let servingRouteTable: RestRouteTable;
    private let servingMaximumRequestBodyBytes: UInt;
    private let servingCorsPolicy: RestCorsPolicy?;
    private let servingPerformanceAttributionEnabled: Bool;
    private let stateLock: NSLock;
    private var isShutdownRequested: Bool;
    private var servingThread: Thread?;

    private init(
        listenerFileDescriptor: Int32,
        boundEndpointValue: SocketEndpoint,
        servingRouteTable: RestRouteTable,
        servingMaximumRequestBodyBytes: UInt,
        servingCorsPolicy: RestCorsPolicy?,
        servingPerformanceAttributionEnabled: Bool
    ) {
        self.listenerFileDescriptor = listenerFileDescriptor;
        self.boundEndpointValue = boundEndpointValue;
        self.servingRouteTable = servingRouteTable;
        self.servingMaximumRequestBodyBytes = servingMaximumRequestBodyBytes;
        self.servingCorsPolicy = servingCorsPolicy;
        self.servingPerformanceAttributionEnabled = servingPerformanceAttributionEnabled;
        self.stateLock = NSLock();
        self.isShutdownRequested = false;
        self.servingThread = nil;
    }

    /// The endpoint actually bound — an ephemeral port publishes its
    /// assigned value here, never 0.
    public var boundEndpoint: SocketEndpoint {
        return self.boundEndpointValue;
    }

    /// Starts serving the given routes on the loopback endpoint.
    ///
    /// The accept loop runs on its own thread because the listener blocks
    /// per connection; each accepted connection is served on a concurrent
    /// dispatch queue so a silent client never stalls the endpoint.
    public static func start(
        bindEndpoint: SocketEndpoint,
        routeTable: RestRouteTable,
        maximumRequestBodyBytes: UInt = RestHttpServer.DEFAULT_MAXIMUM_REQUEST_BODY_BYTES,
        corsPolicy: RestCorsPolicy? = nil,
        performanceAttributionEnabled: Bool = false
    ) throws -> RestHttpServer {
        let listenerFileDescriptor: Int32 = socket(AF_INET, SOCK_STREAM, 0);
        guard listenerFileDescriptor >= 0 else {
            throw RestHttpServerError.socketOperationFailed(
                operation: "socket",
                detail: String(cString: strerror(errno)));
        }
        var reuseAddressFlag: Int32 = 1;
        _ = setsockopt(
            listenerFileDescriptor, SOL_SOCKET, SO_REUSEADDR,
            &reuseAddressFlag, socklen_t(MemoryLayout<Int32>.size));

        var bindAddress: sockaddr_in = sockaddr_in();
        bindAddress.sin_len = UInt8(MemoryLayout<sockaddr_in>.size);
        bindAddress.sin_family = sa_family_t(AF_INET);
        bindAddress.sin_port = in_port_t(bindEndpoint.port).bigEndian;
        guard inet_pton(AF_INET, bindEndpoint.host, &bindAddress.sin_addr) == 1 else {
            close(listenerFileDescriptor);
            throw RestHttpServerError.invalidBindHost(host: bindEndpoint.host);
        }
        let bindOutcome: Int32 = withUnsafePointer(to: &bindAddress, { (addressPointer: UnsafePointer<sockaddr_in>) -> Int32 in
            return bind(
                listenerFileDescriptor,
                UnsafePointer<sockaddr>(OpaquePointer(addressPointer)),
                socklen_t(MemoryLayout<sockaddr_in>.size));
        });
        guard bindOutcome == 0 else {
            close(listenerFileDescriptor);
            throw RestHttpServerError.socketOperationFailed(
                operation: "bind",
                detail: String(cString: strerror(errno)));
        }
        guard listen(listenerFileDescriptor, RestHttpServer.LISTEN_BACKLOG) == 0 else {
            close(listenerFileDescriptor);
            throw RestHttpServerError.socketOperationFailed(
                operation: "listen",
                detail: String(cString: strerror(errno)));
        }

        var boundAddress: sockaddr_in = sockaddr_in();
        var boundAddressLength: socklen_t = socklen_t(MemoryLayout<sockaddr_in>.size);
        let boundPort: UInt16 = withUnsafeMutablePointer(to: &boundAddress, { (addressPointer: UnsafeMutablePointer<sockaddr_in>) -> UInt16 in
            _ = getsockname(
                listenerFileDescriptor,
                UnsafeMutablePointer<sockaddr>(OpaquePointer(addressPointer)),
                &boundAddressLength);
            return UInt16(bigEndian: addressPointer.pointee.sin_port);
        });

        let server: RestHttpServer = RestHttpServer(
            listenerFileDescriptor: listenerFileDescriptor,
            boundEndpointValue: SocketEndpoint(host: bindEndpoint.host, port: boundPort),
            servingRouteTable: routeTable,
            servingMaximumRequestBodyBytes: maximumRequestBodyBytes,
            servingCorsPolicy: corsPolicy,
            servingPerformanceAttributionEnabled: performanceAttributionEnabled);
        let servingThread: Thread = Thread {
            server.serveUntilShutdown();
        };
        servingThread.name = "astronomicald-rest-endpoint";
        servingThread.start();
        server.stateLock.lock();
        server.servingThread = servingThread;
        server.stateLock.unlock();
        return server;
    }

    /// Stops serving, waits for the accept thread to end, and closes the
    /// listener so clients cannot connect to a dead endpoint. Stopping an
    /// already-stopped server is a no-op — the listener descriptor is
    /// closed exactly once, so a reused descriptor is never closed again.
    public func stop() -> Void {
        self.stateLock.lock();
        let wasAlreadyShutdown: Bool = self.isShutdownRequested;
        self.isShutdownRequested = true;
        self.stateLock.unlock();
        if wasAlreadyShutdown {
            return;
        }
        let servingThread: Thread? = self.servingThread;
        // A throwaway loopback connection unblocks the accept that is
        // waiting for a client; the loop then observes the flag and exits.
        let wakeupDescriptor: Int32 = socket(AF_INET, SOCK_STREAM, 0);
        if wakeupDescriptor >= 0 {
            var wakeupAddress: sockaddr_in = sockaddr_in();
            wakeupAddress.sin_len = UInt8(MemoryLayout<sockaddr_in>.size);
            wakeupAddress.sin_family = sa_family_t(AF_INET);
            wakeupAddress.sin_port = in_port_t(self.boundEndpointValue.port).bigEndian;
            wakeupAddress.sin_addr = in_addr(s_addr: inet_addr("127.0.0.1"));
            _ = withUnsafePointer(to: &wakeupAddress, { (addressPointer: UnsafePointer<sockaddr_in>) -> Int32 in
                return connect(
                    wakeupDescriptor,
                    UnsafePointer<sockaddr>(OpaquePointer(addressPointer)),
                    socklen_t(MemoryLayout<sockaddr_in>.size));
            });
            close(wakeupDescriptor);
        }
        if let servingThread: Thread = servingThread {
            let joinDeadline: Date = Date().addingTimeInterval(RestHttpServer.SHUTDOWN_JOIN_SECONDS);
            while servingThread.isExecuting && Date() < joinDeadline {
                Thread.sleep(forTimeInterval: 0.01);
            }
        }
        close(self.listenerFileDescriptor);
    }

    private func serveUntilShutdown() -> Void {
        while true {
            self.stateLock.lock();
            let shouldShutdown: Bool = self.isShutdownRequested;
            self.stateLock.unlock();
            if shouldShutdown {
                return;
            }
            let acceptStart: ContinuousClock.Instant? = RestHttpPerformanceAttribution.startedOperation(
                operationName: "rest_accept",
                performanceAttributionEnabled: self.servingPerformanceAttributionEnabled);
            let connectionDescriptor: Int32 = accept(self.listenerFileDescriptor, nil, nil);
            RestHttpPerformanceAttribution.finishedOperation(
                operationName: "rest_accept",
                operationStart: acceptStart,
                operationOutcome: connectionDescriptor >= 0 ? "success" : "failed",
                performanceAttributionEnabled: self.servingPerformanceAttributionEnabled);
            if connectionDescriptor < 0 {
                self.stateLock.lock();
                let shouldShutdownAfterFailedAccept: Bool = self.isShutdownRequested;
                self.stateLock.unlock();
                if shouldShutdownAfterFailedAccept {
                    return;
                }
                usleep(RestHttpServer.ACCEPT_ERROR_RETRY_MICROSECONDS);
                continue;
            }
            self.stateLock.lock();
            let shouldShutdownBeforeServing: Bool = self.isShutdownRequested;
            self.stateLock.unlock();
            if shouldShutdownBeforeServing {
                close(connectionDescriptor);
                return;
            }
            let servingRouteTable: RestRouteTable = self.servingRouteTable;
            let servingMaximumRequestBodyBytes: UInt = self.servingMaximumRequestBodyBytes;
            let servingCorsPolicy: RestCorsPolicy? = self.servingCorsPolicy;
            let servingPerformanceAttributionEnabled: Bool = self.servingPerformanceAttributionEnabled;
            DispatchQueue.global().async(execute: {
                RestHttpServer.serveSingleRequest(
                    connectionFileDescriptor: connectionDescriptor,
                    routeTable: servingRouteTable,
                    maximumRequestBodyBytes: servingMaximumRequestBodyBytes,
                    corsPolicy: servingCorsPolicy,
                    performanceAttributionEnabled: servingPerformanceAttributionEnabled);
            });
        }
    }

    private static func serveSingleRequest(
        connectionFileDescriptor: Int32,
        routeTable: RestRouteTable,
        maximumRequestBodyBytes: UInt,
        corsPolicy: RestCorsPolicy?,
        performanceAttributionEnabled: Bool
    ) -> Void {
        let connection: RestHttpConnection = RestHttpConnection(fileDescriptor: connectionFileDescriptor);
        defer { connection.discard(); }

        var response: RestHttpResponse?;
        var endpointHandleStart: ContinuousClock.Instant?;
        var parsedRequestOrigin: String? = nil;
        let parseStart: ContinuousClock.Instant? = RestHttpPerformanceAttribution.startedOperation(
            operationName: "rest_request_parse",
            performanceAttributionEnabled: performanceAttributionEnabled);
        do {
            let request: RestHttpRequest = try RestHttpRequestParser.parseRequest(
                connection: connection,
                maximumRequestBodyBytes: maximumRequestBodyBytes);
            parsedRequestOrigin = request.headerValue(named: "Origin");
            RestHttpPerformanceAttribution.finishedOperation(
                operationName: "rest_request_parse",
                operationStart: parseStart,
                operationOutcome: "success",
                performanceAttributionEnabled: performanceAttributionEnabled);
            // A CORS preflight never reaches the route table: it asks the
            // transport whether the method and headers may cross the origin,
            // whatever path the real call will later target.
            if let preflightResponse: RestHttpResponse = RestHttpServer.corsPreflightAnswer(
                request: request,
                corsPolicy: corsPolicy) {
                response = preflightResponse;
            } else {
                switch (routeTable.outcome(method: request.method, path: request.path)) {
                case let .handler(endpointHandler):
                    endpointHandleStart = RestHttpPerformanceAttribution.startedOperation(
                        operationName: "rest_endpoint_handle",
                        performanceAttributionEnabled: performanceAttributionEnabled);
                    response = try endpointHandler(request);
                    RestHttpPerformanceAttribution.finishedOperation(
                        operationName: "rest_endpoint_handle",
                        operationStart: endpointHandleStart,
                        operationOutcome: "success",
                        performanceAttributionEnabled: performanceAttributionEnabled);
                case let .methodNotAllowed(allowedMethods):
                    let failureEnvelope: RestHttpResponse = try RestEndpointFailure(
                        statusCode: 405,
                        message: "method \(request.method) is not allowed on \(request.path)")
                        .envelopeResponse();
                    response = RestHttpResponse(
                        statusCode: 405,
                        contentType: failureEnvelope.contentType,
                        bodyBytes: failureEnvelope.bodyBytes,
                        additionalHeaderLines: ["Allow: \(allowedMethods.joined(separator: ", "))"]);
                case .notFound:
                    response = try RestEndpointFailure(
                        statusCode: 404,
                        message: "no route answers the requested path \(request.path)")
                        .envelopeResponse();
                }
            }
        } catch let endpointFailure as RestEndpointFailure {
            // A typed failure from the handler belongs to the handle
            // operation; anything earlier belongs to the parse operation.
            if endpointHandleStart != nil {
                RestHttpPerformanceAttribution.finishedOperation(
                    operationName: "rest_endpoint_handle",
                    operationStart: endpointHandleStart,
                    operationOutcome: "failed",
                    performanceAttributionEnabled: performanceAttributionEnabled);
            } else {
                RestHttpPerformanceAttribution.finishedOperation(
                    operationName: "rest_request_parse",
                    operationStart: parseStart,
                    operationOutcome: "rejected",
                    performanceAttributionEnabled: performanceAttributionEnabled);
            }
            do {
                response = try endpointFailure.envelopeResponse();
            } catch {
                response = RestHttpResponse.text(
                    statusCode: endpointFailure.statusCode,
                    body: "the failure envelope could not be serialized");
            }
        } catch is RestConnectionError {
            if endpointHandleStart != nil {
                RestHttpPerformanceAttribution.finishedOperation(
                    operationName: "rest_endpoint_handle",
                    operationStart: endpointHandleStart,
                    operationOutcome: "disconnected",
                    performanceAttributionEnabled: performanceAttributionEnabled);
            } else {
                RestHttpPerformanceAttribution.finishedOperation(
                    operationName: "rest_request_parse",
                    operationStart: parseStart,
                    operationOutcome: "disconnected",
                    performanceAttributionEnabled: performanceAttributionEnabled);
            }
            // No answer is possible; the client is gone or stalled.
            return;
        } catch {
            if endpointHandleStart != nil {
                RestHttpPerformanceAttribution.finishedOperation(
                    operationName: "rest_endpoint_handle",
                    operationStart: endpointHandleStart,
                    operationOutcome: "failed",
                    performanceAttributionEnabled: performanceAttributionEnabled);
            } else {
                RestHttpPerformanceAttribution.finishedOperation(
                    operationName: "rest_request_parse",
                    operationStart: parseStart,
                    operationOutcome: "failed",
                    performanceAttributionEnabled: performanceAttributionEnabled);
            }
            do {
                response = try RestEndpointFailure.internalError(message: "the endpoint handler failed")
                    .envelopeResponse();
            } catch {
                response = RestHttpResponse.text(statusCode: 500, body: "internal server error");
            }
        }

        guard var answerResponse: RestHttpResponse = response else {
            return;
        }
        // Every answer to an allowed origin carries the echo, exactly the
        // way the Rust CorsLayer decorates the router's whole surface —
        // successes, failures, and unknown paths alike.
        if let corsPolicy: RestCorsPolicy = corsPolicy,
            let requestOrigin: String = parsedRequestOrigin,
            corsPolicy.isOriginAllowed(requestOrigin) {
            answerResponse = answerResponse.withAdditionalHeaderLines(
                corsPolicy.answerHeaderLines(origin: requestOrigin));
        }
        let writeStart: ContinuousClock.Instant? = RestHttpPerformanceAttribution.startedOperation(
            operationName: "rest_response_write",
            performanceAttributionEnabled: performanceAttributionEnabled);
        do {
            try connection.writeBytes(answerResponse.serializedBytes());
            RestHttpPerformanceAttribution.finishedOperation(
                operationName: "rest_response_write",
                operationStart: writeStart,
                operationOutcome: "success",
                performanceAttributionEnabled: performanceAttributionEnabled);
        } catch {
            RestHttpPerformanceAttribution.finishedOperation(
                operationName: "rest_response_write",
                operationStart: writeStart,
                operationOutcome: "failed",
                performanceAttributionEnabled: performanceAttributionEnabled);
        }
    }
    /// A preflight is an OPTIONS request from an allowed origin naming the
    /// method it intends to use; the browser sends it before the real call.
    /// Anything else answers nil and takes the normal routing path.
    private static func corsPreflightAnswer(
        request: RestHttpRequest,
        corsPolicy: RestCorsPolicy?
    ) -> RestHttpResponse? {
        guard let corsPolicy: RestCorsPolicy = corsPolicy,
            request.method.uppercased() == "OPTIONS",
            let requestOrigin: String = request.headerValue(named: "Origin"),
            request.headerValue(named: "Access-Control-Request-Method") != nil,
            corsPolicy.isOriginAllowed(requestOrigin) else {
            return nil;
        }
        return RestHttpResponse(
            statusCode: 200,
            contentType: RestHttpResponse.textContentType,
            bodyBytes: Data(),
            additionalHeaderLines: corsPolicy.preflightHeaderLines(origin: requestOrigin));
    }
}

/// Why the REST listener could not start; the daemon answers these with a
/// startup failure and a stderr reason.
public enum RestHttpServerError: Error, CustomStringConvertible {

    case invalidBindHost(host: String);
    case socketOperationFailed(operation: String, detail: String);

    public var description: String {
        switch (self) {
        case let .invalidBindHost(host):
            return "the REST bind host \(host) is not an IPv4 loopback address";
        case let .socketOperationFailed(operation, detail):
            return "the REST listener \(operation) failed: \(detail)";
        }
    }
}
