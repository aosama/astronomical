import Foundation;
import XCTest;

import AstronomicalConfig;

@testable import Supervisor;

/**
 * Acceptance journey for the REST transport foundation: an HTTP client on the
 * loopback speaks to the daemon's REST endpoint and receives correct answers
 * for the live routes, and structured shared-vocabulary failures for unknown
 * paths, wrong methods, malformed requests, and bodies beyond the transport
 * cap. The bound endpoint is discoverable, a misbehaving client never
 * disturbs the endpoint, and shutdown refuses further connections.
 */
final class RestHttpServerTests: XCTestCase {

    func testHealthJourneyAnswersOkOverLoopback() throws {
        let server: RestHttpServer = try self.startFoundationServer(readiness: .ready);

        let responseText: String? = RawLoopbackHttpClient.exchange(
            port: server.boundEndpoint.port,
            requestText: "GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n");

        let responseBody: String = try self.requireResponseText(responseText);
        XCTAssertTrue(responseBody.hasPrefix("HTTP/1.1 200 OK"), "unexpected status line: \(responseBody)");
        XCTAssertTrue(responseBody.contains("Connection: close"), "response must be a close-delimited reply");
        XCTAssertTrue(responseBody.contains("Content-Type: text/plain"), "health is a plain-text reply");
        XCTAssertTrue(responseBody.hasSuffix("\r\n\r\nok"), "health body must be exactly ok: \(responseBody)");

        let queryResponse: String? = RawLoopbackHttpClient.exchange(
            port: server.boundEndpoint.port,
            requestText: "GET /health?live=1 HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n");
        let queryBody: String = try self.requireResponseText(queryResponse);
        XCTAssertTrue(queryBody.hasPrefix("HTTP/1.1 200 "), "query parts must not change routing: \(queryBody)");
        server.stop();
    }

    func testOversizedHeaderBlockAnswersBadRequestAndTheEndpointKeepsServing() throws {
        let server: RestHttpServer = try self.startFoundationServer(readiness: .ready);

        let oversizedHeaderLine: String = String(repeating: "x", count: 40_000);
        let oversizedResponse: String? = RawLoopbackHttpClient.exchange(
            port: server.boundEndpoint.port,
            requestText: "GET /health HTTP/1.1\r\nX-Bulk: \(oversizedHeaderLine)\r\n\r\n");
        let oversizedBody: String = try self.requireResponseText(oversizedResponse);
        XCTAssertTrue(oversizedBody.hasPrefix("HTTP/1.1 400 "), "an oversized header block must be 400: \(oversizedBody)");

        let followUpResponse: String? = RawLoopbackHttpClient.exchange(
            port: server.boundEndpoint.port,
            requestText: "GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n");
        let followUpBody: String = try self.requireResponseText(followUpResponse);
        XCTAssertTrue(followUpBody.hasPrefix("HTTP/1.1 200 "), "endpoint must keep serving after oversized headers");
        server.stop();
    }

    func testReadinessFollowsTheWorkerHealthProvider() throws {
        let readinessBox: ReadinessBox = ReadinessBox(initialStatus: .loading);
        let server: RestHttpServer = try self.startFoundationServer(readinessBox: readinessBox);

        let loadingResponse: String? = RawLoopbackHttpClient.exchange(
            port: server.boundEndpoint.port,
            requestText: "GET /ready HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n");
        let loadingBody: String = try self.requireResponseText(loadingResponse);
        XCTAssertTrue(loadingBody.hasPrefix("HTTP/1.1 503 "), "not-ready must be 503: \(loadingBody)");
        XCTAssertTrue(loadingBody.hasSuffix("\r\n\r\nloading"), "readiness body names the status: \(loadingBody)");

        readinessBox.overwrite(status: .ready);
        let readyResponse: String? = RawLoopbackHttpClient.exchange(
            port: server.boundEndpoint.port,
            requestText: "GET /ready HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n");
        let readyBody: String = try self.requireResponseText(readyResponse);
        XCTAssertTrue(readyBody.hasPrefix("HTTP/1.1 200 "), "ready must be 200: \(readyBody)");
        XCTAssertTrue(readyBody.hasSuffix("\r\n\r\nready"), "readiness body names the status: \(readyBody)");
        server.stop();
    }

    func testUnknownPathAnswersNotFoundWithTheSharedErrorEnvelope() throws {
        let server: RestHttpServer = try self.startFoundationServer(readiness: .ready);

        let responseText: String? = RawLoopbackHttpClient.exchange(
            port: server.boundEndpoint.port,
            requestText: "GET /nope HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n");
        let responseBody: String = try self.requireResponseText(responseText);
        XCTAssertTrue(responseBody.hasPrefix("HTTP/1.1 404 "), "unknown path must be 404: \(responseBody)");
        XCTAssertTrue(responseBody.contains("Content-Type: application/json"), "failures are JSON envelopes");

        let envelope: [String: Any] = try self.decodeEnvelope(fromResponseBody: responseBody);
        let errorObject: [String: Any] = try self.requireErrorObject(envelope);
        XCTAssertEqual(errorObject["type"] as? String, "invalid_request_error");
        let messageText: String = try self.requireMessageText(errorObject);
        XCTAssertFalse(messageText.isEmpty, "the failure must say what was wrong");
        server.stop();
    }

    func testWrongMethodOnAKnownPathAnswersMethodNotAllowed() throws {
        let server: RestHttpServer = try self.startFoundationServer(readiness: .ready);

        let responseText: String? = RawLoopbackHttpClient.exchange(
            port: server.boundEndpoint.port,
            requestText: "POST /health HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Length: 2\r\n\r\n{}");
        let responseBody: String = try self.requireResponseText(responseText);
        XCTAssertTrue(responseBody.hasPrefix("HTTP/1.1 405 "), "wrong method must be 405: \(responseBody)");
        XCTAssertTrue(responseBody.contains("Allow:"), "405 must name the allowed methods");

        let envelope: [String: Any] = try self.decodeEnvelope(fromResponseBody: responseBody);
        let errorObject: [String: Any] = try self.requireErrorObject(envelope);
        XCTAssertEqual(errorObject["type"] as? String, "invalid_request_error");
        let messageText: String = try self.requireMessageText(errorObject);
        XCTAssertFalse(messageText.isEmpty, "the failure must say what was wrong");
        server.stop();
    }

    func testMalformedRequestAnswersBadRequestAndTheEndpointKeepsServing() throws {
        let server: RestHttpServer = try self.startFoundationServer(readiness: .ready);

        let malformedResponse: String? = RawLoopbackHttpClient.exchange(
            port: server.boundEndpoint.port,
            requestText: "BOGUS LINE\r\n\r\n");
        let malformedBody: String = try self.requireResponseText(malformedResponse);
        XCTAssertTrue(malformedBody.hasPrefix("HTTP/1.1 400 "), "garbage must be 400: \(malformedBody)");

        let followUpResponse: String? = RawLoopbackHttpClient.exchange(
            port: server.boundEndpoint.port,
            requestText: "GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n");
        let followUpBody: String = try self.requireResponseText(followUpResponse);
        XCTAssertTrue(followUpBody.hasPrefix("HTTP/1.1 200 "), "endpoint must keep serving after garbage");
        server.stop();
    }

    func testRequestBodyBeyondTheServerCapIsRejectedAndWithinItServed() throws {
        let readinessBox: ReadinessBox = ReadinessBox(initialStatus: .ready);
        var routeTable: RestRouteTable = RestEndpointRoutes.foundationRouteTable(readinessProvider: {
            return readinessBox.readStatus();
        });
        routeTable.register(
            method: "POST",
            path: "/test/echo",
            handler: { (_ request: RestHttpRequest) -> RestHttpResponse in
                return RestHttpResponse.text(statusCode: 200, body: "echo");
            });
        let server: RestHttpServer = try RestHttpServer.start(
            bindEndpoint: SocketEndpoint.loopback(port: 0),
            routeTable: routeTable,
            maximumRequestBodyBytes: 64);

        let withinCapResponse: String? = RawLoopbackHttpClient.exchange(
            port: server.boundEndpoint.port,
            requestText: "POST /test/echo HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Length: 10\r\n\r\n0123456789");
        let withinCapBody: String = try self.requireResponseText(withinCapResponse);
        XCTAssertTrue(withinCapBody.hasPrefix("HTTP/1.1 200 "), "body within the cap must reach the handler: \(withinCapBody)");

        let beyondCapResponse: String? = RawLoopbackHttpClient.exchange(
            port: server.boundEndpoint.port,
            requestText: "POST /test/echo HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Length: 65\r\n\r\n" + String(repeating: "x", count: 65));
        let beyondCapBody: String = try self.requireResponseText(beyondCapResponse);
        XCTAssertTrue(beyondCapBody.hasPrefix("HTTP/1.1 413 "), "declared body beyond the cap must be 413: \(beyondCapBody)");
        server.stop();
    }

    func testServerExposesTheActuallyBoundLoopbackEndpoint() throws {
        let server: RestHttpServer = try self.startFoundationServer(readiness: .ready);
        XCTAssertEqual(server.boundEndpoint.host, "127.0.0.1");
        XCTAssertNotEqual(server.boundEndpoint.port, 0, "an ephemeral bind must publish its actual port");
        server.stop();
    }

    func testStopEndsTheEndpoint() throws {
        let server: RestHttpServer = try self.startFoundationServer(readiness: .ready);
        let boundPort: UInt16 = server.boundEndpoint.port;
        server.stop();

        XCTAssertFalse(
            RawLoopbackHttpClient.canConnect(port: boundPort),
            "after stop() the endpoint must refuse connections");
        server.stop();
    }

    func testAbruptClientDisconnectDoesNotDisturbTheEndpoint() throws {
        let server: RestHttpServer = try self.startFoundationServer(readiness: .ready);

        RawLoopbackHttpClient.connectThenCloseWithoutSpeaking(port: server.boundEndpoint.port);
        let followUpResponse: String? = RawLoopbackHttpClient.exchange(
            port: server.boundEndpoint.port,
            requestText: "GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n");
        let followUpBody: String = try self.requireResponseText(followUpResponse);
        XCTAssertTrue(followUpBody.hasPrefix("HTTP/1.1 200 "), "endpoint must survive silent clients");
        server.stop();
    }

    // MARK: - Journey helpers

    private func startFoundationServer(readiness: WorkerHealthStatus) throws -> RestHttpServer {
        let readinessBox: ReadinessBox = ReadinessBox(initialStatus: readiness);
        return try self.startFoundationServer(readinessBox: readinessBox);
    }

    private func startFoundationServer(readinessBox: ReadinessBox) throws -> RestHttpServer {
        let routeTable: RestRouteTable = RestEndpointRoutes.foundationRouteTable(readinessProvider: {
            return readinessBox.readStatus();
        });
        return try RestHttpServer.start(
            bindEndpoint: SocketEndpoint.loopback(port: 0),
            routeTable: routeTable);
    }

    private func requireResponseText(_ responseText: String?) throws -> String {
        guard let unwrappedResponseText: String = responseText else {
            XCTFail("the loopback exchange must produce a response");
            throw RestHttpServerTestFailure.missingResponse;
        }
        return unwrappedResponseText;
    }

    private func decodeEnvelope(fromResponseBody responseText: String) throws -> [String: Any] {
        guard let bodyStart: String.Index = responseText.range(of: "\r\n\r\n")?.upperBound else {
            throw RestHttpServerTestFailure.missingResponseBody;
        }
        let bodyText: String = String(responseText[bodyStart...]);
        guard let envelopeObject: [String: Any] = try JSONSerialization.jsonObject(with: Data(bodyText.utf8)) as? [String: Any] else {
            throw RestHttpServerTestFailure.nonJsonEnvelope;
        }
        return envelopeObject;
    }

    private func requireErrorObject(_ envelope: [String: Any]) throws -> [String: Any] {
        guard let errorObject: [String: Any] = envelope["error"] as? [String: Any] else {
            throw RestHttpServerTestFailure.missingErrorObject;
        }
        return errorObject;
    }

    private func requireMessageText(_ errorObject: [String: Any]) throws -> String {
        guard let messageText: String = errorObject["message"] as? String else {
            throw RestHttpServerTestFailure.missingMessageText;
        }
        return messageText;
    }
}

/// Thread-safe readiness handoff between the test body and the serving thread.
private final class ReadinessBox: @unchecked Sendable {

    private let stateLock: NSLock;
    private var currentStatus: WorkerHealthStatus;

    init(initialStatus: WorkerHealthStatus) {
        self.stateLock = NSLock();
        self.currentStatus = initialStatus;
    }

    func readStatus() -> WorkerHealthStatus {
        self.stateLock.lock();
        defer { self.stateLock.unlock(); }
        return self.currentStatus;
    }

    func overwrite(status: WorkerHealthStatus) -> Void {
        self.stateLock.lock();
        self.currentStatus = status;
        self.stateLock.unlock();
    }
}

private enum RestHttpServerTestFailure: Error {
    case missingResponse;
    case missingResponseBody;
    case nonJsonEnvelope;
    case missingErrorObject;
    case missingMessageText;
}

/// Raw loopback HTTP client used by the journeys: speaks exactly the bytes a
/// command-line client would send and reads until the server closes, so the
/// tests exercise the wire instead of a Foundation URL stack.
private final class RawLoopbackHttpClient: @unchecked Sendable {

    private static let receiveTimeoutSeconds: Int = 5;

    static func exchange(port: UInt16, requestText: String) -> String? {
        let connectionDescriptor: Int32 = self.openConnectedSocket(port: port);
        guard connectionDescriptor >= 0 else {
            return nil;
        }
        defer { close(connectionDescriptor); }

        let requestData: Data = Data(requestText.utf8);
        requestData.withUnsafeBytes({ (rawBuffer: UnsafeRawBufferPointer) -> Void in
            if let basePointer: UnsafeRawPointer = rawBuffer.baseAddress {
                _ = send(connectionDescriptor, basePointer, rawBuffer.count, 0);
            }
        });

        var receivedData: Data = Data();
        var readBuffer: Array<UInt8> = Array(repeating: 0, count: 4096);
        while true {
            let bytesRead: Int = read(connectionDescriptor, &readBuffer, readBuffer.count);
            if bytesRead <= 0 {
                break;
            }
            receivedData.append(contentsOf: readBuffer[0..<bytesRead]);
            if receivedData.count > 1_048_576 {
                break;
            }
        }
        return String(data: receivedData, encoding: .utf8);
    }

    static func canConnect(port: UInt16) -> Bool {
        let connectionDescriptor: Int32 = self.openConnectedSocket(port: port);
        if connectionDescriptor >= 0 {
            close(connectionDescriptor);
            return true;
        }
        return false;
    }

    static func connectThenCloseWithoutSpeaking(port: UInt16) -> Void {
        let connectionDescriptor: Int32 = self.openConnectedSocket(port: port);
        if connectionDescriptor >= 0 {
            close(connectionDescriptor);
        }
    }

    private static func openConnectedSocket(port: UInt16) -> Int32 {
        let connectionDescriptor: Int32 = socket(AF_INET, SOCK_STREAM, 0);
        guard connectionDescriptor >= 0 else {
            return -1;
        }
        var noSigPipeFlag: Int32 = 1;
        _ = setsockopt(
            connectionDescriptor, SOL_SOCKET, SO_NOSIGPIPE,
            &noSigPipeFlag, socklen_t(MemoryLayout<Int32>.size));
        var receiveTimeout: timeval = timeval(
            tv_sec: self.receiveTimeoutSeconds, tv_usec: 0);
        _ = setsockopt(
            connectionDescriptor, SOL_SOCKET, SO_RCVTIMEO,
            &receiveTimeout, socklen_t(MemoryLayout<timeval>.size));

        var address: sockaddr_in = sockaddr_in();
        address.sin_len = UInt8(MemoryLayout<sockaddr_in>.size);
        address.sin_family = sa_family_t(AF_INET);
        address.sin_port = in_port_t(port).bigEndian;
        address.sin_addr = in_addr(s_addr: inet_addr("127.0.0.1"));
        let connectOutcome: Int32 = withUnsafePointer(to: &address, { (addressPointer: UnsafePointer<sockaddr_in>) -> Int32 in
            return connect(
                connectionDescriptor,
                UnsafePointer<sockaddr>(OpaquePointer(addressPointer)),
                socklen_t(MemoryLayout<sockaddr_in>.size));
        });
        if connectOutcome != 0 {
            close(connectionDescriptor);
            return -1;
        }
        return connectionDescriptor;
    }
}
