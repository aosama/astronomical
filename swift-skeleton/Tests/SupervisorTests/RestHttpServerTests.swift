import Foundation;

import Testing;

import AstronomicalConfig;
import JourneyCategories;

@testable import Supervisor;

/**
 * Acceptance journey for the REST transport foundation: an HTTP client on the
 * loopback speaks to the daemon's REST endpoint and receives correct answers
 * for the live routes, and structured shared-vocabulary failures for unknown
 * paths, wrong methods, malformed requests, and bodies beyond the transport
 * cap. The bound endpoint is discoverable, a misbehaving client never
 * disturbs the endpoint, and shutdown refuses further connections.
 */
@Suite(.serialized, .tags(.hermeticJourney))
final class RestHttpServerTests {

    @Test
    func should_answer_the_health_journey_with_ok_over_loopback() throws {
        let server: RestHttpServer = try self.startFoundationServer(readiness: .ready);

        let responseText: String? = RawLoopbackHttpClient.exchange(
            port: server.boundEndpoint.port,
            requestText: "GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n");

        let responseBody: String = try self.requireResponseText(responseText);
        #expect(responseBody.hasPrefix("HTTP/1.1 200 OK"), "unexpected status line: \(responseBody)");
        #expect(responseBody.contains("Connection: close"), "response must be a close-delimited reply");
        #expect(responseBody.contains("Content-Type: text/plain"), "health is a plain-text reply");
        #expect(responseBody.hasSuffix("\r\n\r\nok"), "health body must be exactly ok: \(responseBody)");

        let queryResponse: String? = RawLoopbackHttpClient.exchange(
            port: server.boundEndpoint.port,
            requestText: "GET /health?live=1 HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n");
        let queryBody: String = try self.requireResponseText(queryResponse);
        #expect(queryBody.hasPrefix("HTTP/1.1 200 "), "query parts must not change routing: \(queryBody)");
        server.stop();
    }

    @Test
    func should_answer_bad_request_for_an_oversized_header_block_and_keep_serving() throws {
        let server: RestHttpServer = try self.startFoundationServer(readiness: .ready);

        let oversizedHeaderLine: String = String(repeating: "x", count: 40_000);
        let oversizedResponse: String? = RawLoopbackHttpClient.exchange(
            port: server.boundEndpoint.port,
            requestText: "GET /health HTTP/1.1\r\nX-Bulk: \(oversizedHeaderLine)\r\n\r\n");
        let oversizedBody: String = try self.requireResponseText(oversizedResponse);
        #expect(oversizedBody.hasPrefix("HTTP/1.1 400 "), "an oversized header block must be 400: \(oversizedBody)");

        let followUpResponse: String? = RawLoopbackHttpClient.exchange(
            port: server.boundEndpoint.port,
            requestText: "GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n");
        let followUpBody: String = try self.requireResponseText(followUpResponse);
        #expect(followUpBody.hasPrefix("HTTP/1.1 200 "), "endpoint must keep serving after oversized headers");
        server.stop();
    }

    @Test
    func should_follow_the_worker_health_provider_for_readiness() throws {
        let readinessBox: ReadinessBox = ReadinessBox(initialStatus: .loading);
        let server: RestHttpServer = try self.startFoundationServer(readinessBox: readinessBox);

        let loadingResponse: String? = RawLoopbackHttpClient.exchange(
            port: server.boundEndpoint.port,
            requestText: "GET /ready HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n");
        let loadingBody: String = try self.requireResponseText(loadingResponse);
        #expect(loadingBody.hasPrefix("HTTP/1.1 503 "), "not-ready must be 503: \(loadingBody)");
        #expect(loadingBody.hasSuffix("\r\n\r\nloading"), "readiness body names the status: \(loadingBody)");

        readinessBox.overwrite(status: .ready);
        let readyResponse: String? = RawLoopbackHttpClient.exchange(
            port: server.boundEndpoint.port,
            requestText: "GET /ready HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n");
        let readyBody: String = try self.requireResponseText(readyResponse);
        #expect(readyBody.hasPrefix("HTTP/1.1 200 "), "ready must be 200: \(readyBody)");
        #expect(readyBody.hasSuffix("\r\n\r\nready"), "readiness body names the status: \(readyBody)");
        server.stop();
    }

    @Test
    func should_answer_not_found_with_the_shared_error_envelope_for_an_unknown_path() throws {
        let server: RestHttpServer = try self.startFoundationServer(readiness: .ready);

        let responseText: String? = RawLoopbackHttpClient.exchange(
            port: server.boundEndpoint.port,
            requestText: "GET /nope HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n");
        let responseBody: String = try self.requireResponseText(responseText);
        #expect(responseBody.hasPrefix("HTTP/1.1 404 "), "unknown path must be 404: \(responseBody)");
        #expect(responseBody.contains("Content-Type: application/json"), "failures are JSON envelopes");

        let envelope: [String: Any] = try self.decodeEnvelope(fromResponseBody: responseBody);
        let errorObject: [String: Any] = try self.requireErrorObject(envelope);
        #expect(errorObject["type"] as? String == "invalid_request_error");
        let messageText: String = try self.requireMessageText(errorObject);
        #expect(!messageText.isEmpty, "the failure must say what was wrong");
        server.stop();
    }

    @Test
    func should_answer_method_not_allowed_for_a_wrong_method_on_a_known_path() throws {
        let server: RestHttpServer = try self.startFoundationServer(readiness: .ready);

        let responseText: String? = RawLoopbackHttpClient.exchange(
            port: server.boundEndpoint.port,
            requestText: "POST /health HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Length: 2\r\n\r\n{}");
        let responseBody: String = try self.requireResponseText(responseText);
        #expect(responseBody.hasPrefix("HTTP/1.1 405 "), "wrong method must be 405: \(responseBody)");
        #expect(responseBody.contains("Allow:"), "405 must name the allowed methods");

        let envelope: [String: Any] = try self.decodeEnvelope(fromResponseBody: responseBody);
        let errorObject: [String: Any] = try self.requireErrorObject(envelope);
        #expect(errorObject["type"] as? String == "invalid_request_error");
        let messageText: String = try self.requireMessageText(errorObject);
        #expect(!messageText.isEmpty, "the failure must say what was wrong");
        server.stop();
    }

    @Test
    func should_answer_bad_request_for_a_malformed_request_and_keep_serving() throws {
        let server: RestHttpServer = try self.startFoundationServer(readiness: .ready);

        let malformedResponse: String? = RawLoopbackHttpClient.exchange(
            port: server.boundEndpoint.port,
            requestText: "BOGUS LINE\r\n\r\n");
        let malformedBody: String = try self.requireResponseText(malformedResponse);
        #expect(malformedBody.hasPrefix("HTTP/1.1 400 "), "garbage must be 400: \(malformedBody)");

        let followUpResponse: String? = RawLoopbackHttpClient.exchange(
            port: server.boundEndpoint.port,
            requestText: "GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n");
        let followUpBody: String = try self.requireResponseText(followUpResponse);
        #expect(followUpBody.hasPrefix("HTTP/1.1 200 "), "endpoint must keep serving after garbage");
        server.stop();
    }

    @Test
    func should_reject_a_request_body_beyond_the_server_cap_and_serve_within_it() throws {
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
        #expect(withinCapBody.hasPrefix("HTTP/1.1 200 "), "body within the cap must reach the handler: \(withinCapBody)");

        let beyondCapResponse: String? = RawLoopbackHttpClient.exchange(
            port: server.boundEndpoint.port,
            requestText: "POST /test/echo HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Length: 65\r\n\r\n" + String(repeating: "x", count: 65));
        let beyondCapBody: String = try self.requireResponseText(beyondCapResponse);
        #expect(beyondCapBody.hasPrefix("HTTP/1.1 413 "), "declared body beyond the cap must be 413: \(beyondCapBody)");
        server.stop();
    }

    @Test
    func should_expose_the_actually_bound_loopback_endpoint() throws {
        let server: RestHttpServer = try self.startFoundationServer(readiness: .ready);
        #expect(server.boundEndpoint.host == "127.0.0.1");
        #expect(server.boundEndpoint.port != 0, "an ephemeral bind must publish its actual port");
        server.stop();
    }

    @Test
    func should_end_the_endpoint_on_stop() throws {
        let server: RestHttpServer = try self.startFoundationServer(readiness: .ready);
        let boundPort: UInt16 = server.boundEndpoint.port;
        server.stop();

        #expect(
            !RawLoopbackHttpClient.canConnect(port: boundPort),
            "after stop() the endpoint must refuse connections");
        server.stop();
    }

    @Test
    func should_keep_the_endpoint_disturbed_by_nothing_when_a_client_disconnects_abruptly() throws {
        let server: RestHttpServer = try self.startFoundationServer(readiness: .ready);

        RawLoopbackHttpClient.connectThenCloseWithoutSpeaking(port: server.boundEndpoint.port);
        let followUpResponse: String? = RawLoopbackHttpClient.exchange(
            port: server.boundEndpoint.port,
            requestText: "GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n");
        let followUpBody: String = try self.requireResponseText(followUpResponse);
        #expect(followUpBody.hasPrefix("HTTP/1.1 200 "), "endpoint must survive silent clients");
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

/// Thread-safe readiness handoff between the journey body and the serving thread.
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
