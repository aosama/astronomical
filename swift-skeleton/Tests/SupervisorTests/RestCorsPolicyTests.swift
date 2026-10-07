import Foundation;

import Testing;

import AstronomicalConfig;
import JourneyCategories;

@testable import Supervisor;

/**
 * Acceptance journey for the CORS (Cross-Origin Resource Sharing) layer the
 * daemon's REST endpoint answers with, migrating the CorsLayer wiring from
 * apps/supervisor/src/application.rs: the Thin Talk canvas calls this API
 * from the thintalk-asset:// shell origin, so allowed origins read every
 * answer, preflights short-circuit before routing, and foreign origins get
 * nothing. The Rust tree had no CORS journey of its own; this suite is the
 * new contract proof.
 */
@Suite(.serialized, .tags(.hermeticJourney))
final class RestCorsPolicyTests {

    @Test
    func should_echo_the_canvas_shell_origin_on_every_answer() throws {
        let server: RestHttpServer = try RestCorsPolicyTests.startCorsServer();

        let responseText: String = try RestCorsPolicyTests.exchange(
            port: server.boundEndpoint.port,
            requestText: "GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\nOrigin: thintalk-asset://shell\r\n\r\n");

        #expect(responseText.hasPrefix("HTTP/1.1 200 OK"), "unexpected status line: \(responseText)");
        #expect(
            responseText.contains("Access-Control-Allow-Origin: thintalk-asset://shell"),
            "the shell origin must read the answer: \(responseText)");
        server.stop();
    }

    @Test
    func should_allow_loopback_origins_on_any_port() throws {
        let server: RestHttpServer = try RestCorsPolicyTests.startCorsServer();

        for loopbackOrigin: String in ["http://127.0.0.1:6733", "http://localhost:6732", "http://127.0.0.1"] {
            let responseText: String = try RestCorsPolicyTests.exchange(
                port: server.boundEndpoint.port,
                requestText: "GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\nOrigin: \(loopbackOrigin)\r\n\r\n");
            #expect(
                responseText.contains("Access-Control-Allow-Origin: \(loopbackOrigin)"),
                "the loopback origin \(loopbackOrigin) must read the answer: \(responseText)");
        }
        server.stop();
    }

    @Test
    func should_withhold_cross_origin_reads_from_foreign_origins() throws {
        let server: RestHttpServer = try RestCorsPolicyTests.startCorsServer();

        let responseText: String = try RestCorsPolicyTests.exchange(
            port: server.boundEndpoint.port,
            requestText: "GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\nOrigin: https://example.com\r\n\r\n");

        #expect(responseText.hasPrefix("HTTP/1.1 200 "), "the answer itself stays valid: \(responseText)");
        #expect(
            !responseText.contains("Access-Control"),
            "a foreign origin must never receive cross-origin read permission: \(responseText)");
        server.stop();
    }

    @Test
    func should_short_circuit_a_preflight_with_the_canvas_cors_contract() throws {
        let server: RestHttpServer = try RestCorsPolicyTests.startCorsServer();

        let responseText: String = try RestCorsPolicyTests.exchange(
            port: server.boundEndpoint.port,
            requestText: "OPTIONS /v1/chat/completions HTTP/1.1\r\n"
                + "Host: 127.0.0.1\r\n"
                + "Origin: thintalk-asset://shell\r\n"
                + "Access-Control-Request-Method: POST\r\n"
                + "Access-Control-Request-Headers: content-type\r\n"
                + "\r\n");

        #expect(responseText.hasPrefix("HTTP/1.1 200 OK"), "a preflight is answered, never routed: \(responseText)");
        #expect(
            responseText.contains("Access-Control-Allow-Origin: thintalk-asset://shell"),
            "the preflight names the allowed origin: \(responseText)");
        #expect(
            responseText.contains("Access-Control-Allow-Methods: GET, POST, DELETE, OPTIONS"),
            "the preflight names the canvas method set: \(responseText)");
        #expect(
            responseText.contains("Access-Control-Allow-Headers: content-type"),
            "the preflight names the content-type header: \(responseText)");
        #expect(
            responseText.contains("Access-Control-Max-Age: 3600"),
            "the preflight grants one hour of caching: \(responseText)");
        #expect(responseText.hasSuffix("\r\n\r\n"), "a preflight carries no body: \(responseText)");
        server.stop();
    }

    @Test
    func should_decorate_failures_and_unknown_paths_for_allowed_origins() throws {
        let server: RestHttpServer = try RestCorsPolicyTests.startCorsServer();

        let responseText: String = try RestCorsPolicyTests.exchange(
            port: server.boundEndpoint.port,
            requestText: "GET /v1/nonexistent HTTP/1.1\r\nHost: 127.0.0.1\r\nOrigin: thintalk-asset://shell\r\n\r\n");

        #expect(responseText.hasPrefix("HTTP/1.1 404 "), "unknown paths still answer 404: \(responseText)");
        #expect(
            responseText.contains("Access-Control-Allow-Origin: thintalk-asset://shell"),
            "the whole surface is decorated, failures included: \(responseText)");
        server.stop();
    }

    @Test
    func should_not_decorate_answers_to_requests_without_an_origin() throws {
        let server: RestHttpServer = try RestCorsPolicyTests.startCorsServer();

        let responseText: String = try RestCorsPolicyTests.exchange(
            port: server.boundEndpoint.port,
            requestText: "GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n");

        #expect(responseText.hasPrefix("HTTP/1.1 200 "), "the answer stays valid: \(responseText)");
        #expect(
            !responseText.contains("Access-Control"),
            "non-browser clients get no cross-origin decoration: \(responseText)");
        server.stop();
    }

    @Test
    func should_route_a_foreign_preflight_like_any_other_options_request() throws {
        let server: RestHttpServer = try RestCorsPolicyTests.startCorsServer();

        // A foreign origin's preflight is not the transport's to answer: it
        // routes like a plain OPTIONS and carries no cross-origin grant.
        let responseText: String = try RestCorsPolicyTests.exchange(
            port: server.boundEndpoint.port,
            requestText: "OPTIONS /health HTTP/1.1\r\n"
                + "Host: 127.0.0.1\r\n"
                + "Origin: https://example.com\r\n"
                + "Access-Control-Request-Method: GET\r\n"
                + "\r\n");

        #expect(!responseText.contains("Access-Control"), "no cross-origin grant for a foreign preflight: \(responseText)");
        server.stop();
    }

    private static func startCorsServer() throws -> RestHttpServer {
        return try RestHttpServer.start(
            bindEndpoint: SocketEndpoint.loopback(port: 0),
            routeTable: RestEndpointRoutes.foundationRouteTable(readinessProvider: {
                return WorkerHealthStatus.ready;
            }),
            corsPolicy: RestCorsPolicy.canvasShell());
    }

    private static func exchange(port: UInt16, requestText: String) throws -> String {
        guard let responseText: String = RawLoopbackHttpClient.exchange(
            port: port,
            requestText: requestText) else {
            throw RestCorsPolicyTestFailure.missingResponse;
        }
        return responseText;
    }
}

enum RestCorsPolicyTestFailure: Error {
    case missingResponse;
}
