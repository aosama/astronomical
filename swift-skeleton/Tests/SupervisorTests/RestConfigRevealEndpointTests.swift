import Foundation;

import Testing;

import AstronomicalConfig;
import JourneyCategories;

@testable import Supervisor;

/**
 * Acceptance journey for POST /v1/config/reveal, migrating
 * apps/supervisor/tests/rest_api/config_reveal_endpoint.rs: the route
 * reveals the active configuration in Finder through a bounded opener —
 * 204 when the reveal succeeds, 500 when it fails, and 404 when no
 * configuration resolver exists. The journeys inject the opener so nothing
 * ever opens a real Finder window during tests.
 */
@Suite(.tags(.hermeticJourney))
final class RestConfigRevealEndpointTests {

    @Test
    func should_answer_not_found_when_no_configuration_reveal_action_exists() throws {
        let response: RestHttpResponse = RestConfigRevealEndpoint.handle(
            RestConfigRevealEndpointTests.revealRequest(),
            revealContext: nil);

        #expect(response.statusCode == 404, "a missing resolver advertises an absent route, got \(response.statusCode)");
        #expect(
            String(decoding: response.bodyBytes, as: UTF8.self).contains("resolver"),
            "the failure names what is missing");
    }

    @Test
    func should_answer_no_content_when_the_active_config_is_revealed() throws {
        let response: RestHttpResponse = RestConfigRevealEndpoint.handle(
            RestConfigRevealEndpointTests.revealRequest(),
            revealContext: RestConfigRevealRouteContext(revealActiveConfig: {
                return true;
            }));

        #expect(response.statusCode == 204, "a successful reveal answers without content, got \(response.statusCode)");
        #expect(response.bodyBytes.isEmpty, "the 204 answer carries no body");
        #expect(response.reasonPhrase() == "No Content");
    }

    @Test
    func should_answer_server_error_when_the_reveal_fails() throws {
        let response: RestHttpResponse = RestConfigRevealEndpoint.handle(
            RestConfigRevealEndpointTests.revealRequest(),
            revealContext: RestConfigRevealRouteContext(revealActiveConfig: {
                return false;
            }));

        #expect(response.statusCode == 500, "a failed reveal answers 500, got \(response.statusCode)");
    }

    @Test
    func should_serve_the_reveal_route_over_the_loopback_transport() throws {
        var routeTable: RestRouteTable = RestRouteTable();
        routeTable.register(
            method: RestConfigRevealEndpoint.routeMethod,
            path: RestConfigRevealEndpoint.routePath,
            handler: { (request: RestHttpRequest) -> RestHttpResponse in
                return RestConfigRevealEndpoint.handle(request, revealContext: RestConfigRevealRouteContext(revealActiveConfig: {
                    return true;
                }));
            });
        let server: RestHttpServer = try RestHttpServer.start(
            bindEndpoint: SocketEndpoint.loopback(port: 0),
            routeTable: routeTable);

        let responseText: String = try RestConfigRevealEndpointTests.exchange(
            port: server.boundEndpoint.port,
            requestText: "POST /v1/config/reveal HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Length: 0\r\n\r\n");

        #expect(responseText.hasPrefix("HTTP/1.1 204 No Content"), "the wire answer is a 204: \(responseText)");
        #expect(responseText.contains("Content-Length: 0"), "the 204 answer is empty: \(responseText)");
        server.stop();
    }

    private static func revealRequest() -> RestHttpRequest {
        return RestHttpRequest(
            method: "POST",
            path: RestConfigRevealEndpoint.routePath,
            requestTarget: RestConfigRevealEndpoint.routePath,
            headersByLowercasedName: Dictionary(),
            bodyBytes: Data());
    }

    private static func exchange(port: UInt16, requestText: String) throws -> String {
        guard let responseText: String = RawLoopbackHttpClient.exchange(
            port: port,
            requestText: requestText) else {
            throw RestConfigRevealTestFailure.missingResponse;
        }
        return responseText;
    }
}

enum RestConfigRevealTestFailure: Error {
    case missingResponse;
}
