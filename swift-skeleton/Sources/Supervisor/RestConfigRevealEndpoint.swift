import Foundation;

/// The context the daemon hands the config-reveal route: the action that
/// reveals the active configuration file, or nil when no configuration
/// resolver exists and the route must answer that it is absent — the Swift
/// shape of the Rust resolver-presence gate.
public struct RestConfigRevealRouteContext: Sendable {

    public let revealActiveConfig: @Sendable () -> Bool;

    public init(revealActiveConfig: @escaping @Sendable () -> Bool) {
        self.revealActiveConfig = revealActiveConfig;
    }
}

/// Reveals the active configuration in Finder, migrating
/// apps/supervisor/src/config_reveal_endpoint.rs: POST /v1/config/reveal
/// answers 204 when the reveal succeeded, 500 when the opener failed or
/// timed out, and 404 when no configuration resolver exists.
public enum RestConfigRevealEndpoint {

    public static let routeMethod: String = "POST";
    public static let routePath: String = "/v1/config/reveal";

    public static func handle(
        _ request: RestHttpRequest,
        revealContext: RestConfigRevealRouteContext?
    ) -> RestHttpResponse {
        guard let revealContext = revealContext else {
            return RestHttpResponse.text(
                statusCode: 404,
                body: "no configuration resolver is available");
        }
        if revealContext.revealActiveConfig() {
            return RestHttpResponse(
                statusCode: 204,
                contentType: RestHttpResponse.textContentType,
                bodyBytes: Data());
        }
        return RestHttpResponse.text(
            statusCode: 500,
            body: "the configuration file could not be revealed");
    }
}
