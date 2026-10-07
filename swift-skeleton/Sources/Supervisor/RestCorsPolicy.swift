import Foundation;

/**
 * The CORS (Cross-Origin Resource Sharing) allowlist the REST endpoint
 * decorates its answers with, migrating the CorsLayer in
 * apps/supervisor/src/application.rs: the Thin Talk canvas runs inside a
 * WKWebView served from the thintalk-asset:// shell scheme and calls this
 * REST API directly with fetch, so the browser enforces CORS against that
 * exact origin. Only the shell origin, the loopback origins the page may
 * also target, and the methods and headers the client uses are allowed.
 *
 * The policy is a pure decision module: the transport asks it whether an
 * origin may read answers, asks for the preflight answer, and appends the
 * returned header lines; it never touches sockets.
 */
public struct RestCorsPolicy: Sendable {

    /// `thintalk-asset://shell` is a scheme-origin, so any port suffix is
    /// meaningless; the loopback scheme-origins allow any port.
    private let exactOrigins: Set<String>;
    private let portSuffixOrigins: Array<String>;

    public let allowedMethods: String;
    public let allowedHeaders: String;
    public let preflightCacheSeconds: Int;

    /// The canvas allowlist from the Rust application builder.
    public static func canvasShell() -> RestCorsPolicy {
        return RestCorsPolicy(
            exactOrigins: ["thintalk-asset://shell"],
            portSuffixOrigins: ["http://127.0.0.1", "http://localhost"]);
    }

    public init(
        exactOrigins: Set<String>,
        portSuffixOrigins: Array<String>,
        allowedMethods: String = "GET, POST, DELETE, OPTIONS",
        allowedHeaders: String = "content-type",
        preflightCacheSeconds: Int = 3600
    ) {
        self.exactOrigins = exactOrigins;
        self.portSuffixOrigins = portSuffixOrigins;
        self.allowedMethods = allowedMethods;
        self.allowedHeaders = allowedHeaders;
        self.preflightCacheSeconds = preflightCacheSeconds;
    }

    /// Whether one Origin header value may read answers from this endpoint.
    public func isOriginAllowed(_ origin: String) -> Bool {
        if self.exactOrigins.contains(origin) {
            return true;
        }
        return self.portSuffixOrigins.contains { (allowedOrigin: String) -> Bool in
            return origin == allowedOrigin || origin.hasPrefix("\(allowedOrigin):");
        };
    }

    /// The header lines every answer to an allowed origin carries; the
    /// allowed origin echoes back exactly the way the Rust predicate layer
    /// does, so any allowed loopback port keeps working.
    public func answerHeaderLines(origin: String) -> Array<String> {
        return ["Access-Control-Allow-Origin: \(origin)"];
    }

    /// The header lines a preflight answer carries, naming the methods, the
    /// request headers, and how long the browser may cache the answer.
    public func preflightHeaderLines(origin: String) -> Array<String> {
        return self.answerHeaderLines(origin: origin) + [
            "Access-Control-Allow-Methods: \(self.allowedMethods)",
            "Access-Control-Allow-Headers: \(self.allowedHeaders)",
            "Access-Control-Max-Age: \(self.preflightCacheSeconds)"
        ];
    }
}
