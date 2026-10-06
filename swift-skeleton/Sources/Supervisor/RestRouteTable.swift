import Foundation;

/// One endpoint handler: consumes a parsed request, answers one response.
public typealias RestEndpointHandler = @Sendable (RestHttpRequest) throws -> RestHttpResponse;

/// How the route table answers one method and path combination.
public enum RestRouteOutcome: Sendable {
    case handler(RestEndpointHandler);
    case methodNotAllowed(allowedMethods: Array<String>);
    case notFound;
}

/**
 * The exact method-and-path route table the REST endpoint serves. Routes are
 * registered by the endpoint slices; the transport asks for the outcome of
 * every request and answers unknown paths with 404 and known paths with a
 * wrong method with 405 naming the allowed methods.
 */
public struct RestRouteTable: Sendable {

    private var handlersByPathThenMethod: Dictionary<String, Dictionary<String, RestEndpointHandler>>;

    public init() {
        self.handlersByPathThenMethod = Dictionary();
    }

    public mutating func register(
        method: String,
        path: String,
        handler: @escaping RestEndpointHandler
    ) -> Void {
        let uppercasedMethod: String = method.uppercased();
        var handlersByMethod: Dictionary<String, RestEndpointHandler> = self.handlersByPathThenMethod[path] ?? Dictionary();
        handlersByMethod[uppercasedMethod] = handler;
        self.handlersByPathThenMethod[path] = handlersByMethod;
    }

    public func outcome(method: String, path: String) -> RestRouteOutcome {
        guard let handlersByMethod: Dictionary<String, RestEndpointHandler> = self.handlersByPathThenMethod[path] else {
            return .notFound;
        }
        guard let matchedHandler: RestEndpointHandler = handlersByMethod[method.uppercased()] else {
            return .methodNotAllowed(allowedMethods: handlersByMethod.keys.sorted());
        }
        return .handler(matchedHandler);
    }
}
