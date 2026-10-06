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
    private var handlersByPathPrefix: Array<(pathPrefix: String, handlersByMethod: Dictionary<String, RestEndpointHandler>)>;

    public init() {
        self.handlersByPathThenMethod = Dictionary();
        self.handlersByPathPrefix = Array();
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

    /// Registers a handler for every path under the given prefix, for the
    /// routes that carry a variable tail (for example `/v1/models/{id}`).
    public mutating func registerPrefix(
        method: String,
        pathPrefix: String,
        handler: @escaping RestEndpointHandler
    ) -> Void {
        var prefixHandlers: Array<(pathPrefix: String, handlersByMethod: Dictionary<String, RestEndpointHandler>)> = self.handlersByPathPrefix;
        if let existingIndex: Array<(pathPrefix: String, handlersByMethod: Dictionary<String, RestEndpointHandler>)>.Index = prefixHandlers.firstIndex(where: { (entry: (pathPrefix: String, handlersByMethod: Dictionary<String, RestEndpointHandler>)) -> Bool in
            return entry.pathPrefix == pathPrefix;
        }) {
            var handlersByMethod: Dictionary<String, RestEndpointHandler> = prefixHandlers[existingIndex].handlersByMethod;
            handlersByMethod[method.uppercased()] = handler;
            prefixHandlers[existingIndex] = (pathPrefix, handlersByMethod);
        } else {
            prefixHandlers.append((pathPrefix, [method.uppercased(): handler]));
        }
        self.handlersByPathPrefix = prefixHandlers;
    }

    public func outcome(method: String, path: String) -> RestRouteOutcome {
        if let handlersByMethod: Dictionary<String, RestEndpointHandler> = self.handlersByPathThenMethod[path] {
            return self.methodOutcome(handlersByMethod: handlersByMethod, method: method);
        }
        // Longest registered prefix wins so nested prefixes stay exact.
        var bestMatch: (pathPrefix: String, handlersByMethod: Dictionary<String, RestEndpointHandler>)?;
        for prefixEntry: (pathPrefix: String, handlersByMethod: Dictionary<String, RestEndpointHandler>) in self.handlersByPathPrefix {
            guard path.hasPrefix(prefixEntry.pathPrefix) else {
                continue;
            }
            if bestMatch == nil || prefixEntry.pathPrefix.count > bestMatch!.pathPrefix.count {
                bestMatch = prefixEntry;
            }
        }
        if let matchedPrefix: (pathPrefix: String, handlersByMethod: Dictionary<String, RestEndpointHandler>) = bestMatch {
            return self.methodOutcome(handlersByMethod: matchedPrefix.handlersByMethod, method: method);
        }
        return .notFound;
    }

    private func methodOutcome(
        handlersByMethod: Dictionary<String, RestEndpointHandler>,
        method: String
    ) -> RestRouteOutcome {
        guard let matchedHandler: RestEndpointHandler = handlersByMethod[method.uppercased()] else {
            return .methodNotAllowed(allowedMethods: handlersByMethod.keys.sorted());
        }
        return .handler(matchedHandler);
    }
}
