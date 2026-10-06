import Foundation;

/**
 * One parsed REST request: the method and path the router matches on, the
 * raw request target, and the header and body content the endpoint handler
 * consumes. Header names are matched case-insensitively.
 */
public struct RestHttpRequest {

    public let method: String;
    /// The path component of the request target with any query part removed.
    public let path: String;
    /// The raw request target exactly as the client sent it.
    public let requestTarget: String;
    public let bodyBytes: Data;

    private let headersByLowercasedName: Dictionary<String, String>;

    public init(
        method: String,
        path: String,
        requestTarget: String,
        headersByLowercasedName: Dictionary<String, String>,
        bodyBytes: Data
    ) {
        self.method = method;
        self.path = path;
        self.requestTarget = requestTarget;
        self.headersByLowercasedName = headersByLowercasedName;
        self.bodyBytes = bodyBytes;
    }

    /// The last header value sent under the given name, matched without case.
    public func headerValue(named headerName: String) -> String? {
        return self.headersByLowercasedName[headerName.lowercased()];
    }
}
