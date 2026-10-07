import Foundation;

import IpcProtocol;

/**
 * One REST response ready for the wire: status, content type, fixed-length
 * body, and the close-delimited framing the transport always answers with.
 * Every response carries `Connection: close` — the endpoint serves exactly
 * one request per connection, which keeps the transport free of keep-alive
 * state while staying valid HTTP/1.1 for every loopback client.
 */
public struct RestHttpResponse {

    public static let textContentType: String = "text/plain; charset=utf-8";
    public static let jsonContentType: String = "application/json";

    public let statusCode: Int;
    public let contentType: String;
    public let bodyBytes: Data;
    /// Extra header lines (already formatted `Name: value`) such as `Allow`.
    public let additionalHeaderLines: Array<String>;

    /// The same answer with more header lines appended, the seam the CORS
    /// decoration uses without rebuilding every endpoint response.
    public func withAdditionalHeaderLines(_ extraHeaderLines: Array<String>) -> RestHttpResponse {
        return RestHttpResponse(
            statusCode: self.statusCode,
            contentType: self.contentType,
            bodyBytes: self.bodyBytes,
            additionalHeaderLines: self.additionalHeaderLines + extraHeaderLines);
    }

    public init(
        statusCode: Int,
        contentType: String,
        bodyBytes: Data,
        additionalHeaderLines: Array<String> = Array()
    ) {
        self.statusCode = statusCode;
        self.contentType = contentType;
        self.bodyBytes = bodyBytes;
        self.additionalHeaderLines = additionalHeaderLines;
    }

    public static func text(statusCode: Int, body: String) -> RestHttpResponse {
        return RestHttpResponse(
            statusCode: statusCode,
            contentType: RestHttpResponse.textContentType,
            bodyBytes: Data(body.utf8));
    }

    /// Builds a JSON reply over the shared wire writer, the same way the
    /// RestContract types serialize everywhere else.
    public static func json(statusCode: Int, wireValue: JsonWireValue) throws -> RestHttpResponse {
        var jsonWireWriter: JsonWireWriter = JsonWireWriter();
        try jsonWireWriter.appendValue(wireValue);
        return RestHttpResponse(
            statusCode: statusCode,
            contentType: RestHttpResponse.jsonContentType,
            bodyBytes: jsonWireWriter.serializedUtf8Bytes);
    }

    /// The reason phrase for the response status, per the HTTP/1.1 registry.
    public func reasonPhrase() -> String {
        switch (self.statusCode) {
        case 200: return "OK";
        case 202: return "Accepted";
        case 204: return "No Content";
        case 400: return "Bad Request";
        case 404: return "Not Found";
        case 405: return "Method Not Allowed";
        case 413: return "Content Too Large";
        case 500: return "Internal Server Error";
        case 503: return "Service Unavailable";
        case 505: return "HTTP Version Not Supported";
        default: return "Unknown";
        }
    }

    /// The full response bytes: status line, headers, and body.
    public func serializedBytes() -> Data {
        var responseText: String = "HTTP/1.1 \(self.statusCode) \(self.reasonPhrase())\r\n";
        responseText += "Content-Type: \(self.contentType)\r\n";
        responseText += "Content-Length: \(self.bodyBytes.count)\r\n";
        responseText += "Connection: close\r\n";
        for additionalHeaderLine: String in self.additionalHeaderLines {
            responseText += "\(additionalHeaderLine)\r\n";
        }
        responseText += "\r\n";
        var responseData: Data = Data(responseText.utf8);
        responseData.append(self.bodyBytes);
        return responseData;
    }
}
