import Foundation;

/**
 * The HTTP/1.1 request grammar the REST transport accepts. The accepted
 * subset is deliberate and owned: request line, plain headers, and a
 * Content-Length body within the transport cap. Requests outside the subset
 * (chunked bodies, duplicate headers, unknown HTTP versions) are rejected
 * with the shared failure envelope instead of being approximated — no
 * loopback client (curl, the CLI, the canvas webview) sends them.
 */
public enum RestHttpRequestParser {

    private static let MAXIMUM_HEADER_BLOCK_BYTES: UInt = 32_768;
    private static let SUPPORTED_HTTP_VERSIONS: Array<String> = ["HTTP/1.0", "HTTP/1.1"];

    public static func parseRequest(
        connection: RestHttpConnection,
        maximumRequestBodyBytes: UInt
    ) throws -> RestHttpRequest {
        let headerBlockBytes: Data = try connection.readHeaderBlock(
            maximumHeaderBlockBytes: RestHttpRequestParser.MAXIMUM_HEADER_BLOCK_BYTES);
        guard let headerBlockText: String = String(data: headerBlockBytes, encoding: .utf8) else {
            throw RestEndpointFailure(statusCode: 400, message: "the request head is not valid UTF-8");
        }
        // Swift folds "\r\n" into one grapheme Character, so the head is
        // normalized to bare-LF lines before line splitting.
        let normalizedHeadText: String = headerBlockText.replacingOccurrences(of: "\r\n", with: "\n");
        var headLines: Array<Substring> = normalizedHeadText.split(separator: "\n", omittingEmptySubsequences: false);
        while let lastHeadLine: Substring = headLines.last, lastHeadLine.isEmpty {
            headLines.removeLast();
        }
        guard headLines.isEmpty == false else {
            throw RestEndpointFailure(statusCode: 400, message: "the request is empty");
        }

        let requestLineComponents: Array<Substring> = headLines[0].split(separator: " ", omittingEmptySubsequences: true);
        guard requestLineComponents.count == 3 else {
            throw RestEndpointFailure(statusCode: 400, message: "the request line could not be parsed");
        }
        let requestMethod: String = String(requestLineComponents[0]);
        let requestTarget: String = String(requestLineComponents[1]);
        let httpVersion: String = String(requestLineComponents[2]);
        if RestHttpRequestParser.SUPPORTED_HTTP_VERSIONS.contains(httpVersion) == false {
            throw RestEndpointFailure(
                statusCode: 505,
                message: "the HTTP version \(httpVersion) is not supported; use HTTP/1.1");
        }
        let requestPath: String = requestTarget.split(separator: "?", maxSplits: 1, omittingEmptySubsequences: false)[0].description;

        let headersByLowercasedName: Dictionary<String, String> = try RestHttpRequestParser.parseHeaders(
            headLines: Array(headLines.dropFirst()));

        if headersByLowercasedName["transfer-encoding"] != nil {
            throw RestEndpointFailure(
                statusCode: 400,
                message: "chunked request bodies are not accepted; send Content-Length");
        }
        var requestBodyBytes: Data = Data();
        if let contentLengthText: String = headersByLowercasedName["content-length"] {
            guard let contentLengthValue: UInt = UInt(contentLengthText) else {
                throw RestEndpointFailure(statusCode: 400, message: "the Content-Length header is not a valid byte count");
            }
            if contentLengthValue > maximumRequestBodyBytes {
                throw RestEndpointFailure(
                    statusCode: 413,
                    message: "the request body exceeds the allowed \(maximumRequestBodyBytes) bytes");
            }
            if contentLengthValue > 0 {
                let expectsInterimContinue: Bool = (headersByLowercasedName["expect"]?.lowercased() == "100-continue");
                if expectsInterimContinue {
                    try connection.writeInterimContinue();
                }
                requestBodyBytes = try connection.readBodyBytes(byteCount: contentLengthValue);
            }
        }

        return RestHttpRequest(
            method: requestMethod,
            path: requestPath,
            requestTarget: requestTarget,
            headersByLowercasedName: headersByLowercasedName,
            bodyBytes: requestBodyBytes);
    }

    private static func parseHeaders(headLines: Array<Substring>) throws -> Dictionary<String, String> {
        var headersByLowercasedName: Dictionary<String, String> = Dictionary();
        for headLine: Substring in headLines {
            guard let colonIndex: Substring.Index = headLine.firstIndex(of: ":") else {
                throw RestEndpointFailure(statusCode: 400, message: "a request header line could not be parsed");
            }
            let headerName: String = String(headLine[headLine.startIndex..<colonIndex]).trimmingCharacters(in: .whitespaces);
            let headerValue: String = String(headLine[headLine.index(after: colonIndex)...]).trimmingCharacters(in: .whitespaces);
            if headerName.isEmpty || headerValue.isEmpty {
                throw RestEndpointFailure(statusCode: 400, message: "a request header line could not be parsed");
            }
            let lowercasedName: String = headerName.lowercased();
            if headersByLowercasedName[lowercasedName] != nil {
                throw RestEndpointFailure(statusCode: 400, message: "duplicate request header \(headerName)");
            }
            headersByLowercasedName[lowercasedName] = headerValue;
        }
        return headersByLowercasedName;
    }
}
