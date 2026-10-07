import Foundation

/**
 * Bounded loopback GET for launch, porting http.rs. This layer reports
 * transport outcomes; launch maps them to one-line user errors so a running
 * instance is never described as missing.
 */
enum LaunchHttp {

    private static let maximumJsonBodyBytes: Int = 1_000_000;

    /// Why a loopback GET could not produce JSON.
    enum LoopbackRequestError: Error {

        case unreachable
        case rejected
        case oversized
        case malformed
    }

    /// GET JSON from one Astronomical loopback path with an explicit timeout.
    static func getLoopbackJson(
        host: String,
        port: UInt16,
        requestPath: String,
        timeoutSeconds: Double
    ) -> Result<Any, LoopbackRequestError> {
        let responseBody: [UInt8];
        switch (LaunchHttp.fetchHttpBody(host: host, port: port, requestPath: requestPath, timeoutSeconds: timeoutSeconds)) {
        case let .success(fetchedBody):
            responseBody = fetchedBody;
        case let .failure(requestError):
            return .failure(requestError);
        }
        do {
            let parsedJsonValue: Any = try JSONSerialization.jsonSerialization(with: Data(responseBody), options: [])
            return .success(parsedJsonValue);
        } catch {
            return .failure(.malformed);
        }
    }

    private static func fetchHttpBody(
        host: String,
        port: UInt16,
        requestPath: String,
        timeoutSeconds: Double
    ) -> Result<[UInt8], LoopbackRequestError> {
        let addressDescription: String = "\(host):\(port)";
        var clientAddress: sockaddr_in = sockaddr_in();
        clientAddress.sin_family = sa_family_t(AF_INET);
        clientAddress.sin_port = port.bigEndian;
        guard inet_pton(AF_INET, host, &clientAddress.sin_addr) == 1 else {
            return .failure(.unreachable);
        }
        let fileDescriptor: Int32 = socket(AF_INET, SOCK_STREAM, 0);
        guard fileDescriptor >= 0 else {
            return .failure(.unreachable);
        }
        defer {
            close(fileDescriptor);
        }
        let timeoutMicroseconds: Int = Int(max(1, timeoutSeconds * 1_000_000));
        var receiveTimeout: timeval = timeval(
            tv_sec: timeoutMicroseconds / 1_000_000,
            tv_usec: __darwin_suseconds_t(timeoutMicroseconds % 1_000_000)
        );
        var sendTimeout: timeval = receiveTimeout;
        _ = withUnsafePointer(to: &receiveTimeout) { (timeoutPointer: UnsafePointer<timeval>) -> Int32 in
            return setsockopt(fileDescriptor, SOL_SOCKET, SO_RCVTIMEO, timeoutPointer, socklen_t(MemoryLayout<timeval>.size));
        };
        _ = withUnsafePointer(to: &sendTimeout) { (timeoutPointer: UnsafePointer<timeval>) -> Int32 in
            return setsockopt(fileDescriptor, SOL_SOCKET, SO_SNDTIMEO, timeoutPointer, socklen_t(MemoryLayout<timeval>.size));
        };
        let connectOutcome: Int32 = withUnsafePointer(to: &clientAddress) { (addressPointer: UnsafePointer<sockaddr_in>) -> Int32 in
            return addressPointer.withMemoryRebound(to: sockaddr.self, capacity: 1) { (socketAddress: UnsafePointer<sockaddr>) -> Int32 in
                return connect(fileDescriptor, socketAddress, socklen_t(MemoryLayout<sockaddr_in>.size));
            }
        };
        guard connectOutcome == 0 else {
            return .failure(.unreachable);
        }
        let httpRequest: String = "GET \(requestPath) HTTP/1.1\r\nHost: \(addressDescription)\r\n"
            + "Accept: application/json\r\nConnection: close\r\n\r\n";
        guard LaunchHttp.writeAll(fileDescriptor, requestBytes: Array(httpRequest.utf8)) else {
            return .failure(.unreachable);
        }
        var responseBytes: [UInt8] = [];
        var readBuffer: [UInt8] = [UInt8](repeating: 0, count: 8192);
        var headerEnd: Int? = nil;
        var contentLength: Int? = nil;
        while (true) {
            if (responseBytes.count > LaunchHttp.maximumJsonBodyBytes + 8192) {
                return .failure(.oversized);
            }
            let bytesRead: Int = read(fileDescriptor, &readBuffer, readBuffer.count);
            if (bytesRead == 0) {
                break;
            }
            if (bytesRead < 0) {
                return .failure(.unreachable);
            }
            responseBytes.append(contentsOf: readBuffer[0..<bytesRead]);
            if (headerEnd == nil) {
                if let foundEnd: Int = LaunchHttp.findHeaderEnd(responseBytes) {
                    headerEnd = foundEnd;
                    guard let headerText: String = String(
                        bytes: responseBytes[0..<foundEnd],
                        encoding: .utf8
                    ) else {
                        return .failure(.malformed);
                    }
                    if headerText.lowercased().contains("transfer-encoding: chunked") {
                        // Astronomical JSON responses use Content-Length. Refuse
                        // chunked encoding rather than hanging on a stream.
                        return .failure(.rejected);
                    }
                    switch (LaunchHttp.parseContentLength(headerText)) {
                    case let .success(parsedLength):
                        contentLength = parsedLength;
                    case let .failure(requestError):
                        return .failure(requestError);
                    }
                }
            }
            if let end: Int = headerEnd, let bodyLength: Int = contentLength {
                let bodyStart: Int = end + 4;
                if (responseBytes.count - bodyStart >= bodyLength) {
                    break;
                }
            }
        }
        guard let resolvedHeaderEnd: Int = headerEnd else {
            return .failure(.malformed);
        }
        guard let headerText: String = String(
            bytes: responseBytes[0..<resolvedHeaderEnd],
            encoding: .utf8
        ) else {
            return .failure(.malformed);
        }
        let statusLine: String = headerText.split(separator: "\r\n", maxSplits: 1).first.map(String.init) ?? "";
        if !(statusLine.hasPrefix("HTTP/1.1 200") || statusLine.hasPrefix("HTTP/1.0 200")) {
            return .failure(.rejected);
        }
        let bodyStart: Int = resolvedHeaderEnd + 4;
        let bodyBytes: [UInt8] = Array(responseBytes.dropFirst(bodyStart));
        if (bodyBytes.count > LaunchHttp.maximumJsonBodyBytes) {
            return .failure(.oversized);
        }
        if let bodyLength: Int = contentLength {
            if (bodyBytes.count < bodyLength) {
                return .failure(.unreachable);
            }
            return .success(Array(bodyBytes[0..<bodyLength]));
        }
        return .success(bodyBytes);
    }

    private static func writeAll(_ fileDescriptor: Int32, requestBytes: [UInt8]) -> Bool {
        var writtenCount: Int = 0;
        while (writtenCount < requestBytes.count) {
            let writeOutcome: Int = requestBytes.withUnsafeBufferPointer { (bufferPointer: UnsafeBufferPointer<UInt8>) -> Int in
                return write(fileDescriptor, bufferPointer.baseAddress! + writtenCount, requestBytes.count - writtenCount);
            };
            if (writeOutcome <= 0) {
                return false;
            }
            writtenCount += writeOutcome;
        }
        return true;
    }

    private static func findHeaderEnd(_ responseBytes: [UInt8]) -> Int? {
        let terminator: [UInt8] = Array("\r\n\r\n".utf8);
        if (responseBytes.count < terminator.count) {
            return nil;
        }
        for candidateIndex: Int in 0...(responseBytes.count - terminator.count) {
            if (responseBytes[candidateIndex..<(candidateIndex + terminator.count)] == terminator[...]) {
                return candidateIndex;
            }
        }
        return nil;
    }

    private static func parseContentLength(_ headerText: String) -> Result<Int?, LoopbackRequestError> {
        for headerLine: Substring in headerText.split(separator: "\r\n").dropFirst() {
            guard let separatorIndex: Substring.Index = headerLine.firstIndex(of: ":") else {
                continue;
            }
            let headerName: Substring = headerLine[..<separatorIndex];
            if !(headerName.lowercased() == "content-length") {
                continue;
            }
            let headerValue: String = headerLine[headerLine.index(after: separatorIndex)...].trimmingCharacters(in: .whitespaces);
            guard let parsedLength: Int = Int(headerValue) else {
                return .failure(.malformed);
            }
            if (parsedLength > LaunchHttp.maximumJsonBodyBytes) {
                return .failure(.oversized);
            }
            return .success(parsedLength);
        }
        return .success(nil);
    }
}

extension JSONSerialization {

    /// Small shim so the launch HTTP layer reads like its caller.
    static func jsonSerialization(with data: Data, options: JSONSerialization.ReadingOptions) throws -> Any {
        return try JSONSerialization.jsonObject(with: data, options: options);
    }
}
