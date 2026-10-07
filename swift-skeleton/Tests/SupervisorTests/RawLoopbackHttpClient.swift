import Foundation;

@testable import Supervisor;

/// Raw loopback HTTP client used by the journeys: speaks exactly the bytes a
/// command-line client would send and reads until the server closes, so the
/// tests exercise the wire instead of a Foundation URL stack.
final class RawLoopbackHttpClient: @unchecked Sendable {

    // Generous on purpose: under a fully parallel test stampede on a machine
    // that is also serving a live instance, the serving threads can be
    // starved for tens of seconds. The bound still catches a wedged server
    // well inside the 120-second journey cap.
    private static let receiveTimeoutSeconds: Int = 30;

    static func exchange(port: UInt16, requestText: String) -> String? {
        let connectionDescriptor: Int32 = self.openConnectedSocket(port: port);
        guard connectionDescriptor >= 0 else {
            return nil;
        }
        defer { close(connectionDescriptor); }

        let requestData: Data = Data(requestText.utf8);
        requestData.withUnsafeBytes({ (rawBuffer: UnsafeRawBufferPointer) -> Void in
            if let basePointer: UnsafeRawPointer = rawBuffer.baseAddress {
                _ = send(connectionDescriptor, basePointer, rawBuffer.count, 0);
            }
        });

        var receivedData: Data = Data();
        var readBuffer: Array<UInt8> = Array(repeating: 0, count: 4096);
        while true {
            let bytesRead: Int = read(connectionDescriptor, &readBuffer, readBuffer.count);
            if bytesRead <= 0 {
                break;
            }
            receivedData.append(contentsOf: readBuffer[0..<bytesRead]);
            if receivedData.count > 1_048_576 {
                break;
            }
        }
        return String(data: receivedData, encoding: .utf8);
    }

    static func canConnect(port: UInt16) -> Bool {
        let connectionDescriptor: Int32 = self.openConnectedSocket(port: port);
        if connectionDescriptor >= 0 {
            close(connectionDescriptor);
            return true;
        }
        return false;
    }

    static func connectThenCloseWithoutSpeaking(port: UInt16) -> Void {
        let connectionDescriptor: Int32 = self.openConnectedSocket(port: port);
        if connectionDescriptor >= 0 {
            close(connectionDescriptor);
        }
    }

    private static func openConnectedSocket(port: UInt16) -> Int32 {
        let connectionDescriptor: Int32 = socket(AF_INET, SOCK_STREAM, 0);
        guard connectionDescriptor >= 0 else {
            return -1;
        }
        var noSigPipeFlag: Int32 = 1;
        _ = setsockopt(
            connectionDescriptor, SOL_SOCKET, SO_NOSIGPIPE,
            &noSigPipeFlag, socklen_t(MemoryLayout<Int32>.size));
        var receiveTimeout: timeval = timeval(
            tv_sec: self.receiveTimeoutSeconds, tv_usec: 0);
        _ = setsockopt(
            connectionDescriptor, SOL_SOCKET, SO_RCVTIMEO,
            &receiveTimeout, socklen_t(MemoryLayout<timeval>.size));

        var address: sockaddr_in = sockaddr_in();
        address.sin_len = UInt8(MemoryLayout<sockaddr_in>.size);
        address.sin_family = sa_family_t(AF_INET);
        address.sin_port = in_port_t(port).bigEndian;
        address.sin_addr = in_addr(s_addr: inet_addr("127.0.0.1"));
        let connectOutcome: Int32 = withUnsafePointer(to: &address, { (addressPointer: UnsafePointer<sockaddr_in>) -> Int32 in
            return connect(
                connectionDescriptor,
                UnsafePointer<sockaddr>(OpaquePointer(addressPointer)),
                socklen_t(MemoryLayout<sockaddr_in>.size));
        });
        if connectOutcome != 0 {
            close(connectionDescriptor);
            return -1;
        }
        return connectionDescriptor;
    }
}
