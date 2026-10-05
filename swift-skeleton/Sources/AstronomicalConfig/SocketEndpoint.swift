import Foundation;

/**
 * Listener endpoint for one Astronomical instance, the Swift stand-in for the
 * Rust `SocketAddr` loopback values the instance boundary passes around.
 */
public struct SocketEndpoint: Equatable, CustomStringConvertible, Sendable {
    public let host: String;
    public let port: UInt16;

    public init(host: String, port: UInt16) {
        self.host = host;
        self.port = port;
    }

    public static func loopback(port: UInt16) -> SocketEndpoint {
        return SocketEndpoint(host: "127.0.0.1", port: port);
    }

    public var description: String {
        return "\(self.host):\(self.port)";
    }
}
