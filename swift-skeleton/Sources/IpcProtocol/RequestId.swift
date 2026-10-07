import Foundation;

/// Supervisor-local correlation identifier for one generation request.
public struct RequestId: Equatable, Hashable, Sendable {
    internal let rawRequestId: UInt64;

    /// Creates a request identifier from a supervisor-local monotonic value.
    public init(rawRequestId: UInt64) {
        self.rawRequestId = rawRequestId;
    }

    /// Returns the numeric correlation value used in diagnostics.
    public func value() -> UInt64 {
        return self.rawRequestId;
    }

    internal func wireValue() -> JsonWireValue {
        return .unsignedInteger(self.rawRequestId);
    }

    internal static func fromWireValue(_ wireValue: JsonWireValue) throws -> RequestId {
        return RequestId(rawRequestId: try JsonWireValue.extractUInt64(wireValue));
    }
}
