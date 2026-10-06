import Foundation;

/** User-visible runtime identity that keeps Stable and Development state apart. */
public enum AstronomicalRuntimeInstance: String, Equatable {
    case stable = "stable"
    case development = "development"

    private static let STABLE_LOOPBACK_PORT: UInt16 = 6732;
    private static let DEVELOPMENT_LOOPBACK_PORT: UInt16 = 6733;

    /** The wire/config name of the instance ("stable" or "development"). */
    public var rawInstanceName: String {
        return self.rawValue;
    }

    public var displayName: String {
        switch (self) {
        case .stable:
            return "Stable";
        case .development:
            return "Development";
        }
    }

    /** Loopback listener owned by this runtime instance. */
    public var loopbackEndpoint: SocketEndpoint {
        switch (self) {
        case .stable:
            return SocketEndpoint.loopback(port: AstronomicalRuntimeInstance.STABLE_LOOPBACK_PORT);
        case .development:
            return SocketEndpoint.loopback(port: AstronomicalRuntimeInstance.DEVELOPMENT_LOOPBACK_PORT);
        }
    }

    /**
     * - Parameters:
     *   - rawInstance: Raw channel name coming from configuration or the
     *     environment, exactly as the Rust `FromStr` implementation receives it.
     * - Throws: `AstronomicalConfigError.invalidRuntimeInstance` when the raw
     *   value names neither channel.
     */
    public init(rawInstance: String) throws {
        switch (rawInstance) {
        case AstronomicalRuntimeInstance.stable.rawValue:
            self = AstronomicalRuntimeInstance.stable;
        case AstronomicalRuntimeInstance.development.rawValue:
            self = AstronomicalRuntimeInstance.development;
        default:
            throw AstronomicalConfigError.invalidRuntimeInstance(rawInstance: rawInstance);
        }
    }
}
