import Foundation;

/**
 * A human-readable discovery outcome that never embeds absolute paths, so a
 * report stays safe to surface in configuration errors and logs.
 */
public struct DiscoveryModelDiscoveryDiagnostic: Equatable, Sendable {
    public let code: DiscoveryModelDiscoveryDiagnosticCode;
    public let modelId: String;
    public let configuredRootNumbers: Array<Int>;

    private init(code: DiscoveryModelDiscoveryDiagnosticCode, modelId: String, configuredRootNumbers: Array<Int>) {
        self.code = code;
        self.modelId = modelId;
        self.configuredRootNumbers = configuredRootNumbers;
    }

    /** A model id appeared under more than one configured root; roots are 1-based. */
    public static func ambiguousModelIdentity(
        modelId: String,
        configuredRootNumbers: Array<Int>
    ) -> DiscoveryModelDiscoveryDiagnostic {
        return DiscoveryModelDiscoveryDiagnostic(
            code: DiscoveryModelDiscoveryDiagnosticCode.ambiguousModelIdentity,
            modelId: modelId,
            configuredRootNumbers: configuredRootNumbers
        );
    }

    /** A configured root could not be scanned; the number is 1-based. */
    public static func unavailableModelDirectory(configuredRootNumber: Int) -> DiscoveryModelDiscoveryDiagnostic {
        return DiscoveryModelDiscoveryDiagnostic(
            code: DiscoveryModelDiscoveryDiagnosticCode.unavailableModelDirectory,
            modelId: "",
            configuredRootNumbers: Array<Int>([configuredRootNumber])
        );
    }
}
