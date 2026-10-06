import Foundation;

/** Aggregated discovery outcome: what each root yielded plus non-fatal diagnostics. */
public struct DiscoveryModelDiscoveryReport: Equatable, Sendable {
    public let directoryScans: Array<DiscoveryModelDiscoveryDirectoryScan>;
    public let diagnostics: Array<DiscoveryModelDiscoveryDiagnostic>;

    public init(
        directoryScans: Array<DiscoveryModelDiscoveryDirectoryScan>,
        diagnostics: Array<DiscoveryModelDiscoveryDiagnostic>
    ) {
        self.directoryScans = directoryScans;
        self.diagnostics = diagnostics;
    }
}
