import Foundation;

/** Aggregated discovery outcome: what each root yielded plus non-fatal diagnostics. */
internal struct DiscoveryModelDiscoveryReport: Equatable, Sendable {
    internal let directoryScans: Array<DiscoveryModelDiscoveryDirectoryScan>;
    internal let diagnostics: Array<DiscoveryModelDiscoveryDiagnostic>;

    internal init(
        directoryScans: Array<DiscoveryModelDiscoveryDirectoryScan>,
        diagnostics: Array<DiscoveryModelDiscoveryDiagnostic>
    ) {
        self.directoryScans = directoryScans;
        self.diagnostics = diagnostics;
    }
}
