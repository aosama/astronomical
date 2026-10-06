import Foundation;

/// Effective model discovery in the precedence the supervisor daemon serves.
///
/// Migrates crates/config/src/model_discovery/effective_models.rs: one
/// automatic Library destination (the instance state models directory) is
/// scanned first and wins identity collisions with the authored config
/// roots, because Library publication owns that destination. An absent
/// automatic root is simply skipped.
public enum EffectiveModelDiscovery {

    /// Executable models plus diagnostics in the precedence the supervisor serves.
    public struct Outcome: Equatable, Sendable {
        public let discoveredModels: Array<DiscoveryDiscoveredModel>;
        public let diagnostics: Array<DiscoveryModelDiscoveryDiagnostic>;

        public init(discoveredModels: Array<DiscoveryDiscoveredModel>, diagnostics: Array<DiscoveryModelDiscoveryDiagnostic>) {
            self.discoveredModels = discoveredModels;
            self.diagnostics = diagnostics;
        }
    }

    /// Discovers executable models across one automatic Library root plus
    /// the authored config roots. Authored ambiguity cannot hide a validated
    /// Library copy of the same public identity.
    public static func discover(
        automaticModelsDirectory: FilePath,
        configuredModelDirectories: Array<FilePath>
    ) throws -> Outcome {
        var isAutomaticDirectoryPresent: ObjCBool = ObjCBool(false);
        let metadataResult: Bool? = FileManager.default.fileExists(
            atPath: automaticModelsDirectory.string,
            isDirectory: &isAutomaticDirectoryPresent);
        guard let metadataResult: Bool = metadataResult else {
            // Metadata access failed before the optional automatic root
            // could be classified as present: treat it like the Rust
            // AutomaticModelDirectoryMetadata arm — fatal to resolution.
            throw DiscoveryDiscoveredModelError.readDirectory(
                directoryPath: automaticModelsDirectory,
                underlyingError: NSError(domain: NSCocoaErrorDomain, code: 4, userInfo: [
                    NSFilePathErrorKey: automaticModelsDirectory.string,
                ]));
        }
        guard metadataResult && isAutomaticDirectoryPresent.boolValue else {
            let discoveryReport: DiscoveryModelDiscoveryReport = try DiscoveryModels.discoverModelsExcludingAmbiguousIdentities(
                modelDirectories: configuredModelDirectories);
            return Outcome(
                discoveredModels: discoveryReport.directoryScans.flatMap { (directoryScan: DiscoveryModelDiscoveryDirectoryScan) -> Array<DiscoveryDiscoveredModel> in
                    return directoryScan.discoveredModels;
                },
                diagnostics: discoveryReport.diagnostics);
        }

        let automaticScans: Array<DiscoveryModelDiscoveryDirectoryScan> = try DiscoveryModels.discoverModels(
            modelDirectories: [automaticModelsDirectory]);
        var effectiveModels: Array<DiscoveryDiscoveredModel> = automaticScans.flatMap { (directoryScan: DiscoveryModelDiscoveryDirectoryScan) -> Array<DiscoveryDiscoveredModel> in
            return directoryScan.discoveredModels;
        };
        let automaticModelIds: Set<String> = Set<String>(effectiveModels.map { (discoveredModel: DiscoveryDiscoveredModel) -> String in
            return discoveredModel.modelId;
        });
        let configuredDiscoveryReport: DiscoveryModelDiscoveryReport = try DiscoveryModels.discoverModelsExcludingAmbiguousIdentities(
            modelDirectories: configuredModelDirectories);
        let configuredModels: Array<DiscoveryDiscoveredModel> = configuredDiscoveryReport.directoryScans
            .flatMap { (directoryScan: DiscoveryModelDiscoveryDirectoryScan) -> Array<DiscoveryDiscoveredModel> in
                return directoryScan.discoveredModels;
            }
            .filter { (discoveredModel: DiscoveryDiscoveredModel) -> Bool in
                return !automaticModelIds.contains(discoveredModel.modelId);
            };
        effectiveModels.append(contentsOf: configuredModels);
        let survivingDiagnostics: Array<DiscoveryModelDiscoveryDiagnostic> = configuredDiscoveryReport.diagnostics
            .filter { (diagnostic: DiscoveryModelDiscoveryDiagnostic) -> Bool in
                return !automaticModelIds.contains(diagnostic.modelId);
            };
        return Outcome(discoveredModels: effectiveModels, diagnostics: survivingDiagnostics);
    }
}
