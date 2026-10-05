import Foundation;

/** Models discovered under one configured root directory. */
internal struct DiscoveryModelDiscoveryDirectoryScan: Equatable, Sendable {
    internal let path: FilePath;
    internal let discoveredModels: Array<DiscoveryDiscoveredModel>;

    internal init(path: FilePath, discoveredModels: Array<DiscoveryDiscoveredModel>) {
        self.path = path;
        self.discoveredModels = discoveredModels;
    }
}
