import Foundation;

/** Models discovered under one configured root directory. */
public struct DiscoveryModelDiscoveryDirectoryScan: Equatable, Sendable {
    public let path: FilePath;
    public var discoveredModels: Array<DiscoveryDiscoveredModel>;

    public init(path: FilePath, discoveredModels: Array<DiscoveryDiscoveredModel>) {
        self.path = path;
        self.discoveredModels = discoveredModels;
    }
}
