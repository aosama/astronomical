import Foundation;

/** One model a discovery scan found on disk. */
public struct DiscoveryDiscoveredModel: Equatable, Sendable {
    public let modelId: String;
    public let providerModelId: String?;
    public let modelFamily: ModelFamily;
    public let revision: String;
    public let modelDirectory: FilePath;
    public let capabilities: DiscoveryModelCapabilities;
    public let license: ModelLicense?;
    public let modelSizeBytes: UInt64;

    public init(
        modelId: String,
        providerModelId: String?,
        modelFamily: ModelFamily,
        revision: String,
        modelDirectory: FilePath,
        capabilities: DiscoveryModelCapabilities,
        license: ModelLicense?,
        modelSizeBytes: UInt64
    ) {
        self.modelId = modelId;
        self.providerModelId = providerModelId;
        self.modelFamily = modelFamily;
        self.revision = revision;
        self.modelDirectory = modelDirectory;
        self.capabilities = capabilities;
        self.license = license;
        self.modelSizeBytes = modelSizeBytes;
    }
}
