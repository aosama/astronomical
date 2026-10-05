import Foundation;

/** One model a discovery scan found on disk. */
internal struct DiscoveryDiscoveredModel: Equatable, Sendable {
    internal let modelId: String;
    internal let providerModelId: String?;
    internal let modelFamily: ModelFamily;
    internal let revision: String;
    internal let modelDirectory: FilePath;
    internal let capabilities: DiscoveryModelCapabilities;
    internal let license: ModelLicense?;
    internal let modelSizeBytes: UInt64;

    internal init(
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
