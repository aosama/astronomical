import Foundation;

/** The Qwen4-Exp architecture fields discovery needs to describe a model. */
internal struct DiscoveryQwen4ExpConfigurationSummary: Equatable, Sendable {
    internal let decoderLayers: UInt32;
    internal let contextWindowTokens: UInt64;
    internal let routedExperts: UInt32;

    internal init(decoderLayers: UInt32, contextWindowTokens: UInt64, routedExperts: UInt32) {
        self.decoderLayers = decoderLayers;
        self.contextWindowTokens = contextWindowTokens;
        self.routedExperts = routedExperts;
    }
}
