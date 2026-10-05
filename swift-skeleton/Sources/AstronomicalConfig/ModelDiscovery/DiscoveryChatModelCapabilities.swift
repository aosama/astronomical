import Foundation;

/** Chat-serving limits carried by a discovered model, in tokens. */
internal struct DiscoveryChatModelCapabilities: Equatable, Sendable {
    internal let contextWindowTokens: UInt32;
    internal let maximumInputTokens: UInt32;
    internal let maximumOutputTokens: UInt32;
    internal let supportsVision: Bool;
    internal let supportsReasoning: Bool;
    internal let supportsToolCalls: Bool;

    internal init(
        contextWindowTokens: UInt32,
        maximumInputTokens: UInt32,
        maximumOutputTokens: UInt32,
        supportsVision: Bool,
        supportsReasoning: Bool,
        supportsToolCalls: Bool
    ) {
        self.contextWindowTokens = contextWindowTokens;
        self.maximumInputTokens = maximumInputTokens;
        self.maximumOutputTokens = maximumOutputTokens;
        self.supportsVision = supportsVision;
        self.supportsReasoning = supportsReasoning;
        self.supportsToolCalls = supportsToolCalls;
    }
}
