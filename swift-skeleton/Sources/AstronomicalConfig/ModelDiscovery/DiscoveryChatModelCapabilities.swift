import Foundation;

/** Chat-serving limits carried by a discovered model, in tokens. */
public struct DiscoveryChatModelCapabilities: Equatable, Sendable {
    public let contextWindowTokens: UInt32;
    public let maximumInputTokens: UInt32;
    public let maximumOutputTokens: UInt32;
    public let supportsVision: Bool;
    public let supportsReasoning: Bool;
    public let supportsToolCalls: Bool;

    public init(
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
