import Foundation

/// Domain failures raised while assembling materialized expert slices into
/// a layer's SwitchGLU parameters.
public enum Qwen35MoePagedExpertDecoratorError: Error {

    /// The SwitchGLU did not expose the parameter the materialized slices
    /// target, so installing them would silently skip expert weights.
    case missingLayerParameter(parameterName: String)

    /// The layer input and the routing indices disagree on the token count,
    /// which the upstream gather/sort path would silently read out of
    /// bounds instead of failing loudly.
    case mismatchedForwardShapes(tokenCount: Int, routingTokenCount: Int)
}
