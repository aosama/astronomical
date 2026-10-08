import Foundation

/// Sorted unique routed expert identifiers one sparse decoder layer
/// selected for one token, port of the Rust `LayerRoutedExpertIds`.
/// Quantized expert populations fit far below `UInt16.max`, so the compact
/// element keeps one token's complete route small enough for a
/// multi-thousand-observation resident history.
public typealias LayerRoutedExpertIds = [UInt16]

/// One decode token's routed expert selection across all decoder layers in
/// layer order, port of the Rust `ObservedExpertRoute`. A `nil` layer marks
/// one that routed nothing (dense feed-forward or an unobserved layer),
/// which is itself part of the training label.
public typealias ObservedExpertRoute = [LayerRoutedExpertIds?]

/// One labeled training example for the expert-route predictor, port of
/// the Rust `RouteObservationRecord`.
public struct RouteObservationRecord: Equatable, Sendable {

    /// The token identifier whose forward produced `tokenRoute`.
    public let inputTokenId: UInt32

    /// The immediately preceding observed decode token's route, or `nil`
    /// when this is the first observed token of its request.
    public let previousTokenRoute: ObservedExpertRoute?

    /// This token's true routed selection, the prediction label.
    public let tokenRoute: ObservedExpertRoute

    public init(
        inputTokenId: UInt32,
        previousTokenRoute: ObservedExpertRoute?,
        tokenRoute: ObservedExpertRoute
    ) {
        self.inputTokenId = inputTokenId
        self.previousTokenRoute = previousTokenRoute
        self.tokenRoute = tokenRoute
    }
}
