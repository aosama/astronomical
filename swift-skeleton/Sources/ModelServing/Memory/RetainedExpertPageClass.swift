import Foundation

/// Explicit residency class of one retained expert page.
///
/// Reclamation and telemetry reason over this same vocabulary so partial
/// (elastic, routed) pages always yield before stable complete layers.
public enum RetainedExpertPageClass: Equatable, Sendable {

    /// A complete decoder layer's experts, pinned for the request's
    /// lifetime; released only by exact deficit, never opportunistically.
    case stableCompleteLayer

    /// Routed-expert pages retained opportunistically; first to yield when
    /// a later phase needs the budget back.
    case elasticRoutedExperts
}
