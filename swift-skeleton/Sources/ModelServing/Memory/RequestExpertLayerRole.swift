import Foundation

/// Role of one sparse layer for the rest of this request's prefill.
public enum RequestExpertLayerRole: Equatable, Hashable, Sendable {

    /// Keep or promote the complete layer until the request ends or shrinks.
    case pinnedComplete

    /// Read when a forward needs it. Never promote during this prefill.
    case streamed
}
