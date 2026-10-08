import Foundation

/// One decision in the paged-expert reclaim loop that admits a forward.
///
/// A single reclaim pass can leave peak still over when MLX active bytes
/// fall by less than the retired payload. Decode must keep yielding
/// leftover experts until the same forward fits, or until a pass releases
/// nothing.
public enum PagedExpertReclamationStep: Equatable, Sendable {

    /// Stable and peak projections fit; the forward may proceed.
    case admit

    /// Release this many retained expert bytes, then re-project.
    case reclaim(targetBytes: Int)

    /// No reclamation can make the forward fit; reject the operation.
    case reject
}
