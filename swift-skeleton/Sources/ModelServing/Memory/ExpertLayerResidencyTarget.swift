import Foundation

/**
 * Planned behavior for one decoder layer.
 *
 * One target per sparse decoder layer per plan; the execution family enacts
 * exactly the named target and invents no second disposition.
 */
public enum ExpertLayerResidencyTarget: Equatable, Hashable, Sendable {

    /// The layer's complete expert payload is already resident and stays.
    case preserveComplete

    /// The layer is not resident, but this operation must read it anyway:
    /// promote the complete payload to retained RAM rather than streaming it.
    case promoteCompleteOnMandatoryRead

    /// A partial page set is already retained and stays for this operation.
    case preservePartial

    /// A partial layer is admitted because this operation's routes demand
    /// it; the admission charges the admitted pages to the retained budget.
    case admitPartialOnMandatoryRouteRead

    /// The layer is streamed for this operation only and retained by nobody.
    case streamOperationLocal

    /// The layer's partial retention yields (fully or partly) to free budget.
    case releasePartial

    /// The layer's complete retention yields by exactly the computed
    /// deficit, leaving the rest of the layer retained.
    case releaseCompleteForExactDeficit
}
