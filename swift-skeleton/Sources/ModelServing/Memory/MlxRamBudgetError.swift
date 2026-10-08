import Foundation

/// Invalid configuration for the RAM budget owner
/// (port of `ram_values.rs::MlxRamBudgetError`).
public enum MlxRamBudgetError: Error, Equatable, Sendable {

    /// MLX RAM budget requires a positive active-memory ceiling.
    case invalidCeiling
}
