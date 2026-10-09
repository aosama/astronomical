import Foundation;

/// Structural bootstrap reserve shared by startup residency admission until
/// the serving runtime learns context-window memory from completed requests.
public enum MlxRamBudgetDefaults {

    /// One decimal SI gigabyte reserved for initial context and activations.
    public static let BOOTSTRAP_CONTEXT_WINDOW_RESERVE_BYTES: UInt64 = 1_000_000_000;
}
