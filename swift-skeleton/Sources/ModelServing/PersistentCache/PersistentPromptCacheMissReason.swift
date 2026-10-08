import Foundation;

/// The exact reason a prompt prefix could not be restored from the
/// persistent prompt cache, port of the Rust
/// `PersistentPromptCacheMissReason`.
public enum PersistentPromptCacheMissReason: Equatable, Sendable {

    /// The prompt cannot produce even one safely restorable block.
    case promptTooShortForPersistentPromptCache;

    /// The first model-derived prompt block does not match tracked sequence state.
    case rootSequenceStateBlockMissing;

    /// Matched KV blocks exist, but none has a usable recurrent-state snapshot.
    case boundaryStateSnapshotMissing;
}
