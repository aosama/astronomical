import Foundation;

import IpcProtocol;

/// Architecture-specific prepared input accepted by one inference engine.
///
/// The paired chat processor produces it after prompt preparation; the
/// engine consumes its concrete type through the runtime pairing the model
/// family factory guarantees (Rust binds the pair with an associated type;
/// Swift erases it at the worker loop seam).
public protocol PreparedInferenceRequest: AnyObject {

    /// The token count used for protocol progress reporting.
    var promptTokenCount: Int { get };
}
