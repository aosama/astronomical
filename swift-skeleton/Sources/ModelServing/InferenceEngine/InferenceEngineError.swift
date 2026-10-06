import Foundation;

import IpcProtocol;

/// Bounded engine failures the worker loop translates into wire reasons.
///
/// Mirrors crates/model-serving/src/inference_engine/error.rs. Native detail
/// travels in logs and stderr, never in the wire reason.
public enum InferenceEngineError: Error, Equatable {

    /// A different generation already owns the engine's single-flight capacity.
    case engineBusy;
    /// The request or engine call was rejected without mutating engine state.
    case invalidRequest(reason: String);
    /// The engine could not load its model payload.
    case modelLoad(reason: String);
    /// A fatal execution failure the worker reports before exiting.
    case fatalExecution(reason: String);

    /// The wire-safe reason string the failure events carry.
    public var publicFailureReason: String {
        switch self {
        case .engineBusy: return "the inference engine is busy with another generation";
        case let .invalidRequest(reason): return reason;
        case let .modelLoad(reason): return reason;
        case let .fatalExecution(reason): return reason;
        }
    }
}
