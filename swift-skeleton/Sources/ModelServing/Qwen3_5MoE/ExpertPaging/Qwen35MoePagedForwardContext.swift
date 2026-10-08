import Foundation;

import MLX;
import MLXLMCommon;
import MLXNN;

/**
 * The token id the engine attributes the current MoE forward to, so the
 * paged SwitchGLU can record route observations keyed by the input token
 * without the engine threading ids through upstream call frames.
 *
 * One instance serves one engine: the engine sets it before every model
 * forward — the selected token id at decode, the chunk's first token id
 * at prefill — and every decorated layer forward reads it.
 */
public final class Qwen35MoePagedForwardContext {

    /// The token id attributed to the forward currently executing.
    public var currentInputTokenId: UInt32;

    /// The first fault any paged layer forward recorded during the current
    /// generation. The upstream SwitchGLU signature cannot throw, so the
    /// paged primitive records here and the engine aborts the generation
    /// fail-closed after the forward returns.
    public private(set) var firstRecordedFault: (any Error)?;

    public init(currentInputTokenId: UInt32) {
        self.currentInputTokenId = currentInputTokenId;
        self.firstRecordedFault = nil;
    }

    /// Keeps the first fault; later faults during the same generation are
    /// symptoms of the same broken paging install and stay unsurfaced.
    public func recordForwardFault(_ fault: any Error) -> Void {
        guard self.firstRecordedFault == nil else {
            return;
        }
        self.firstRecordedFault = fault;
    }

    /// Returns and clears the recorded fault so one generation reports it
    /// exactly once.
    public func consumeRecordedForwardFault() -> (any Error)? {
        guard let recordedFault = self.firstRecordedFault else {
            return nil;
        }
        self.firstRecordedFault = nil;
        return recordedFault;
    }
}
