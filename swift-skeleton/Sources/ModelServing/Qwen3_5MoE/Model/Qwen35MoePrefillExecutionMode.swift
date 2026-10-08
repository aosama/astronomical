// Qwen35MoePrefillExecutionMode.swift — ModelServing.Qwen3_5MoE.Model
//
// Port of crates/model-serving/src/qwen3_5_moe/model/prefill_execution_mode.rs.

/// Selects the paged MoE execution path for multi-token target forwards.
///
/// Diagnostic variants remain explicit test seams rather than runtime settings.
public enum Qwen35MoePagedPrefillExecutionMode {

    /// Uses adaptive direct layer pages for normal inference.
    case productionDefault
    /// Executes sparse MoE separately for each prompt token through the decode cache.
    case tokenLocalDiagnostic
    /// Forces one compact selected-expert page for all prompt tokens.
    case compactPromptDiagnostic

    /// Returns whether this forward may execute before exact routes are host-visible.
    ///
    /// One-token production decode benefits from validating its small hot route
    /// with the forward completion root. Multi-token prefill must resolve each
    /// layer first: a holey layer output would otherwise alter every downstream
    /// route while preserving exact per-layer execution.
    public func shouldDeferHostRouteMaterialization(tokenCount: Int) -> Bool {
        guard case .productionDefault = self else {
            return false;
        }
        return tokenCount == 1;
    }
}
