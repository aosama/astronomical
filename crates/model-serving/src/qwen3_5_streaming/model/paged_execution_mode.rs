#[cfg(test)]
mod tests;

/// Selects the paged MoE execution path for target forwards, prefill and decode alike.
///
/// `ProductionDefault` is the only production variant: it governs multi-token
/// prefill chunks and single-token decode forwards, including the decode-time
/// warm-table use in `forward.rs`. The diagnostic variants remain explicit
/// test seams rather than runtime settings and force non-production prefill
/// paths only.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Qwen3_5MoEPagedExecutionMode {
    /// Uses adaptive direct layer pages for normal inference.
    ProductionDefault,
    /// Executes sparse MoE separately for each prompt token through the decode cache.
    TokenLocalDiagnostic,
    /// Forces one compact selected-expert page for all prompt tokens.
    CompactPromptDiagnostic,
}

impl Qwen3_5MoEPagedExecutionMode {
    /// Returns whether this forward may execute before exact routes are host-visible.
    ///
    /// One-token production decode benefits from validating its small hot route
    /// with the forward completion root. Multi-token prefill must resolve each
    /// layer first: a holey layer output would otherwise alter every downstream
    /// route while preserving exact per-layer execution.
    #[must_use]
    pub const fn should_defer_host_route_materialization(self, token_count: i32) -> bool {
        matches!(self, Self::ProductionDefault) && token_count == 1
    }
}
