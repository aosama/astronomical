/// Selects the paged MoE execution path for multi-token target forwards.
///
/// Diagnostic variants remain explicit test seams rather than runtime settings.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Qwen3_5MoEPagedPrefillExecutionMode {
    /// Uses adaptive direct layer pages for normal inference.
    ProductionDefault,
    /// Uses retained decode expert pages for the two-token MTP verification window.
    ProductionDecodeVerification,
    /// Executes sparse MoE separately for each prompt token through the decode cache.
    TokenLocalDiagnostic,
    /// Forces one compact selected-expert page for all prompt tokens.
    CompactMultiTokenDiagnostic,
}
