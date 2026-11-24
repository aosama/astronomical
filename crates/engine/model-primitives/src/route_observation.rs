//! Compact routed-expert values shared by inference and residency-neutral attribution.

/// Sorted unique routed expert identifiers selected for one sparse decoder layer.
pub type LayerRoutedExpertIds = Vec<u16>;

/// One token's routed experts across decoder layers; `None` marks an unobserved layer.
pub type ObservedExpertRoute = Vec<Option<LayerRoutedExpertIds>>;
