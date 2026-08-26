//! Whether just-read experts may stay in the retained cache.

/// Prefill seats a complete layer after a mandatory read when the leftover
/// expert budget can hold it. Decode does not promote complete layers here.
#[must_use]
pub const fn should_commit_mandatory_complete_layer(
    route_token_count: i32,
    production_default_paging: bool,
    residency_target: Option<super::ExpertLayerResidencyTarget>,
) -> bool {
    production_default_paging
        && route_token_count > 1
        && matches!(
            residency_target,
            Some(super::ExpertLayerResidencyTarget::PromoteCompleteOnMandatoryRead)
        )
}

/// Keep routed experts after a mandatory read so the next chunk can hit them.
/// Overflow is handled by evicting the least-used retained page, not by
/// refusing to cache what this forward used.
#[must_use]
pub const fn should_commit_mandatory_routed_page(
    _route_token_count: i32,
    production_default_paging: bool,
    _residency_target: Option<super::ExpertLayerResidencyTarget>,
    _layer_has_no_retained_page: bool,
) -> bool {
    production_default_paging
}
