//! Owned decode-layer paging decision copied out of a retained-page lookup.
//!
//! # Why this type exists
//!
//! First-token decode looks at the retained-page cache to decide:
//!
//! - stream the whole layer from disk, or
//! - keep the already-resident experts and stream only the missing ones.
//!
//! That lookup happens under an immutable `RefCell` borrow
//! (`retained_expert_layers.borrow()`). Streaming a missing page then records a
//! disk load with `borrow_mut()`. Rust `RefCell` forbids those two borrows at
//! the same time. If the caller keeps the immutable borrow live while it
//! streams, the inference owner panics with `RefCell already borrowed`.
//!
//! This type is the copy that lets the immutable borrow die first. The caller
//! builds one owned value, drops the cache borrow, then streams or records.
//!
//! # What "split" means
//!
//! A demand-selected page does not always cover every expert this token routed
//! to. When it covers some but not all, decode can reuse the resident experts
//! and stream only the misses. When it covers none, or all, there is nothing to
//! split: stream the whole layer (all = the complete-hit path already returned
//! earlier; none = the page is useless for this route).

use super::{ExpertPageRoutePartition, QuantizedExpertPageManifest};

/// How one-token decode should execute one mixture-of-experts layer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PagedDecodeLayerDisposition {
    /// No useful retained page covers this route, so stream the whole layer.
    StreamEntireLayer,
    /// Keep retained experts on-device and stream only the route misses.
    SplitRetainedAndMissing(ExpertPageRoutePartition),
}

impl PagedDecodeLayerDisposition {
    /// Builds an owned decode plan from a retained-page lookup.
    ///
    /// `None` means this layer has no retained page, so stream everything.
    /// A present page that does not cover both retained and missing experts
    /// also streams the whole layer:
    ///
    /// - missing list empty: every routed expert is already resident; the
    ///   complete-hit path should have returned before this helper is used
    /// - retained list empty: the page owns none of this token's experts
    #[must_use]
    pub fn from_retained_page(
        retained_page_manifest: Option<&QuantizedExpertPageManifest>,
        selected_expert_ids: &[usize],
    ) -> Self {
        let Some(retained_page_manifest) = retained_page_manifest else {
            return Self::StreamEntireLayer;
        };
        let route_partition =
            retained_page_manifest.partition_route_assignments(selected_expert_ids);
        if route_partition.missing_expert_ids.is_empty()
            || route_partition.retained_expert_ids.is_empty()
        {
            return Self::StreamEntireLayer;
        }
        Self::SplitRetainedAndMissing(route_partition)
    }
}
