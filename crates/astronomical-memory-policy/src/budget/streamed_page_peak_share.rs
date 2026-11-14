//! How much of one forward's MLX peak was expert weights read from storage.
//!
//! A forward that streams expert pages reads them, uses them and releases
//! them. Their bytes show in the forward's peak but not in the active memory
//! sampled afterwards, so the learning that follows must not mistake them for
//! activation workspace (issue #691).
//!
//! The first version subtracted every byte the forward streamed. That is
//! right for a forward that streams one page. A prefill chunk on a model
//! larger than memory streams every nonresident layer in turn: 36 GB over
//! one 4,096-token chunk of a 48-layer model, of which only a page or two is
//! alive at any instant. Subtracting the total drove the learned workspace to
//! zero, the reserve stayed at its static floor, and the next full chunk ran
//! out of memory. And the total came from a performance-attribution counter,
//! which reads zero when attribution is off, so with everyday settings
//! nothing was subtracted at all and each streamed page was reserved twice,
//! once as workspace and once by the forward's own page reservation
//! (issue #2218).
//!
//! What can be said for certain is this: the forward's admission already
//! reserves one page reservation for streamed weights, separately from the
//! learned workspace. So the peak may give back at most that reservation.
//! With `W` the learned workspace and `R` the reservation, admission then
//! projects `W + R`, which is exactly the observed peak when the forward
//! streamed at least `R`, and more than the peak when it streamed less. It
//! never projects less than what was observed, whatever was alive at the
//! peak, and it no longer counts the same page twice.

/// The promoted expert-page bytes one forward's peak owes to streaming:
/// everything that stayed resident, plus the released remainder up to the
/// forward's own page reservation.
///
/// Pages that stayed resident are in the active memory sampled after the
/// forward, so they are always expert bytes and never workspace. Pages that
/// were read and released again count only as far as the reservation the
/// forward's admission makes for them, for the reason in the module comment.
#[must_use]
pub const fn promoted_expert_page_bytes_within_forward_peak(
    streamed_expert_page_bytes: u64,
    retained_expert_payload_growth_bytes: u64,
    forward_page_reservation_bytes: u64,
) -> u64 {
    let retained_streamed_bytes =
        if streamed_expert_page_bytes < retained_expert_payload_growth_bytes {
            streamed_expert_page_bytes
        } else {
            retained_expert_payload_growth_bytes
        };
    let released_streamed_bytes =
        streamed_expert_page_bytes.saturating_sub(retained_expert_payload_growth_bytes);
    let released_bytes_within_peak = if released_streamed_bytes < forward_page_reservation_bytes {
        released_streamed_bytes
    } else {
        forward_page_reservation_bytes
    };
    retained_streamed_bytes.saturating_add(released_bytes_within_peak)
}

#[cfg(test)]
#[path = "streamed_page_peak_share_tests.rs"]
mod streamed_page_peak_share_tests;
