//! The single portable request budget for the OpenAI-compatible surface.
//!
//! Every ceiling the local OpenAI-compatible endpoints enforce lives here so the
//! HTTP body cap, the per-image cap, the per-request inline-media cap, the
//! image-count cap, and the worker IPC frame cap cannot drift apart. The local
//! rejection ceilings mirror the strictest mainstream hosted figure for each
//! dimension, so a request Astronomical accepts should also be accepted by the
//! hosted APIs: a local run becomes an early fidelity check instead of hiding
//! an oversized request until it hits a distant hosted 413.
//!
//! # Ceiling values, sources, and retrieval date
//!
//! All vendor figures below were re-verified against the vendor documentation
//! pages on 2026-10-22.
//!
//! | Ceiling | Value | Mirrors | Source (retrieved 2026-10-22) |
//! | --- | --- | --- | --- |
//! | HTTP request body | [`MAX_OPENAI_REQUEST_BODY_BYTES`] | 32 MB standard-endpoint request size limit | Anthropic, <https://platform.claude.com/docs/en/build-with-claude/vision> ("request size limits (32 MB for standard endpoints)") |
//! | Per-image decoded payload | [`MAX_OPENAI_IMAGE_DECODED_BYTES`] | 7 MB maximum file size per file for inline data | Google Cloud Gemini Enterprise Agent Platform, <https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/capabilities/image-understanding>; stricter than Anthropic's 10 MB base64 per image via the Claude API |
//! | Total inline media per request | [`MAX_OPENAI_INLINE_MEDIA_BYTES`] | 20 MB total request size for inline data (text prompts, system instructions, and inline bytes combined) | Gemini API, <https://ai.google.dev/gemini-api/docs/image-understanding>; stricter than Anthropic's 32 MB and OpenAI's 512 MB total payload |
//! | Image parts per request | [`MAX_OPENAI_IMAGE_PARTS_PER_REQUEST`] | 100 images per request on the API for 200k-token-context models | Anthropic, <https://platform.claude.com/docs/en/build-with-claude/vision>; stricter than Gemini's 3,600 and OpenAI's 1,500 (developers.openai.com/api/docs/guides/images-vision) |
//!
//! Reference points that are intentionally not the local default: OpenAI allows
//! 512 MB total payload and 1,500 images per request
//! (developers.openai.com/api/docs/guides/images-vision), and hosted serverless
//! platforms that front many OpenAI-compatible routers allow only 4.5 MB
//! request bodies (Vercel, <https://vercel.com/docs/functions/limitations>,
//! returning 413 `FUNCTION_PAYLOAD_TOO_LARGE`). A 4.5 MB-profile envelope may
//! become an opt-in profile later; the API-document figures above are the
//! default because they are what the hosted model APIs themselves enforce.
//!
//! # Unit convention
//!
//! Vendor pages state these figures in "MB" without pinning the byte
//! interpretation. The wire-facing [`MAX_OPENAI_REQUEST_BODY_BYTES`] uses the
//! conservative decimal reading (32,000,000 bytes, matching the byte-bisection
//! methodology the issue used against Vercel), so a locally accepted body can
//! never exceed the documented Anthropic figure. The decoded-media ceilings use
//! the binary MiB reading exactly as the issue proposal states them
//! (7 MiB and 20 MiB); unlike the body cap, that reading is a few percent
//! *looser* than the vendor decimal figures, so a locally accepted media
//! payload sitting in that gap can still be rejected by the mirrored vendor.
//!
//! # Rejection ordering
//!
//! For one request the checks fire outermost-first, so the specific typed
//! OpenAI-shaped 413 always wins over the transport cap:
//!
//! 1. HTTP body wire size ([`MAX_OPENAI_REQUEST_BODY_BYTES`], enforced by the
//!    daemon body-budget middleware). The budget exceeds the base64 wire size
//!    of the largest accepted media envelope (20 MiB decoded is about
//!    26.7 MiB on the wire), so every contract-layer check below stays
//!    reachable.
//! 2. Image parts per request ([`MAX_OPENAI_IMAGE_PARTS_PER_REQUEST`],
//!    checked before any base64 decode work).
//! 3. Per-image decoded payload ([`MAX_OPENAI_IMAGE_DECODED_BYTES`], during
//!    data-URI decode).
//! 4. Total decoded inline media per request
//!    ([`MAX_OPENAI_INLINE_MEDIA_BYTES`], accumulated across all images).
//! 5. Worker IPC frame (the daemon aliases this body budget as
//!    `MAX_IPC_FRAME_BYTES`), a local backstop that one validated request
//!    always fits one frame.
//!
//! # Dimension rule not enforced locally
//!
//! Anthropic additionally applies a stricter per-image dimension limit (every
//! image no larger than 2000 px on either dimension) whenever a request carries
//! more than 20 images, inside the 8000x8000 px per-image maximum. The local
//! surface does not decode pixel dimensions, so this rule is documented here
//! rather than enforced: a locally accepted request with more than 20 images
//! may still be rejected by Anthropic on image dimensions alone.

/// Maximum HTTP request body wire size accepted by the OpenAI-compatible
/// endpoints, in bytes.
///
/// Mirrors the 32 MB request size limit Anthropic documents for standard
/// endpoints (retrieved 2026-10-22), which is the widest mainstream body
/// ceiling; the strictest body ceiling, Gemini API's 20 MB inline total, is
/// expressed on decoded media by [`MAX_OPENAI_INLINE_MEDIA_BYTES`] so the
/// contract layer can name the exceeded ceiling precisely. The body budget is
/// deliberately larger than the base64 wire size of a fully loaded media
/// envelope so the typed media rejections stay reachable ahead of it.
pub const MAX_OPENAI_REQUEST_BODY_BYTES: usize = 32_000_000;

/// Maximum decoded byte payload accepted for one image, in bytes.
///
/// Mirrors the 7 MB per-inline-file limit documented by Google Cloud Gemini
/// Enterprise Agent Platform (retrieved 2026-10-22), the strictest mainstream
/// single-image figure; Anthropic allows 10 MB base64 per image via the Claude
/// API.
pub const MAX_OPENAI_IMAGE_DECODED_BYTES: usize = 7 * 1024 * 1024;

/// Maximum total decoded inline media bytes accepted per request across all
/// images, in bytes.
///
/// Mirrors the 20 MB total inline request size documented by the Gemini API
/// (inline bytes plus text prompts plus system instructions combined,
/// retrieved 2026-10-22). Because the standard base64 alphabet inflates
/// payloads by 4/3, a request carrying the full 20 MiB of decoded media is
/// about 26.7 MiB on the wire, which still fits
/// [`MAX_OPENAI_REQUEST_BODY_BYTES`].
pub const MAX_OPENAI_INLINE_MEDIA_BYTES: usize = 20 * 1024 * 1024;

/// Maximum number of image parts accepted per request.
///
/// Mirrors the 100-images-per-request limit Anthropic documents on the API for
/// 200k-token-context models (retrieved 2026-10-22), the strictest mainstream
/// image-count figure. Requests above 20 images additionally trip Anthropic's
/// 2000 px per-image dimension rule, which the local surface documents but
/// does not enforce (see the module documentation).
pub const MAX_OPENAI_IMAGE_PARTS_PER_REQUEST: usize = 100;

/// Returns the on-wire byte count of the standard-base64 encoding of a payload.
pub fn base64_wire_byte_count(decoded_byte_count: usize) -> usize {
    decoded_byte_count.div_ceil(3).saturating_mul(4)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn should_inflate_a_decoded_payload_by_four_thirds_on_the_wire() {
        assert_eq!(base64_wire_byte_count(0), 0);
        assert_eq!(base64_wire_byte_count(1), 4);
        assert_eq!(base64_wire_byte_count(3), 4);
        assert_eq!(base64_wire_byte_count(4), 8);
    }

    #[test]
    fn should_keep_the_fully_loaded_media_envelope_inside_the_body_budget_on_the_wire() {
        let fully_loaded_wire_bytes = base64_wire_byte_count(MAX_OPENAI_INLINE_MEDIA_BYTES);
        assert!(
            fully_loaded_wire_bytes < MAX_OPENAI_REQUEST_BODY_BYTES,
            "the body budget must stay above the wire size of the largest accepted media envelope \
             ({fully_loaded_wire_bytes} wire bytes) or the typed media rejections become \
             unreachable behind the transport cap"
        );
    }

    #[test]
    fn should_keep_the_largest_accepted_image_wire_size_inside_the_body_budget() {
        let largest_image_wire_bytes = base64_wire_byte_count(MAX_OPENAI_IMAGE_DECODED_BYTES);
        assert!(
            largest_image_wire_bytes < MAX_OPENAI_REQUEST_BODY_BYTES,
            "the per-image guard must stay reachable: the base64 wire size of one \
             budget-sized image ({largest_image_wire_bytes} bytes) must fit the body budget"
        );
    }

    #[test]
    fn should_keep_the_decoded_media_budget_above_the_per_image_budget() {
        // The relation is between two constants, so it is asserted at
        // compile time for the whole crate: a budget change that breaks the
        // ordering fails this test target's build, not just this run.
        const _: () = assert!(
            MAX_OPENAI_INLINE_MEDIA_BYTES >= MAX_OPENAI_IMAGE_DECODED_BYTES,
            "the inline-media budget must admit at least one budget-sized image"
        );
    }
}
