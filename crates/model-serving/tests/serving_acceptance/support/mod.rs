use std::path::PathBuf;

pub(crate) mod exact_model_prompt;
pub(crate) mod mtp_support;
#[cfg(feature = "direct-mlx")]
pub(crate) mod performance_attribution;

pub(crate) const IMAGE_PAD_TOKEN_ID: u32 = 248_069;
pub(crate) const SAY_HI_PROMPT_TOKEN_IDS: [u32; 15] = [
    248_045, 846, 198, 44_240, 15_131, 13, 248_046, 198, 248_045, 74_455, 198, 248_068, 271,
    248_069, 271,
];
const MAXIMUM_SPECULATIVE_PREFILL_DRAFT_PAYLOAD_BYTES: u64 = 3_000_000_000;

#[test]
fn should_select_a_small_compatible_speculative_prefill_draft_model() {
    let selected_draft_model = select_smallest_compatible_speculative_prefill_draft_model(
        vec![
            (
                30_124_710_752,
                "Qwen3.6-35B-A3B-oQ6-mtp".to_owned(),
                PathBuf::from("oversized-qwen-draft"),
            ),
            (
                3_034_147_328,
                "Qwen3.5-4B-MLX-4bit".to_owned(),
                PathBuf::from("large-qwen-draft"),
            ),
            (
                1_722_149_056,
                "Qwen3.5-2B-4bit".to_owned(),
                PathBuf::from("two-billion-qwen-draft"),
            ),
            (
                650_168_512,
                "Qwen3.5-0.8B-OptiQ-4bit".to_owned(),
                PathBuf::from("small-qwen-draft"),
            ),
        ],
        MAXIMUM_SPECULATIVE_PREFILL_DRAFT_PAYLOAD_BYTES,
    );

    assert_eq!(
        selected_draft_model,
        Some((
            650_168_512,
            "Qwen3.5-0.8B-OptiQ-4bit".to_owned(),
            PathBuf::from("small-qwen-draft"),
        )),
    );
}

fn select_smallest_compatible_speculative_prefill_draft_model(
    draft_model_candidates: impl IntoIterator<Item = (u64, String, PathBuf)>,
    maximum_draft_payload_bytes: u64,
) -> Option<(u64, String, PathBuf)> {
    draft_model_candidates
        .into_iter()
        .filter(|draft_model_candidate| draft_model_candidate.0 <= maximum_draft_payload_bytes)
        .min_by(|left_candidate, right_candidate| {
            left_candidate
                .0
                .cmp(&right_candidate.0)
                .then_with(|| left_candidate.1.cmp(&right_candidate.1))
        })
}

pub(crate) fn large_sparse_moe_model_directory() -> PathBuf {
    crate::common::configured_installed_model_directory_by_id(
        crate::common::large_sparse_moe_model_id(),
    )
}

pub(crate) fn configured_depth_one_mtp_model_directory() -> PathBuf {
    crate::common::configured_installed_model_directory_by_id(
        crate::common::large_sparse_moe_model_id(),
    )
}

pub(crate) fn configured_resident_sparse_moe_model_directory() -> PathBuf {
    crate::common::configured_installed_model_directory_by_id(
        crate::common::resident_sparse_moe_model_id(),
    )
}

pub(crate) fn dense_mtp_model_directory() -> PathBuf {
    crate::common::configured_installed_model_directory_by_id(crate::common::dense_mtp_model_id())
}

#[cfg(feature = "direct-mlx")]
pub(crate) fn configured_speculative_prefill_draft_model(
    _target_model_directory: &std::path::Path,
) -> (PathBuf, String) {
    let draft_model_id = crate::common::small_dense_model_id().to_owned();
    let draft_model_directory =
        crate::common::configured_installed_model_directory_by_id(&draft_model_id);
    (draft_model_directory, draft_model_id)
}
