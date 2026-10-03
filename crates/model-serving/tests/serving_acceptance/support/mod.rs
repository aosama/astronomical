use std::path::PathBuf;

pub(crate) mod exact_model_prompt;
pub(crate) mod mtp_support;
#[cfg(feature = "direct-mlx")]
pub(crate) mod performance_attribution;
pub(crate) mod romeo_and_juliet;

pub(crate) const IMAGE_PAD_TOKEN_ID: u32 = 248_069;
pub(crate) const LOCAL_AI_PROMPT_TOKEN_IDS: [u32; 20] = [
    248_045, 846, 198, 657, 799, 14_542, 8_495, 314, 2_136, 14_791, 13, 248_046, 198, 248_045,
    74_455, 198, 248_068, 271, 248_069, 271,
];
pub(crate) const SAY_HI_PROMPT_TOKEN_IDS: [u32; 15] = [
    248_045, 846, 198, 44_240, 15_131, 13, 248_046, 198, 248_045, 74_455, 198, 248_068, 271,
    248_069, 271,
];
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
