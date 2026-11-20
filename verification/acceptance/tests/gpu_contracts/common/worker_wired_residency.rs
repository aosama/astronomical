//! The serving worker's wired-residency claim, made once per test process.
//!
//! The worker claims up to Metal's recommended working set as MLX's wired limit at
//! startup, so macOS keeps the model's weights and the `astro_expert_cache`
//! wired instead of compressing them. A test process that skips the claim
//! measures something the worker never does: near a 37 GB limit on a 48 GB
//! Mac, about half the warm streaming speed runs fell from 62 to 0.2 tokens a
//! second while the macOS compressor grew to 26 GB and wired memory drained
//! to under 4 GB. With the claim, six of six runs held at 62.
//!
//! Like the worker's, a failed claim never stops a test: the outcome is
//! printed so a run's log says whether its numbers were taken wired.

use std::sync::OnceLock;

use runtime_integration::maximum_recommended_gpu_working_set_size_bytes;

/// What the one claim of this test process came to.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum WiredResidencyClaim {
    /// MLX may keep this many allocated bytes wired, as in the worker.
    Granted { claimed_wired_limit_bytes: usize },
    /// No claim is in force; allocations stay pageable and compressible.
    Unprotected { reason: String },
}

/// Claims wired residency the first time it is called and returns that same
/// outcome afterwards; the wired limit is process-wide and lasts until exit.
pub(crate) fn claim_wired_residency_as_the_worker_does() -> &'static WiredResidencyClaim {
    static PROCESS_WIRED_RESIDENCY_CLAIM: OnceLock<WiredResidencyClaim> = OnceLock::new();
    PROCESS_WIRED_RESIDENCY_CLAIM.get_or_init(claim_and_report_wired_residency)
}

fn claim_and_report_wired_residency() -> WiredResidencyClaim {
    let recommended_working_set_bytes = match maximum_recommended_gpu_working_set_size_bytes() {
        Ok(recommended_working_set_bytes) => Ok(recommended_working_set_bytes),
        Err(recommendation_error) => Err(recommendation_error.to_string()),
    };
    let wired_residency_claim =
        wired_residency_claim_from(recommended_working_set_bytes, set_mlx_wired_limit);
    eprintln!("[direct-mlx-test] wired_residency_claim={wired_residency_claim:?}");
    wired_residency_claim
}

fn set_mlx_wired_limit(wired_limit_bytes: usize) -> Result<usize, String> {
    match mlx_bridge::set_wired_limit(wired_limit_bytes) {
        Ok(previous_wired_limit_bytes) => Ok(previous_wired_limit_bytes),
        Err(claim_error) => Err(claim_error.to_string()),
    }
}

/// Decides the claim from Metal's recommendation and the platform's answer.
/// The recommendation is the most MLX accepts as a wired limit, and the
/// worker claims all of it on a Mac whose memory ceiling is not lower.
pub(crate) fn wired_residency_claim_from(
    recommended_working_set_bytes: Result<usize, String>,
    set_wired_limit: fn(usize) -> Result<usize, String>,
) -> WiredResidencyClaim {
    let claimed_wired_limit_bytes = match recommended_working_set_bytes {
        Ok(0) => {
            return WiredResidencyClaim::Unprotected {
                reason: "Metal recommends no working set".to_owned(),
            };
        }
        Ok(recommended_working_set_bytes) => recommended_working_set_bytes,
        Err(recommendation_error) => {
            return WiredResidencyClaim::Unprotected {
                reason: format!(
                    "the recommended working set is unreadable: {recommendation_error}"
                ),
            };
        }
    };
    match set_wired_limit(claimed_wired_limit_bytes) {
        Ok(_previous_wired_limit_bytes) => WiredResidencyClaim::Granted {
            claimed_wired_limit_bytes,
        },
        Err(claim_error) => WiredResidencyClaim::Unprotected {
            reason: format!("MLX rejected the wired limit: {claim_error}"),
        },
    }
}
