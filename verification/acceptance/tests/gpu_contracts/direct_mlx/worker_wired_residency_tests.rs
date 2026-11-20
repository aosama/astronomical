//! Contracts for the wired-residency claim every direct-MLX test process
//! makes when it takes the shared GPU guard. They run in the direct-MLX
//! contract binary because the helper lives in this package's shared
//! GPU-contract module.

use runtime_integration::maximum_recommended_gpu_working_set_size_bytes;
use serial_test::serial;

use crate::common::worker_wired_residency::{
    WiredResidencyClaim, claim_wired_residency_as_the_worker_does, wired_residency_claim_from,
};

const RECOMMENDED_WORKING_SET_BYTES: usize = 40_200_896_512;

fn accept_wired_limit(_wired_limit_bytes: usize) -> Result<usize, String> {
    Ok(0)
}

fn reject_wired_limit(_wired_limit_bytes: usize) -> Result<usize, String> {
    Err("invalid_argument".to_owned())
}

fn fail_if_asked_to_set_a_wired_limit(_wired_limit_bytes: usize) -> Result<usize, String> {
    panic!("no wired limit may be set without a recommendation");
}

#[test]
fn should_claim_the_whole_recommended_working_set() {
    assert_eq!(
        wired_residency_claim_from(Ok(RECOMMENDED_WORKING_SET_BYTES), accept_wired_limit),
        WiredResidencyClaim::Granted {
            claimed_wired_limit_bytes: RECOMMENDED_WORKING_SET_BYTES
        }
    );
}

#[test]
fn should_stay_unprotected_when_mlx_rejects_the_wired_limit() {
    assert!(matches!(
        wired_residency_claim_from(Ok(RECOMMENDED_WORKING_SET_BYTES), reject_wired_limit),
        WiredResidencyClaim::Unprotected { .. }
    ));
}

#[test]
fn should_stay_unprotected_without_setting_a_limit_when_the_recommendation_is_unreadable() {
    assert!(matches!(
        wired_residency_claim_from(
            Err("no Metal device".to_owned()),
            fail_if_asked_to_set_a_wired_limit
        ),
        WiredResidencyClaim::Unprotected { .. }
    ));
}

#[test]
fn should_stay_unprotected_without_setting_a_limit_when_the_recommendation_is_zero() {
    assert!(matches!(
        wired_residency_claim_from(Ok(0), fail_if_asked_to_set_a_wired_limit),
        WiredResidencyClaim::Unprotected { .. }
    ));
}

#[tokio::test]
#[serial]
async fn should_hold_the_workers_wired_limit_once_the_gpu_guard_is_taken() {
    let _gpu_exclusive = crate::common::direct_mlx_test_guard().await;
    let recommended_working_set_bytes = maximum_recommended_gpu_working_set_size_bytes()
        .expect("Metal should report its recommended working set");
    assert_eq!(
        claim_wired_residency_as_the_worker_does(),
        &WiredResidencyClaim::Granted {
            claimed_wired_limit_bytes: recommended_working_set_bytes
        }
    );
    // Setting the same limit again reports what was in force: the claim the
    // guard made, not MLX's unclaimed default of zero.
    let wired_limit_in_force_bytes = mlx_bridge::set_wired_limit(recommended_working_set_bytes)
        .expect("MLX should accept the recommended working set again");
    assert_eq!(wired_limit_in_force_bytes, recommended_working_set_bytes);
}
