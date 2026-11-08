//! Beside-source unit tests for the allocator-cache reclaim threshold policy.
//!
//! The threshold boundary (exactly-at-threshold counts as reclaimable) is the
//! contract the prefill cadence depends on; these tests pin it so a future
//! refactor of the comparison cannot quietly change the reclamation
//! behavior.

use crate::mlx_runtime_types::{
    ALLOCATOR_CACHE_RECLAIM_THRESHOLD_BYTES, allocator_cache_exceeds_reclaim_threshold,
};

#[test]
fn should_leave_small_allocator_cache_pooled_after_a_prefill_chunk() {
    assert!(!allocator_cache_exceeds_reclaim_threshold(
        ALLOCATOR_CACHE_RECLAIM_THRESHOLD_BYTES - 1,
        ALLOCATOR_CACHE_RECLAIM_THRESHOLD_BYTES,
    ));
}

#[test]
fn should_reclaim_allocator_cache_at_the_policy_threshold() {
    assert!(allocator_cache_exceeds_reclaim_threshold(
        ALLOCATOR_CACHE_RECLAIM_THRESHOLD_BYTES,
        ALLOCATOR_CACHE_RECLAIM_THRESHOLD_BYTES,
    ));
}

mod live_runtime_admission {
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use crate::mlx_runtime::memory_policy::{
        ProcessMemoryPolicy, admit_runtime_memory_limits, release_runtime_from,
    };
    use crate::{MlxMemoryLimits, MlxRuntimeError};

    // One counter per test that counts native installs, so parallel tests
    // never share one.
    static NATIVE_INSTALLS_FOR_IDENTICAL: AtomicUsize = AtomicUsize::new(0);

    fn accept_native_install(_memory_limits: MlxMemoryLimits) -> Result<(), MlxRuntimeError> {
        Ok(())
    }

    fn count_identical_install(_memory_limits: MlxMemoryLimits) -> Result<(), MlxRuntimeError> {
        NATIVE_INSTALLS_FOR_IDENTICAL.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    fn refuse_native_install(_memory_limits: MlxMemoryLimits) -> Result<(), MlxRuntimeError> {
        Err(MlxRuntimeError::InvalidMemoryLimits {
            description: "scripted native refusal",
        })
    }

    fn limits(active_memory_limit_gigabytes: usize) -> MlxMemoryLimits {
        let limit_bytes = active_memory_limit_gigabytes << 30;
        MlxMemoryLimits::new(limit_bytes, limit_bytes).expect("positive limits should be valid")
    }

    fn policy(
        installed_limits: Option<MlxMemoryLimits>,
        live_runtime_count: usize,
    ) -> Mutex<ProcessMemoryPolicy> {
        Mutex::new(ProcessMemoryPolicy {
            installed_limits,
            live_runtime_count,
        })
    }

    #[test]
    fn should_install_new_limits_once_every_runtime_is_gone() {
        // The 8 GB a dropped engine left behind gives way to the 32 GB the
        // next engine starts at (#1906).
        let process_policy = policy(Some(limits(8)), 0);
        admit_runtime_memory_limits(&process_policy, limits(32), accept_native_install)
            .expect("no live runtime depends on the old limits");
        assert_eq!(
            *process_policy.lock().unwrap(),
            ProcessMemoryPolicy {
                installed_limits: Some(limits(32)),
                live_runtime_count: 1,
            }
        );
    }

    #[test]
    fn should_refuse_different_limits_beside_a_live_runtime() {
        let process_policy = policy(Some(limits(8)), 1);
        assert!(matches!(
            admit_runtime_memory_limits(&process_policy, limits(32), accept_native_install),
            Err(MlxRuntimeError::RuntimeAlreadyConfigured { .. })
        ));
        assert_eq!(
            *process_policy.lock().unwrap(),
            ProcessMemoryPolicy {
                installed_limits: Some(limits(8)),
                live_runtime_count: 1,
            }
        );
    }

    #[test]
    fn should_share_identical_limits_without_a_second_native_install() {
        let process_policy = policy(Some(limits(16)), 1);
        admit_runtime_memory_limits(&process_policy, limits(16), count_identical_install)
            .expect("identical limits are shared");
        assert_eq!(process_policy.lock().unwrap().live_runtime_count, 2);
        assert_eq!(NATIVE_INSTALLS_FOR_IDENTICAL.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn should_leave_the_policy_untouched_when_the_native_install_fails() {
        let process_policy = policy(None, 0);
        assert!(
            admit_runtime_memory_limits(&process_policy, limits(16), refuse_native_install)
                .is_err()
        );
        assert_eq!(
            *process_policy.lock().unwrap(),
            ProcessMemoryPolicy::UNCONFIGURED
        );
    }

    #[test]
    fn should_count_releases_down_to_zero_and_never_below() {
        let process_policy = policy(Some(limits(16)), 2);
        release_runtime_from(&process_policy);
        assert_eq!(process_policy.lock().unwrap().live_runtime_count, 1);
        release_runtime_from(&process_policy);
        release_runtime_from(&process_policy);
        assert_eq!(
            process_policy.lock().unwrap().live_runtime_count,
            0,
            "a stray release must not wrap the count"
        );
        admit_runtime_memory_limits(&process_policy, limits(32), accept_native_install)
            .expect("after the last release new limits install");
    }
}
