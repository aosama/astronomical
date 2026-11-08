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

mod idle_limit_reinstall {
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use crate::mlx_runtime::memory_policy::reinstall_memory_limits_record;
    use crate::{MlxMemoryLimits, MlxRuntimeError};

    // Each test owns a distinct counter, so parallel tests never share one.
    static INSTALL_COUNT_FOR_REPLACEMENT: AtomicUsize = AtomicUsize::new(0);
    static INSTALL_COUNT_FOR_IDENTICAL: AtomicUsize = AtomicUsize::new(0);

    fn count_replacement_install(_memory_limits: MlxMemoryLimits) -> Result<(), MlxRuntimeError> {
        INSTALL_COUNT_FOR_REPLACEMENT.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    fn count_identical_install(_memory_limits: MlxMemoryLimits) -> Result<(), MlxRuntimeError> {
        INSTALL_COUNT_FOR_IDENTICAL.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    fn refuse_native_install(_memory_limits: MlxMemoryLimits) -> Result<(), MlxRuntimeError> {
        Err(MlxRuntimeError::InvalidMemoryLimits {
            description: "scripted native refusal",
        })
    }

    fn limits(active_memory_limit_bytes: usize) -> MlxMemoryLimits {
        MlxMemoryLimits::new(active_memory_limit_bytes, active_memory_limit_bytes)
            .expect("positive limits should be valid")
    }

    #[test]
    fn should_replace_a_stale_record_with_the_idle_ceiling() {
        // The 8 GB a previous engine left behind gives way to the 32 GB the
        // user restored while no model was loaded (#1906).
        let limits_record = Mutex::new(Some(limits(8 << 30)));
        reinstall_memory_limits_record(&limits_record, limits(32 << 30), count_replacement_install)
            .expect("an idle reinstall should succeed");
        assert_eq!(*limits_record.lock().unwrap(), Some(limits(32 << 30)));
        assert_eq!(INSTALL_COUNT_FOR_REPLACEMENT.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn should_install_into_an_empty_record_and_skip_identical_limits() {
        let limits_record = Mutex::new(None);
        reinstall_memory_limits_record(&limits_record, limits(16 << 30), count_identical_install)
            .expect("the first install should succeed");
        reinstall_memory_limits_record(&limits_record, limits(16 << 30), count_identical_install)
            .expect("re-applying identical limits should succeed");
        assert_eq!(*limits_record.lock().unwrap(), Some(limits(16 << 30)));
        assert_eq!(
            INSTALL_COUNT_FOR_IDENTICAL.load(Ordering::SeqCst),
            1,
            "identical limits must not touch the native runtime again"
        );
    }

    #[test]
    fn should_keep_the_previous_record_when_the_native_install_fails() {
        let limits_record = Mutex::new(Some(limits(8 << 30)));
        assert!(
            reinstall_memory_limits_record(&limits_record, limits(32 << 30), refuse_native_install)
                .is_err()
        );
        assert_eq!(*limits_record.lock().unwrap(), Some(limits(8 << 30)));
    }
}
