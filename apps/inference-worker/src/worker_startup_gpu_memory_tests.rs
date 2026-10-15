use astronomical_test_harness::offline_denied_test;

use crate::worker_startup_gpu_memory::{
    GpuWiredMemoryLimitSetting, default_policy_wired_boundary_bytes,
    parse_iogpu_wired_limit_setting,
};
use crate::worker_startup_gpu_verification::{
    VERIFICATION_MAX_PROBE_BYTES, VERIFICATION_STEP_BYTES, VerificationStepVerdict,
    next_verification_probe_bytes, verification_step_verdict,
};
use std::time::Duration;

offline_denied_test!(should_parse_a_gpu_wired_limit_as_bytes, {
    let wired_memory_limit_setting = parse_iogpu_wired_limit_setting("1024\n")
        .expect("the synthetic GPU wired-memory limit should parse");

    assert_eq!(
        wired_memory_limit_setting,
        GpuWiredMemoryLimitSetting::ExplicitLimitBytes(1024 * 1024 * 1024)
    );
});

offline_denied_test!(
    should_treat_zero_iogpu_wired_limit_as_system_default_policy,
    {
        let wired_memory_limit_setting = parse_iogpu_wired_limit_setting("0\n")
            .expect("zero should mean the system default GPU wired-memory policy");

        assert_eq!(
            wired_memory_limit_setting,
            GpuWiredMemoryLimitSetting::SystemDefaultPolicy
        );
    }
);

// Issue #1279: the default kernel wired policy enforces a boundary no query
// exposes, materially below the MLX recommended working set. The captured
// incident machine reported a 48,962,207,744-byte recommendation while the
// operating system throttled wiring at 36,827,991,686 bytes of active memory;
// the default-policy estimate must compose strictly below that boundary so a
// ceiling can never silently sit on top of the throttle point.
offline_denied_test!(
    should_estimate_the_default_policy_boundary_below_the_incident_throttle_point,
    {
        let recommended_gpu_working_set_bytes = 48_962_207_744_usize;
        let incident_os_throttle_point_bytes = 36_827_991_686_usize;

        let default_policy_boundary_bytes =
            default_policy_wired_boundary_bytes(recommended_gpu_working_set_bytes);

        assert_eq!(default_policy_boundary_bytes, 36_721_655_808);
        assert!(
            default_policy_boundary_bytes < incident_os_throttle_point_bytes,
            "the default-policy estimate must sit below the observed throttle point"
        );
    }
);

offline_denied_test!(should_scale_the_default_policy_boundary_exactly, {
    let exact_three_quarters_of_maximum =
        usize::try_from(usize::MAX as u128 * 3 / 4).expect("three quarters of usize::MAX fits");

    assert_eq!(
        default_policy_wired_boundary_bytes(usize::MAX),
        exact_three_quarters_of_maximum,
        "the widest recommendation must scale by the exact default fraction without wrapping"
    );
});

offline_denied_test!(
    should_flag_only_steps_beyond_the_latency_budget_as_boundary_hits,
    {
        let step_latency_budget = Duration::from_millis(2_000);

        assert_eq!(
            verification_step_verdict(Duration::from_millis(65), step_latency_budget),
            VerificationStepVerdict::Continue,
            "a healthy wiring step takes tens of milliseconds"
        );
        assert_eq!(
            verification_step_verdict(step_latency_budget, step_latency_budget),
            VerificationStepVerdict::Continue,
            "a step exactly at budget is still healthy"
        );
        assert_eq!(
            verification_step_verdict(
                step_latency_budget + Duration::from_millis(1),
                step_latency_budget
            ),
            VerificationStepVerdict::BoundaryHit,
            "the operating-system crawl signature is a step beyond budget"
        );
    }
);

offline_denied_test!(
    should_grow_verification_probes_geometrically_up_to_the_cap,
    {
        let half_gigabyte = 512 * 1024 * 1024;
        let one_gigabyte = 1_024 * 1024 * 1024;
        let two_gigabytes = 2 * one_gigabyte;
        let thirty_gigabytes = 30 * 1_000_000_000;

        assert_eq!(
            next_verification_probe_bytes(VERIFICATION_STEP_BYTES, thirty_gigabytes),
            one_gigabyte,
            "the first probe doubles the step granularity"
        );
        assert_eq!(
            next_verification_probe_bytes(one_gigabyte, thirty_gigabytes),
            two_gigabytes,
            "growth doubles until the cap"
        );
        assert_eq!(
            next_verification_probe_bytes(two_gigabytes, thirty_gigabytes),
            VERIFICATION_MAX_PROBE_BYTES,
            "growth stops at the probe cap so a throttled machine can never stall an unbounded probe"
        );
        assert_eq!(
            next_verification_probe_bytes(two_gigabytes, one_gigabyte + half_gigabyte),
            one_gigabyte + half_gigabyte,
            "the final probe never overshoots the remaining distance"
        );
        assert_eq!(
            next_verification_probe_bytes(usize::MAX / 2, thirty_gigabytes),
            VERIFICATION_MAX_PROBE_BYTES,
            "doubling saturates instead of overflowing"
        );
    }
);
