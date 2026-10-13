use astronomical_test_harness::offline_test;

use crate::worker_startup_gpu_memory::{
    GpuWiredMemoryLimitSetting, parse_iogpu_wired_limit_setting,
};

offline_test!(should_parse_a_gpu_wired_limit_as_bytes, {
    let wired_memory_limit_setting = parse_iogpu_wired_limit_setting("1024\n")
        .expect("the synthetic GPU wired-memory limit should parse");

    assert_eq!(
        wired_memory_limit_setting,
        GpuWiredMemoryLimitSetting::ExplicitLimitBytes(1024 * 1024 * 1024)
    );
});

offline_test!(
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
