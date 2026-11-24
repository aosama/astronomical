//! Tests that need something only one platform has (Core ML, the Metal GPU,
//! the macOS system tools, MLX's GPU backend) say so and stop, instead of
//! failing on a platform that cannot have it. The test still runs and passes
//! there, and its output
//! carries a `skipped` line naming what was missing, so a reader of the run
//! sees which contracts were not checked.

/// Something a test needs that not every platform has.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlatformRequirement {
    /// Core ML and the Neural Engine.
    CoreMl = 0,
    /// The Metal GPU, for Astronomical's own Metal kernels. MLX runs other
    /// operations on its CPU backend where there is no Metal, but a custom
    /// Metal kernel has no CPU form.
    MetalGpu = 1,
    /// The macOS system tools (`sw_vers`, `sysctl` with macOS keys).
    MacosSystemTools = 2,
    /// MLX's GPU backend, for tests that read hundreds of prompt tokens
    /// through a half-precision quantized fixture model. They are correct on
    /// MLX's CPU backend, but its float16 quantized matrix multiply has no
    /// vectorized form there, so one such test takes many minutes instead of
    /// seconds and would stall the whole suite. It is decided by the target
    /// OS like the others: the macOS build always compiles MLX with Metal,
    /// and no other platform's build has a GPU backend yet.
    MlxGpuBackend = 3,
}

/// Each requirement's skip reason, at its explicit discriminant. A table
/// rather than a `match` keeps `description` at one branch however many
/// requirements are added, which the complexity ratchet holds it to.
const SKIP_REASONS: [&str; 4] = [
    "Core ML and the Neural Engine, which exist only on macOS",
    "a custom Metal kernel, which runs only on the Metal GPU of macOS",
    "the macOS system tools",
    "MLX's GPU backend; on its CPU backend this many half-precision prompt tokens take many minutes",
];

// Every requirement needs its row: a variant added after `MlxGpuBackend`
// without one fails this build instead of panicking when a test skips.
const _: () = assert!(SKIP_REASONS.len() == PlatformRequirement::LAST as usize + 1);

impl PlatformRequirement {
    /// The requirement with the highest discriminant; a new variant becomes
    /// this, and the exhaustive match in the tests names every variant.
    const LAST: Self = Self::MlxGpuBackend;

    /// Whether the platform this test binary was built for has it.
    #[must_use]
    pub const fn is_met(self) -> bool {
        cfg!(target_os = "macos")
    }

    const fn description(self) -> &'static str {
        SKIP_REASONS[self as usize]
    }
}

/// Returns `true`, after printing why, when the test must stop because the
/// platform lacks `platform_requirement`. Use it as the first line:
///
/// ```ignore
/// if skip_without(PlatformRequirement::CoreMl, "should_compile_the_program") {
///     return;
/// }
/// ```
#[must_use]
pub fn skip_without(platform_requirement: PlatformRequirement, test_name: &str) -> bool {
    if platform_requirement.is_met() {
        return false;
    }
    println!(
        "skipped {test_name}: needs {}",
        platform_requirement.description()
    );
    true
}

#[cfg(test)]
#[path = "platform_requirement_tests.rs"]
mod tests;
