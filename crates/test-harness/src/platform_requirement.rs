//! Tests that need something only one platform has (Core ML, the Metal GPU,
//! the macOS system tools) say so and stop, instead of failing on a platform
//! that cannot have it. The test still runs and passes there, and its output
//! carries a `skipped` line naming what was missing, so a reader of the run
//! sees which contracts were not checked.

/// Something a test needs that not every platform has.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlatformRequirement {
    /// Core ML and the Neural Engine.
    CoreMl,
    /// The Metal GPU, for Astronomical's own Metal kernels. MLX runs other
    /// operations on its CPU backend where there is no Metal, but a custom
    /// Metal kernel has no CPU form.
    MetalGpu,
    /// The macOS system tools (`sw_vers`, `sysctl` with macOS keys).
    MacosSystemTools,
    /// The macOS menu app, whose shared fixtures describe the macOS build's
    /// defaults (for example its 2,048-token prompt chunk).
    MacosMenuApp,
}

impl PlatformRequirement {
    /// Whether the platform this test binary was built for has it.
    #[must_use]
    pub const fn is_met(self) -> bool {
        cfg!(target_os = "macos")
    }

    const fn description(self) -> &'static str {
        match self {
            Self::CoreMl => "Core ML and the Neural Engine, which exist only on macOS",
            Self::MetalGpu => "a custom Metal kernel, which runs only on the Metal GPU of macOS",
            Self::MacosSystemTools => "the macOS system tools",
            Self::MacosMenuApp => "the macOS menu app, whose fixtures describe the macOS build",
        }
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
