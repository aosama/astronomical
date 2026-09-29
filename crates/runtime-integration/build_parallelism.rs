//! Resolves native build parallelism so the pinned MLX build never
//! oversubscribes the machine and never silently falls back to a serial
//! build. Under cargo, the native toolchain joins cargo's jobserver so rustc
//! and clang share one job budget; standalone builds pin an explicit,
//! reported job count where every unresolvable input is a loud error.

use std::num::NonZeroU32;

pub(crate) const NATIVE_BUILD_JOBS_VARIABLE: &str = "ASTRONOMICAL_NATIVE_BUILD_JOBS";
const CARGO_BUILD_JOBS_VARIABLE: &str = "CARGO_BUILD_JOBS";
const CARGO_MAKEFLAGS_VARIABLE: &str = "CARGO_MAKEFLAGS";
const MAKE_JOBS_VARIABLE: &str = "NUM_JOBS";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum NativeBuildJobSource {
    ExplicitOverride,
    CargoBuildJobs,
    MakeJobs,
    MachineParallelism,
}

impl NativeBuildJobSource {
    pub(crate) const fn log_value(self) -> &'static str {
        match self {
            Self::ExplicitOverride => "explicit-override",
            Self::CargoBuildJobs => "cargo-build-jobs",
            Self::MakeJobs => "make-jobs",
            Self::MachineParallelism => "machine-parallelism",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ResolvedNativeBuildJobs {
    pub(crate) job_count: NonZeroU32,
    pub(crate) source: NativeBuildJobSource,
}

/// How the native build tool may run its compiler jobs.
///
/// `CargoJobserver` forwards cargo's jobserver to the native toolchain so
/// rustc and clang together never exceed cargo's job budget, even while
/// dependency libraries compile alongside the build script. `FixedJobCount`
/// pins the native toolchain to an explicit count, trading the shared budget
/// for a deterministic value.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum NativeBuildParallelismStrategy {
    CargoJobserver {
        makeflags: String,
    },
    FixedJobCount {
        job_count: NonZeroU32,
        source: NativeBuildJobSource,
    },
}

impl NativeBuildParallelismStrategy {
    pub(crate) fn progress_line(&self) -> String {
        match self {
            Self::CargoJobserver { .. } => {
                // The job budget belongs to cargo and is dynamic, so there is
                // no honest job count to report.
                "status=parallelism source=cargo-jobserver".to_owned()
            }
            Self::FixedJobCount { job_count, source } => {
                format!(
                    "status=parallelism jobs={} source={}",
                    job_count,
                    source.log_value()
                )
            }
        }
    }
}

pub(crate) fn resolve_native_build_jobs(
    read_environment_variable: impl Fn(&str) -> Option<String>,
    machine_parallelism: impl Fn() -> Option<NonZeroU32>,
) -> Result<ResolvedNativeBuildJobs, String> {
    if let Some(explicit_job_count_text) = read_environment_variable(NATIVE_BUILD_JOBS_VARIABLE) {
        return parse_job_count(&explicit_job_count_text, NATIVE_BUILD_JOBS_VARIABLE).map(
            |job_count| ResolvedNativeBuildJobs {
                job_count,
                source: NativeBuildJobSource::ExplicitOverride,
            },
        );
    }
    if let Some(cargo_job_count_text) = read_environment_variable(CARGO_BUILD_JOBS_VARIABLE) {
        return parse_job_count(&cargo_job_count_text, CARGO_BUILD_JOBS_VARIABLE).map(
            |job_count| ResolvedNativeBuildJobs {
                job_count,
                source: NativeBuildJobSource::CargoBuildJobs,
            },
        );
    }
    if let Some(make_job_count_text) = read_environment_variable(MAKE_JOBS_VARIABLE) {
        return parse_job_count(&make_job_count_text, MAKE_JOBS_VARIABLE).map(|job_count| {
            ResolvedNativeBuildJobs {
                job_count,
                source: NativeBuildJobSource::MakeJobs,
            }
        });
    }
    let machine_job_count = machine_parallelism().ok_or_else(|| {
        "unable to resolve native build parallelism from the machine; \
         set ASTRONOMICAL_NATIVE_BUILD_JOBS to an explicit positive job count"
            .to_owned()
    })?;
    Ok(ResolvedNativeBuildJobs {
        job_count: machine_job_count,
        source: NativeBuildJobSource::MachineParallelism,
    })
}

/// Resolves how the native build runs its compiler jobs. An explicit override
/// wins because the operator asked for a deterministic count; otherwise cargo's
/// jobserver caps the combined rustc and clang total; otherwise the fixed
/// job-count chain applies.
pub(crate) fn resolve_native_build_parallelism(
    read_environment_variable: impl Fn(&str) -> Option<String>,
    machine_parallelism: impl Fn() -> Option<NonZeroU32>,
) -> Result<NativeBuildParallelismStrategy, String> {
    if let Some(explicit_job_count_text) = read_environment_variable(NATIVE_BUILD_JOBS_VARIABLE) {
        return parse_job_count(&explicit_job_count_text, NATIVE_BUILD_JOBS_VARIABLE).map(
            |job_count| NativeBuildParallelismStrategy::FixedJobCount {
                job_count,
                source: NativeBuildJobSource::ExplicitOverride,
            },
        );
    }
    if let Some(makeflags) = read_environment_variable(CARGO_MAKEFLAGS_VARIABLE)
        .map(|makeflags_text| makeflags_text.trim().to_owned())
        .filter(|makeflags| !makeflags.is_empty())
    {
        return Ok(NativeBuildParallelismStrategy::CargoJobserver { makeflags });
    }
    resolve_native_build_jobs(read_environment_variable, machine_parallelism).map(
        |resolved_native_build_jobs| NativeBuildParallelismStrategy::FixedJobCount {
            job_count: resolved_native_build_jobs.job_count,
            source: resolved_native_build_jobs.source,
        },
    )
}

fn parse_job_count(job_count_text: &str, variable_name: &str) -> Result<NonZeroU32, String> {
    let parsed_job_count = job_count_text.trim().parse::<u32>().map_err(|_| {
        format!("{variable_name} must be a positive integer, found {job_count_text:?}")
    })?;
    NonZeroU32::new(parsed_job_count)
        .ok_or_else(|| format!("{variable_name} must be at least 1, found {job_count_text:?}"))
}
