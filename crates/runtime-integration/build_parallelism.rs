//! Resolves native build parallelism so the pinned MLX CMake build always runs
//! with an explicit, reported job count. A silent fallback to a serial build
//! would multiply CI wall time, so every unresolvable input is a loud error.

use std::num::NonZeroU32;

pub(crate) const NATIVE_BUILD_JOBS_VARIABLE: &str = "ASTRONOMICAL_NATIVE_BUILD_JOBS";
const CARGO_BUILD_JOBS_VARIABLE: &str = "CARGO_BUILD_JOBS";
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

fn parse_job_count(job_count_text: &str, variable_name: &str) -> Result<NonZeroU32, String> {
    let parsed_job_count = job_count_text.trim().parse::<u32>().map_err(|_| {
        format!("{variable_name} must be a positive integer, found {job_count_text:?}")
    })?;
    NonZeroU32::new(parsed_job_count)
        .ok_or_else(|| format!("{variable_name} must be at least 1, found {job_count_text:?}"))
}
