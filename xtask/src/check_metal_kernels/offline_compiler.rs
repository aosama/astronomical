//! Runs Apple's offline Metal compiler over the compile jobs, several at a
//! time, remembering each source that compiled so an unchanged kernel is
//! never compiled twice.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use sha2::{Digest, Sha256};

use super::compile_jobs::CompileJob;

/// Where a compiled source's digest is remembered, under the target folder.
const COMPILED_SOURCE_STORE: &str = "target/metal-kernel-check";
/// How often a long run prints its progress.
const PROGRESS_INTERVAL: Duration = Duration::from_secs(5);
/// MLX compiles custom kernels in safe math mode unless asked otherwise
/// (`CompileOptions::math_mode` in MLX's `custom_kernel.h`).
const MATH_MODE_FLAG: &str = "-fmetal-math-mode=safe";

/// Where the compiler is and what it compiles against.
pub(crate) struct OfflineCompiler {
    pub(crate) mlx_source_directory: PathBuf,
    pub(crate) compiled_source_store: PathBuf,
    pub(crate) scratch_directory: PathBuf,
}

/// One job that did not compile, with the compiler's words.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CompileFailure {
    pub(crate) description: String,
    pub(crate) compiler_output: String,
}

impl OfflineCompiler {
    pub(crate) fn new(
        repository_root: &Path,
        mlx_source_directory: PathBuf,
        scratch_directory: PathBuf,
    ) -> Self {
        Self {
            mlx_source_directory,
            compiled_source_store: repository_root.join(COMPILED_SOURCE_STORE),
            scratch_directory,
        }
    }

    /// The digest a job's compile is remembered by: the source, the language
    /// and the flags, so a flag change compiles everything again.
    fn job_digest(&self, compile_job: &CompileJob) -> String {
        let mut hasher = Sha256::new();
        hasher.update(compile_job.full_source.as_bytes());
        hasher.update(compile_job.language_version.as_bytes());
        hasher.update(MATH_MODE_FLAG.as_bytes());
        hasher.update(self.mlx_source_directory.to_string_lossy().as_bytes());
        lowercase_hex::encode_lowercase_hex(&hasher.finalize())
    }

    /// The jobs not compiled before, by digest; the same source twice is one job.
    pub(crate) fn pending_jobs(
        &self,
        compile_jobs: Vec<CompileJob>,
        is_fresh_run: bool,
    ) -> Vec<(String, CompileJob)> {
        let mut pending_jobs: Vec<(String, CompileJob)> = Vec::new();
        for compile_job in compile_jobs {
            let job_digest = self.job_digest(&compile_job);
            let is_remembered =
                !is_fresh_run && self.compiled_source_store.join(&job_digest).is_file();
            let mut is_duplicate = false;
            for (pending_digest, _) in &pending_jobs {
                is_duplicate |= *pending_digest == job_digest;
            }
            if !is_remembered && !is_duplicate {
                pending_jobs.push((job_digest, compile_job));
            }
        }
        pending_jobs
    }

    /// Compiles every pending job on `worker_count` threads, printing
    /// progress every five seconds, and returns the failures.
    pub(crate) fn compile_all(
        &self,
        pending_jobs: &[(String, CompileJob)],
        worker_count: usize,
    ) -> Vec<CompileFailure> {
        let next_job_index = AtomicUsize::new(0);
        let completed_job_count = AtomicUsize::new(0);
        let failures = Mutex::new(Vec::new());
        let started_at = Instant::now();
        std::thread::scope(|thread_scope| {
            for _ in 0..worker_count.max(1) {
                thread_scope.spawn(|| {
                    self.compile_worker(
                        pending_jobs,
                        &next_job_index,
                        &completed_job_count,
                        &failures,
                    )
                });
            }
            report_progress_until_done(pending_jobs.len(), &completed_job_count, started_at);
        });
        failures.into_inner().unwrap_or_default()
    }

    fn compile_worker(
        &self,
        pending_jobs: &[(String, CompileJob)],
        next_job_index: &AtomicUsize,
        completed_job_count: &AtomicUsize,
        failures: &Mutex<Vec<CompileFailure>>,
    ) {
        loop {
            let job_index = next_job_index.fetch_add(1, Ordering::SeqCst);
            let Some((job_digest, compile_job)) = pending_jobs.get(job_index) else {
                return;
            };
            match self.compile_one(job_digest, compile_job) {
                Ok(()) => {
                    let _ = std::fs::write(self.compiled_source_store.join(job_digest), "");
                }
                Err(compile_failure) => {
                    if let Ok(mut recorded_failures) = failures.lock() {
                        recorded_failures.push(compile_failure);
                    }
                }
            }
            completed_job_count.fetch_add(1, Ordering::SeqCst);
        }
    }

    fn compile_one(
        &self,
        job_digest: &str,
        compile_job: &CompileJob,
    ) -> Result<(), CompileFailure> {
        let source_file = self.scratch_directory.join(format!("{job_digest}.metal"));
        let air_file = self.scratch_directory.join(format!("{job_digest}.air"));
        let failure = |compiler_output: String| CompileFailure {
            description: compile_job.description.clone(),
            compiler_output,
        };
        std::fs::write(&source_file, &compile_job.full_source)
            .map_err(|write_error| failure(write_error.to_string()))?;
        let compiler_run = Command::new("xcrun")
            .args(["-sdk", "macosx", "metal", "-c", "-w", MATH_MODE_FLAG])
            .arg(format!("-std={}", compile_job.language_version))
            .arg("-I")
            .arg(&self.mlx_source_directory)
            .arg("-DMLX_METAL_JIT")
            .arg(&source_file)
            .arg("-o")
            .arg(&air_file)
            .output()
            .map_err(|launch_error| failure(format!("xcrun could not start: {launch_error}")))?;
        let _ = std::fs::remove_file(&air_file);
        if compiler_run.status.success() {
            let _ = std::fs::remove_file(&source_file);
            return Ok(());
        }
        // The failed source is kept beside the remembered digests, so the
        // line numbers in the compiler's words can be read after the run.
        let kept_source = self
            .compiled_source_store
            .join(format!("failed-{job_digest}.metal"));
        let _ = std::fs::rename(&source_file, &kept_source);
        Err(failure(format!(
            "{}(source kept at {})",
            String::from_utf8_lossy(&compiler_run.stderr),
            kept_source.display()
        )))
    }
}

fn report_progress_until_done(
    job_count: usize,
    completed_job_count: &AtomicUsize,
    started_at: Instant,
) {
    let mut last_report_at = Instant::now();
    loop {
        let completed_jobs = completed_job_count.load(Ordering::SeqCst);
        if completed_jobs >= job_count {
            return;
        }
        if last_report_at.elapsed() >= PROGRESS_INTERVAL {
            let elapsed_seconds = started_at.elapsed().as_secs_f64();
            let remaining_seconds = elapsed_seconds / completed_jobs.max(1) as f64
                * (job_count - completed_jobs) as f64;
            println!(
                "[check-metal-kernels] compiled {completed_jobs}/{job_count} ({:.0}%) elapsed={elapsed_seconds:.0}s eta={remaining_seconds:.0}s",
                completed_jobs as f64 * 100.0 / job_count as f64
            );
            last_report_at = Instant::now();
        }
        std::thread::sleep(Duration::from_millis(100));
    }
}
