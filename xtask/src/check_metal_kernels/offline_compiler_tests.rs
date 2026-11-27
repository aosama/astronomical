use std::path::PathBuf;

use super::compile_jobs::CompileJob;
use super::offline_compiler::OfflineCompiler;

fn compile_job(full_source: &str, language_version: &'static str) -> CompileJob {
    CompileJob {
        description: format!("a job compiled as {language_version}"),
        language_version,
        full_source: full_source.to_owned(),
    }
}

fn compiler_in(repository_root: &std::path::Path) -> OfflineCompiler {
    let compiler = OfflineCompiler::new(
        repository_root,
        PathBuf::from("/mlx-src"),
        repository_root.join("scratch"),
    );
    std::fs::create_dir_all(&compiler.compiled_source_store).expect("the store is created");
    compiler
}

#[test]
fn should_compile_each_distinct_source_and_language_once() {
    let repository = tempfile::tempdir().expect("a temporary folder");
    let compiler = compiler_in(repository.path());
    let pending_jobs = compiler.pending_jobs(
        vec![
            compile_job("kernel A", "metal4.0"),
            compile_job("kernel A", "metal4.0"),
            compile_job("kernel A", "metal4.1"),
            compile_job("kernel B", "metal4.0"),
        ],
        false,
    );
    assert_eq!(
        pending_jobs.len(),
        3,
        "the repeated source and language is one job"
    );
}

#[test]
fn should_skip_a_source_that_compiled_before_unless_the_run_is_fresh() {
    let repository = tempfile::tempdir().expect("a temporary folder");
    let compiler = compiler_in(repository.path());
    let first_pending = compiler.pending_jobs(vec![compile_job("kernel A", "metal4.0")], false);
    let (job_digest, _) = &first_pending[0];
    std::fs::write(compiler.compiled_source_store.join(job_digest), "")
        .expect("the compile is remembered");
    assert!(
        compiler
            .pending_jobs(vec![compile_job("kernel A", "metal4.0")], false)
            .is_empty()
    );
    assert_eq!(
        compiler
            .pending_jobs(vec![compile_job("kernel A", "metal4.0")], true)
            .len(),
        1
    );
}

#[test]
fn should_find_nothing_to_compile_in_an_empty_list() {
    let repository = tempfile::tempdir().expect("a temporary folder");
    assert!(
        compiler_in(repository.path())
            .pending_jobs(Vec::new(), true)
            .is_empty()
    );
}
