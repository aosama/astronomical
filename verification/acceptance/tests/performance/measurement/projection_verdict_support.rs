//! What the register-ordered projection verdicts share: the fixture every
//! cell starts from and the pass the profiler times, over the shape copies
//! `k_row_verify_cost` builds.

use std::path::PathBuf;

use mlx_bridge::MlxArray;
use mlx_performance_profiler::MeasuredJob;
use model_serving::row_batched_matmul::RowBatchedMatmulKernels;
use runtime_integration::MlxRuntime;

use super::k_row_verify_cost::{
    MatmulPath, PackedProjection, initialize_runtime, random_rows, shape_copies_pass,
};

/// One pass over a shape's copies through one path: a job the profiler
/// times against another.
pub(super) struct ShapeCopiesPass<'fixture> {
    pub name: &'static str,
    pub runtime: &'fixture MlxRuntime,
    pub matmul_path: MatmulPath<'fixture>,
    pub shape_copies: &'fixture [PackedProjection],
    pub input_rows: &'fixture MlxArray,
}

impl MeasuredJob for ShapeCopiesPass<'_> {
    fn name(&self) -> &str {
        self.name
    }

    fn fresh_outputs(&mut self) -> Result<Vec<MlxArray>, String> {
        Ok(vec![shape_copies_pass(
            self.runtime,
            self.matmul_path,
            self.shape_copies,
            self.input_rows,
        )])
    }
}

/// What every projection verdict starts from: the runtime, the kernels, a
/// seeded key and the folder made for the profiler's pages.
pub(super) struct ProjectionVerdictFixture {
    pub runtime: MlxRuntime,
    pub kernels: RowBatchedMatmulKernels,
    pub random_key: MlxArray,
    pub report_directory: PathBuf,
}

impl ProjectionVerdictFixture {
    pub fn prepare(random_seed: u64, report_subdirectory: &str) -> Self {
        let runtime = initialize_runtime();
        let kernels = RowBatchedMatmulKernels::declare().expect("the kernels are declared");
        let random_key = runtime.random_key(random_seed).expect("a key builds");
        let report_directory = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(report_subdirectory);
        std::fs::create_dir_all(&report_directory).expect("the report directory is made");
        Self {
            runtime,
            kernels,
            random_key,
            report_directory,
        }
    }

    /// Rows evaluated before a comparison starts, so building them is never
    /// timed as part of a job.
    pub fn evaluated_rows(&self, row_count: i32, width: i32) -> MlxArray {
        let input_rows = random_rows(&self.runtime, &self.random_key, row_count, width);
        self.runtime
            .evaluate_arrays(&[&input_rows])
            .expect("the input rows evaluate");
        input_rows
    }
}
