//! The sealed fixture pack loaded as a Prism text model, for tests outside
//! the model module that need a real forward on the test GPU.

use std::path::PathBuf;

use runtime_integration::{MlxMemoryLimits, MlxRuntime};

use super::PrismHadamardTextModel;
use crate::prism_hadamard_qwen35::artifacts::FoldedPackValidator;
use crate::prism_hadamard_qwen35::artifacts::tests::pack_fixture::{
    remove_fixture_directory, write_fixture_pack,
};

/// The fixture weights are a few megabytes, so a modest limit keeps these
/// tests far from platform admission behaviour they do not exercise.
const FIXTURE_MLX_MEMORY_LIMIT_BYTES: usize = 512 * 1024 * 1024;

/// A loaded fixture model that removes its pack directory when dropped, so
/// a failing assertion cannot leave the directory behind.
pub(crate) struct LoadedFixtureTextModel {
    pub(crate) text_model: PrismHadamardTextModel,
    fixture_directory: PathBuf,
}

impl Drop for LoadedFixtureTextModel {
    fn drop(&mut self) {
        remove_fixture_directory(&self.fixture_directory);
    }
}

pub(crate) fn load_fixture_text_model() -> LoadedFixtureTextModel {
    let fixture_directory = write_fixture_pack(|_| {});
    let fixture_memory_limits = MlxMemoryLimits::new(
        FIXTURE_MLX_MEMORY_LIMIT_BYTES,
        FIXTURE_MLX_MEMORY_LIMIT_BYTES,
    )
    .expect("the fixture memory limits should be valid");
    let fixture_runtime =
        MlxRuntime::initialize(fixture_memory_limits).expect("the direct MLX runtime should start");
    let validated_pack = FoldedPackValidator::new()
        .validate(&fixture_directory, 4096)
        .expect("the fixture pack should validate");
    let text_model = PrismHadamardTextModel::new(fixture_runtime, validated_pack)
        .expect("the fixture pack should load into the runtime");
    LoadedFixtureTextModel {
        text_model,
        fixture_directory,
    }
}
