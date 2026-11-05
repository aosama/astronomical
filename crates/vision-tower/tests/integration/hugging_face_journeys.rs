//! These pin real answers from public repositories, reading only headers and small
//! tensors. They name public models on purpose: the point is that a real upstream
//! and its real MLX conversion are recognized as one base. Revisions are pinned, so
//! a later upload to those repositories cannot change the answer.

use astronomical_vision_tower::{
    ArtifactReference, ArtifactSource, BaseMatchProgress, BaseMatchVerdict, TowerAdditionProgress,
    TowerAdditionRequest, TowerAdditionStage, TowerStatus, VisionTowerError, add_tower,
    inspect_artifact, match_tower_to_text,
};
use serial_test::serial;

const TEXT_ONLY_MLX: &str =
    "ornith-ai/Ornith-1.5-9B-MLX-4bit@a48173b246ac705be75c05bedf1a0666db522d53";
const UPSTREAM_WITH_TOWER: &str =
    "ornith-ai/Ornith-1.5-9B@489cb97981b8654bcfcf30ce1f94ed1b62e07b53";
const LARGER_TERNARY_MODEL: &str =
    "prism-ml/Ternary-Bonsai-2-27B-mlx-2bit@fcba37d2117a7077eac6b613b2668d14d9779edd";

struct SilentProgress;

impl BaseMatchProgress for SilentProgress {
    fn tensor_compared(&mut self, _completed_count: usize, _total_count: usize) {}
}

impl TowerAdditionProgress for SilentProgress {
    fn stage_bytes(&mut self, _stage: TowerAdditionStage, _completed: u64, _total: u64) {}
}

async fn open(reference_text: &str) -> ArtifactSource {
    let reference = ArtifactReference::parse(reference_text).expect("the reference is well formed");
    ArtifactSource::open(&reference)
        .await
        .expect("the public repository opens")
}

#[tokio::test]
#[serial]
#[ignore = "reads public repositories over the network"]
async fn should_see_no_tower_in_a_text_only_mlx_conversion() {
    let inspection = inspect_artifact(&open(TEXT_ONLY_MLX).await)
        .await
        .expect("the inspection runs");
    assert_eq!(inspection.status, TowerStatus::AbsentWithImageTokens);
    assert_eq!(inspection.total_tensor_count, 927);
    assert_eq!(inspection.tower, None);
}

#[tokio::test]
#[serial]
#[ignore = "reads public repositories over the network"]
async fn should_see_the_whole_tower_in_the_upstream_checkpoint() {
    let inspection = inspect_artifact(&open(UPSTREAM_WITH_TOWER).await)
        .await
        .expect("the inspection runs");
    assert_eq!(inspection.status, TowerStatus::Present);
    let tower = inspection.tower.expect("the upstream has a tower");
    assert_eq!(tower.tensor_count, 333);
    assert_eq!(tower.block_count, 27);
    assert_eq!(tower.parameter_count, 456_010_480);
    assert_eq!(tower.name_prefixes, ["model.visual"]);
    assert!(!tower.is_quantized);
}

#[tokio::test]
#[serial]
#[ignore = "reads public repositories over the network"]
async fn should_prove_an_mlx_conversion_shares_its_upstream_base() {
    let report = match_tower_to_text(
        &open(TEXT_ONLY_MLX).await,
        &open(UPSTREAM_WITH_TOWER).await,
        &mut SilentProgress,
    )
    .await
    .expect("the match runs");
    assert_eq!(report.verdict, BaseMatchVerdict::SameBase, "{report:#?}");
    assert!(report.compared_tensors.len() >= 4);
}

#[tokio::test]
#[serial]
#[ignore = "reads public repositories over the network"]
async fn should_refuse_a_tower_for_a_model_of_another_width() {
    let report = match_tower_to_text(
        &open(LARGER_TERNARY_MODEL).await,
        &open(UPSTREAM_WITH_TOWER).await,
        &mut SilentProgress,
    )
    .await
    .expect("the match runs");
    assert_eq!(
        report.verdict,
        BaseMatchVerdict::Incompatible,
        "{report:#?}"
    );
}

#[tokio::test]
#[serial]
#[ignore = "reads public repositories over the network"]
async fn should_ask_for_a_local_text_model_rather_than_download_one() {
    let text_source = open(TEXT_ONLY_MLX).await;
    let tower_source = open(UPSTREAM_WITH_TOWER).await;
    let output_parent = tempfile::tempdir().expect("a temporary folder");
    let output_directory = output_parent.path().join("output");
    let outcome = add_tower(
        TowerAdditionRequest {
            text_source: &text_source,
            tower_source: &tower_source,
            output_directory: &output_directory,
        },
        &mut SilentProgress,
    )
    .await;
    assert!(matches!(
        outcome,
        Err(VisionTowerError::TextModelNotLocal { .. })
    ));
    assert!(!output_directory.exists());
}
