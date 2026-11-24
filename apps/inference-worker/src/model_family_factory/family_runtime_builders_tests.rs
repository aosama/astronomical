use std::path::PathBuf;

use config::{AstroSsdPromptCacheConfig, ModelFamily, ModelKind};
use ipc_protocol::{
    WorkerDecisionModelConfiguration, WorkerDecodeAstroExpertCachePolicy, WorkerModelConfiguration,
};
use model_serving::ModelFactoryRuntime;

use super::super::ModelFamilyFactory;
use super::super::configured_family_check::FAMILY_MISMATCH_REASON;
use super::super::family_test_support::speech_configuration_naming;
use super::{FAMILY_RUNTIME_BUILDERS, FamilyRuntimeBuilder, family_runtime_builder};

const SERVED_FAMILIES: [ModelFamily; 10] = [
    config::QWEN3_5_MODEL_FAMILY,
    config::PRISM_HADAMARD_QWEN35_MODEL_FAMILY,
    config::FLUX2_KLEIN_MODEL_FAMILY,
    config::QWEN_IMAGE_21_MODEL_FAMILY,
    config::MODERNBERT_MODEL_FAMILY,
    config::LAYA_MODEL_FAMILY,
    config::KEV_MODEL_FAMILY,
    config::CLEF_MODEL_FAMILY,
    config::KOKORO_MODEL_FAMILY,
    config::PHONON_MODEL_FAMILY,
];

fn model_family_factory() -> ModelFamilyFactory {
    ModelFamilyFactory::new(
        1 << 30,
        1 << 28,
        AstroSsdPromptCacheConfig::new(PathBuf::from("astro-ssd-prompt-cache"), 1 << 30),
        false,
        WorkerDecodeAstroExpertCachePolicy::Baseline,
        PathBuf::from("performance-attribution.jsonl"),
        false,
    )
}

const fn builder_kind(runtime_builder: &FamilyRuntimeBuilder) -> ModelKind {
    match runtime_builder {
        FamilyRuntimeBuilder::Chat(_) => ModelKind::Chat,
        FamilyRuntimeBuilder::ImageGeneration(_) => ModelKind::ImageGeneration,
        FamilyRuntimeBuilder::Embedding(_) => ModelKind::Embedding,
        FamilyRuntimeBuilder::Decision(_) => ModelKind::Decision,
        FamilyRuntimeBuilder::Speech(_) => ModelKind::Speech,
    }
}

#[test]
fn should_serve_every_family_with_a_builder_of_its_own_kind() {
    for model_family in SERVED_FAMILIES {
        let runtime_builder = family_runtime_builder(model_family)
            .unwrap_or_else(|| panic!("{} has no builder", model_family.as_str()));
        assert_eq!(
            Some(builder_kind(runtime_builder)),
            config::model_kind_of(model_family),
            "{}",
            model_family.as_str()
        );
    }
    assert_eq!(FAMILY_RUNTIME_BUILDERS.len(), SERVED_FAMILIES.len());
}

#[test]
fn should_have_no_builder_for_a_family_nothing_registers() {
    assert!(family_runtime_builder(ModelFamily::named("qwen3_6")).is_none());
}

#[tokio::test]
async fn should_build_an_unloaded_engine_for_each_speech_family_without_mlx() {
    for (model_family, identifier) in [
        (config::KOKORO_MODEL_FAMILY, "kokoro"),
        (config::PHONON_MODEL_FAMILY, "phonon"),
    ] {
        let speech_runtime = family_runtime_builder(model_family)
            .expect("speech families are served")
            .build(
                &model_family_factory(),
                PathBuf::from("speech-model"),
                speech_configuration_naming(identifier),
            )
            .await;

        assert!(
            matches!(speech_runtime, Ok(ModelFactoryRuntime::Speech(_))),
            "{identifier}"
        );
    }
}

#[tokio::test]
async fn should_refuse_a_policy_of_another_kind_before_building() {
    let decision_policy = WorkerModelConfiguration::Decisions(WorkerDecisionModelConfiguration {
        model_id: "decision-model".to_owned(),
        model_family: "kokoro".to_owned(),
        artifact_revision: "decision-revision".to_owned(),
        maximum_tokens_per_question: 512,
    });

    let refusal = family_runtime_builder(config::KOKORO_MODEL_FAMILY)
        .expect("speech families are served")
        .build(
            &model_family_factory(),
            PathBuf::from("speech-model"),
            decision_policy,
        )
        .await;

    assert!(matches!(refusal, Err(reason) if reason == FAMILY_MISMATCH_REASON));
}

#[tokio::test]
async fn should_refuse_an_empty_folder_for_each_decision_family() {
    // The generic decision builder reaches each family's own folder
    // preparation; an empty folder must come back as the worker's bounded
    // refusal, not a panic and not an engine.
    let empty_model_directory =
        tempfile::tempdir().expect("an empty model directory should be created");
    for (model_family, identifier) in [
        (config::LAYA_MODEL_FAMILY, "laya"),
        (config::KEV_MODEL_FAMILY, "kev"),
        (config::CLEF_MODEL_FAMILY, "clef"),
    ] {
        let decision_policy =
            WorkerModelConfiguration::Decisions(WorkerDecisionModelConfiguration {
                model_id: format!("{identifier}-model"),
                model_family: identifier.to_owned(),
                artifact_revision: "decision-revision".to_owned(),
                maximum_tokens_per_question: 512,
            });

        let refusal = family_runtime_builder(model_family)
            .expect("decision families are served")
            .build(
                &model_family_factory(),
                empty_model_directory.path().to_path_buf(),
                decision_policy,
            )
            .await;

        assert!(
            matches!(&refusal, Err(reason) if reason.starts_with("selected decision artifact failed initialization: ")),
            "{identifier}"
        );
    }
}
