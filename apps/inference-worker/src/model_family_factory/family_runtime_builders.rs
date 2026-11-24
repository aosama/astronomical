//! Which engine each model family is served by (issue #2473, Step 5).
//!
//! The factory looks a classified family up here instead of matching on it.
//! The table is the family list itself (`model_families.list` in the
//! registry crate), so a family has one line for discovery and the worker.
//! Decision and speech families are built by kind from the family's own
//! type, so a new family of those kinds adds one line to the table and no
//! code here; chat, image and embedding families still have a builder each. Each builder
//! takes its own kind of configuration, and a family sent another kind's
//! policy is refused before any builder runs.

use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;

use config::ModelFamily;
use ipc_protocol::{
    WorkerAutoregressiveModelConfiguration, WorkerDecisionModelConfiguration,
    WorkerEmbeddingModelConfiguration, WorkerImageGenerationModelConfiguration,
    WorkerModelConfiguration, WorkerSpeechModelConfiguration,
};
use model_serving::decision_models::decision_engine_shell::{DecisionEngineShell, DecisionFamily};
use model_serving::{EmbeddingFamilyConstruction, ImageFamilyConstruction};
use model_serving::{ModelFactoryRuntime, SpeechEngine, SpeechEngineShell, SpeechFamily};

use super::configured_family_check::FAMILY_MISMATCH_REASON;
use super::{ModelFamilyFactory, ModelFamilyRuntime};

/// A builder's unloaded runtime, borrowing the factory's settings.
pub(super) type RuntimeFuture<'factory> =
    Pin<Box<dyn Future<Output = Result<ModelFamilyRuntime, String>> + Send + 'factory>>;

/// One family's builder, typed by the kind of configuration it takes.
pub(super) enum FamilyRuntimeBuilder {
    Chat(
        for<'factory> fn(
            &'factory ModelFamilyFactory,
            PathBuf,
            WorkerAutoregressiveModelConfiguration,
        ) -> RuntimeFuture<'factory>,
    ),
    ImageGeneration(
        for<'factory> fn(
            &'factory ModelFamilyFactory,
            PathBuf,
            WorkerImageGenerationModelConfiguration,
        ) -> RuntimeFuture<'factory>,
    ),
    Embedding(
        fn(
            &ModelFamilyFactory,
            PathBuf,
            WorkerEmbeddingModelConfiguration,
        ) -> Result<ModelFamilyRuntime, String>,
    ),
    Decision(
        fn(
            &ModelFamilyFactory,
            PathBuf,
            WorkerDecisionModelConfiguration,
        ) -> Result<ModelFamilyRuntime, String>,
    ),
    Speech(
        fn(
            &ModelFamilyFactory,
            PathBuf,
            WorkerSpeechModelConfiguration,
        ) -> Result<ModelFamilyRuntime, String>,
    ),
}

/// Reads a line of the family list as the family and the builder the
/// worker serves it with; the description half is discovery's.
macro_rules! model_family {
    ($family_description:ident, $builder_kind:ident ( $($runtime_builder:tt)* )) => {
        (
            model_family_registry::$family_description.model_family,
            FamilyRuntimeBuilder::$builder_kind($($runtime_builder)*),
        )
    };
}

/// Every family the worker serves, with its builder: the same list, line
/// for line, that discovery recognizes families from.
static FAMILY_RUNTIME_BUILDERS: &[(ModelFamily, FamilyRuntimeBuilder)] =
    &include!("../../../../crates/families/registry/model-family-registry/src/model_families.list");

/// The builder the worker serves `model_family` with, if it serves it.
pub(super) fn family_runtime_builder(
    model_family: ModelFamily,
) -> Option<&'static FamilyRuntimeBuilder> {
    FAMILY_RUNTIME_BUILDERS
        .iter()
        .find(|(registered_family, _)| *registered_family == model_family)
        .map(|(_, runtime_builder)| runtime_builder)
}

impl FamilyRuntimeBuilder {
    /// Builds the family's unloaded runtime when the policy is its kind's.
    pub(super) async fn build(
        &self,
        model_family_factory: &ModelFamilyFactory,
        model_directory_path: PathBuf,
        model_configuration: WorkerModelConfiguration,
    ) -> Result<ModelFamilyRuntime, String> {
        match (self, model_configuration) {
            (Self::Chat(build), WorkerModelConfiguration::Autoregressive(chat_configuration)) => {
                build(
                    model_family_factory,
                    model_directory_path,
                    chat_configuration,
                )
                .await
            }
            (
                Self::ImageGeneration(build),
                WorkerModelConfiguration::ImageGeneration(image_configuration),
            ) => {
                build(
                    model_family_factory,
                    model_directory_path,
                    image_configuration,
                )
                .await
            }
            (
                Self::Embedding(build),
                WorkerModelConfiguration::Embeddings(embedding_configuration),
            ) => build(
                model_family_factory,
                model_directory_path,
                embedding_configuration,
            ),
            (
                Self::Decision(build),
                WorkerModelConfiguration::Decisions(decision_configuration),
            ) => build(
                model_family_factory,
                model_directory_path,
                decision_configuration,
            ),
            (Self::Speech(build), WorkerModelConfiguration::Speech(speech_configuration)) => build(
                model_family_factory,
                model_directory_path,
                speech_configuration,
            ),
            _ => Err(FAMILY_MISMATCH_REASON.to_owned()),
        }
    }
}

fn build_qwen3_5_runtime(
    model_family_factory: &ModelFamilyFactory,
    model_directory_path: PathBuf,
    chat_configuration: WorkerAutoregressiveModelConfiguration,
) -> RuntimeFuture<'_> {
    Box::pin(
        model_family_factory
            .create_qwen3_5_autoregressive_runtime(model_directory_path, chat_configuration),
    )
}

fn build_prism_hadamard_qwen35_runtime(
    model_family_factory: &ModelFamilyFactory,
    model_directory_path: PathBuf,
    chat_configuration: WorkerAutoregressiveModelConfiguration,
) -> RuntimeFuture<'_> {
    Box::pin(
        model_family_factory
            .create_prism_hadamard_autoregressive_runtime(model_directory_path, chat_configuration),
    )
}

/// Every image family: its own folder check, then its engine.
fn build_image_runtime<Family: ImageFamilyConstruction>(
    model_family_factory: &ModelFamilyFactory,
    model_directory_path: PathBuf,
    image_configuration: WorkerImageGenerationModelConfiguration,
) -> RuntimeFuture<'_> {
    Box::pin(
        model_family_factory
            .create_image_engine::<Family>(model_directory_path, image_configuration),
    )
}

/// Every embedding family: its engine from the family's own type.
fn build_embedding_runtime<Family: EmbeddingFamilyConstruction>(
    model_family_factory: &ModelFamilyFactory,
    model_directory_path: PathBuf,
    embedding_configuration: WorkerEmbeddingModelConfiguration,
) -> Result<ModelFamilyRuntime, String> {
    model_family_factory
        .create_embedding_engine::<Family>(model_directory_path, embedding_configuration)
}

/// Every decision family runs in the decision shell, so its engine is the
/// shell over the family's own type.
fn build_decision_runtime<Family: DecisionFamily + 'static>(
    model_family_factory: &ModelFamilyFactory,
    model_directory_path: PathBuf,
    decision_configuration: WorkerDecisionModelConfiguration,
) -> Result<ModelFamilyRuntime, String> {
    let decision_engine = DecisionEngineShell::<Family>::new(
        decision_configuration.model_id,
        &model_directory_path,
        model_family_factory.effective_mlx_memory_ceiling_bytes,
        model_family_factory.mlx_buffer_cache_limit_bytes,
    );
    model_family_factory.decision_runtime(&model_directory_path, decision_engine)
}

/// Every speech family runs in the speech shell. The unloaded engine holds
/// only its settings; the MLX runtime is created inside `load()` on the
/// worker runtime thread, because MLX handles are not `Send` across task
/// boundaries.
fn build_speech_runtime<Family: SpeechFamily>(
    model_family_factory: &ModelFamilyFactory,
    model_directory_path: PathBuf,
    speech_configuration: WorkerSpeechModelConfiguration,
) -> Result<ModelFamilyRuntime, String>
where
    SpeechEngineShell<Family>: SpeechEngine,
{
    Ok(ModelFactoryRuntime::Speech(Box::new(SpeechEngineShell::<
        Family,
    >::new(
        speech_configuration.model_id,
        model_directory_path,
        model_family_factory.effective_mlx_memory_ceiling_bytes,
        model_family_factory.mlx_buffer_cache_limit_bytes,
    ))))
}

#[cfg(test)]
#[path = "family_runtime_builders_tests.rs"]
mod family_runtime_builders_tests;
