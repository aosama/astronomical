//! The engines one worker hosts, named once.
//!
//! A worker serves one autoregressive model at a time, or an image, an
//! embeddings, a decisions or a speech model, and builds each through a model
//! factory. Those are seven types, and every part of the worker needs all of
//! them with the bounds that tie them together. Bundled as associated types
//! of one trait, each part of the worker names one parameter,
//! `impl<Engines: WorkerEngines> EngineBackedWorker<Engines>`, instead of
//! restating seven parameters and their bounds in every file.

use std::marker::PhantomData;

use super::support::ModelFactory;
use crate::model_generation_processor::ModelGenerationProcessor;
use crate::{ImageGenerationEngine, InferenceEngine};

/// The engines, the processor and the factory of one kind of worker.
pub trait WorkerEngines: 'static {
    /// Turns protocol requests into the autoregressive engine's requests.
    type Processor: ModelGenerationProcessor + Send + 'static;
    /// The autoregressive engine.
    type Engine: InferenceEngine<Request = <Self::Processor as ModelGenerationProcessor>::InferenceRequest>
        + Send
        + 'static;
    type ImageEngine: ImageGenerationEngine;
    type EmbeddingsEngine: crate::EmbeddingEngine;
    type DecisionsEngine: crate::DecisionEngine;
    type TextToSpeechEngine: crate::SpeechEngine;
    /// Builds a runtime of any of the kinds above for a model swap.
    type Factory: ModelFactory<
            Self::Processor,
            Self::Engine,
            Self::ImageEngine,
            Self::EmbeddingsEngine,
            Self::DecisionsEngine,
            Self::TextToSpeechEngine,
        > + Send
        + 'static;
}

/// What the worker's processor makes of the autoregressive engine's output.
pub(crate) type RequestOutputOf<Engines> =
    <<Engines as WorkerEngines>::Processor as ModelGenerationProcessor>::RequestOutput;

/// The autoregressive engine's request, as the worker's processor builds it.
pub(crate) type InferenceRequestOf<Engines> =
    <<Engines as WorkerEngines>::Processor as ModelGenerationProcessor>::InferenceRequest;

/// The runtime a worker of these engines has loaded.
pub(crate) type LoadedRuntimeOf<Engines> = super::LoadedRuntime<
    <Engines as WorkerEngines>::Processor,
    <Engines as WorkerEngines>::Engine,
    <Engines as WorkerEngines>::ImageEngine,
    <Engines as WorkerEngines>::EmbeddingsEngine,
    <Engines as WorkerEngines>::DecisionsEngine,
    <Engines as WorkerEngines>::TextToSpeechEngine,
>;

/// What the factory of these engines builds for a model swap.
pub(crate) type ModelFactoryRuntimeOf<Engines> = super::ModelFactoryRuntime<
    <Engines as WorkerEngines>::Processor,
    <Engines as WorkerEngines>::Engine,
    <Engines as WorkerEngines>::ImageEngine,
    <Engines as WorkerEngines>::EmbeddingsEngine,
    <Engines as WorkerEngines>::DecisionsEngine,
    <Engines as WorkerEngines>::TextToSpeechEngine,
>;

/// A replacement runtime a model swap has loaded for these engines.
pub(super) type LoadedReplacementOf<Engines> = super::model_swap::LoadedReplacement<
    <Engines as WorkerEngines>::Processor,
    <Engines as WorkerEngines>::Engine,
    <Engines as WorkerEngines>::ImageEngine,
    <Engines as WorkerEngines>::EmbeddingsEngine,
    <Engines as WorkerEngines>::DecisionsEngine,
    <Engines as WorkerEngines>::TextToSpeechEngine,
>;

/// The engines a worker hosts, named by type in the order `LoadedRuntime`
/// and `ModelFactory` take them. Only a type: no value of it is ever made,
/// so its markers hold the types and nothing else. They are: an autoregressive processor and engine, the factory that builds
/// runtimes on a model swap (none by default), and the other kinds of engine,
/// unavailable unless named.
pub struct FactoryEngines<
    Processor,
    Engine,
    Factory = (),
    ImageEngine = crate::ImageGenerationUnavailableEngine,
    EmbeddingsEngine = crate::EmbeddingUnavailableEngine,
    DecisionsEngine = crate::DecisionUnavailableEngine,
    TextToSpeechEngine = crate::SpeechUnavailableEngine,
>(
    PhantomData<(Processor, Engine, Factory)>,
    PhantomData<(ImageEngine, EmbeddingsEngine)>,
    PhantomData<(DecisionsEngine, TextToSpeechEngine)>,
);

impl<Processor, Engine, Factory, ImageEngine, EmbeddingsEngine, DecisionsEngine, TextToSpeechEngine>
    WorkerEngines
    for FactoryEngines<
        Processor,
        Engine,
        Factory,
        ImageEngine,
        EmbeddingsEngine,
        DecisionsEngine,
        TextToSpeechEngine,
    >
where
    Processor: ModelGenerationProcessor + Send + 'static,
    Engine: InferenceEngine<Request = Processor::InferenceRequest> + Send + 'static,
    Factory: ModelFactory<
            Processor,
            Engine,
            ImageEngine,
            EmbeddingsEngine,
            DecisionsEngine,
            TextToSpeechEngine,
        > + Send
        + 'static,
    ImageEngine: ImageGenerationEngine + 'static,
    EmbeddingsEngine: crate::EmbeddingEngine + 'static,
    DecisionsEngine: crate::DecisionEngine + 'static,
    TextToSpeechEngine: crate::SpeechEngine + 'static,
{
    type Processor = Processor;
    type Engine = Engine;
    type ImageEngine = ImageEngine;
    type EmbeddingsEngine = EmbeddingsEngine;
    type DecisionsEngine = DecisionsEngine;
    type TextToSpeechEngine = TextToSpeechEngine;
    type Factory = Factory;
}

/// A worker given one autoregressive engine directly: no factory, so no
/// model swaps, and none of the other kinds of engine.
pub type DirectEngines<Processor, Engine> = FactoryEngines<Processor, Engine>;
