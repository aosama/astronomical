//! The idle MLX memory observation of an engine whose requests keep nothing
//! between them: an embedding, decision or speech engine.

use super::LoadedRuntime;
use crate::memory::telemetry::MlxMemoryTelemetry;

/// The idle memory observation of an engine whose requests keep nothing
/// between them: an embedding, decision or speech engine. Other runtimes
/// report through their own arms, so they yield nothing here.
pub(super) fn forward_pass_engine_memory_telemetry<
    Processor,
    Engine,
    ImageEngine,
    EmbeddingsEngine,
    DecisionsEngine,
    TextToSpeechEngine,
>(
    loaded_runtime: &LoadedRuntime<
        Processor,
        Engine,
        ImageEngine,
        EmbeddingsEngine,
        DecisionsEngine,
        TextToSpeechEngine,
    >,
) -> Option<MlxMemoryTelemetry>
where
    EmbeddingsEngine: crate::EmbeddingEngine,
    DecisionsEngine: crate::DecisionEngine,
    TextToSpeechEngine: crate::SpeechEngine,
{
    if let LoadedRuntime::Embeddings(embedding_engine) = loaded_runtime {
        return embedding_engine.collect_mlx_memory_telemetry();
    }
    if let LoadedRuntime::Decisions(decision_engine) = loaded_runtime {
        return decision_engine.collect_mlx_memory_telemetry();
    }
    speech_engine_memory_telemetry(loaded_runtime)
}

fn speech_engine_memory_telemetry<
    Processor,
    Engine,
    ImageEngine,
    EmbeddingsEngine,
    DecisionsEngine,
    TextToSpeechEngine,
>(
    loaded_runtime: &LoadedRuntime<
        Processor,
        Engine,
        ImageEngine,
        EmbeddingsEngine,
        DecisionsEngine,
        TextToSpeechEngine,
    >,
) -> Option<MlxMemoryTelemetry>
where
    TextToSpeechEngine: crate::SpeechEngine,
{
    match loaded_runtime {
        LoadedRuntime::Speech(speech_engine) => speech_engine.collect_mlx_memory_telemetry(),
        _ => None,
    }
}
