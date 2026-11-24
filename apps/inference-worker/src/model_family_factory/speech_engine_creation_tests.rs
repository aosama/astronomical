use std::path::PathBuf;

use ipc_protocol::{
    WorkerDecisionModelConfiguration, WorkerModelConfiguration, WorkerSpeechModelConfiguration,
};

use super::{SpeechMemoryGrant, unloaded_speech_engine};

fn memory_grant() -> SpeechMemoryGrant {
    SpeechMemoryGrant {
        effective_mlx_memory_ceiling_bytes: 1 << 30,
        mlx_buffer_cache_limit_bytes: 1 << 28,
    }
}

fn speech_configuration(model_family: String) -> WorkerModelConfiguration {
    WorkerModelConfiguration::Speech(WorkerSpeechModelConfiguration {
        model_id: "Kokoro-82M-bf16".to_owned(),
        model_family,
        artifact_revision: "a71e4d38".to_owned(),
    })
}

// Positive: both speech families build an unloaded engine without touching MLX.
#[test]
fn should_build_an_unloaded_engine_for_each_speech_family() {
    for (model_family, worker_family) in [
        (config::KOKORO_MODEL_FAMILY, "kokoro"),
        (config::PHONON_MODEL_FAMILY, "phonon"),
    ] {
        assert!(
            unloaded_speech_engine(
                Some(model_family),
                PathBuf::from("speech-model"),
                speech_configuration(worker_family.to_owned()),
                &memory_grant(),
            )
            .is_ok()
        );
    }
}

// Negative: a speech configuration for a folder of another family is refused.
#[test]
fn should_refuse_a_speech_configuration_for_another_family() {
    let Err(mismatch_reason) = unloaded_speech_engine(
        Some(config::LAYA_MODEL_FAMILY),
        PathBuf::from("decision-model"),
        speech_configuration("kokoro".to_owned()),
        &memory_grant(),
    ) else {
        panic!("a Laya folder is not a speech model");
    };
    assert!(mismatch_reason.contains("does not match"));
}

// Edge: a speech folder paired with a decision configuration is refused too.
#[test]
fn should_refuse_a_decision_configuration_for_a_speech_folder() {
    let decision_configuration =
        WorkerModelConfiguration::Decisions(WorkerDecisionModelConfiguration {
            model_id: "laya".to_owned(),
            model_family: "laya".to_owned(),
            artifact_revision: "abc".to_owned(),
            maximum_tokens_per_question: 512,
        });
    assert!(
        unloaded_speech_engine(
            Some(config::KOKORO_MODEL_FAMILY),
            PathBuf::from("speech-model"),
            decision_configuration,
            &memory_grant(),
        )
        .is_err()
    );
}
