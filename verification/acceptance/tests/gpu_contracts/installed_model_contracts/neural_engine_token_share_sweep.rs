//! How prefill speed changes with the Neural Engine's share of each chunk, on several
//! real models, with the MLP programs alone and with the output gates beside them.
//!
//! One loaded model runs every arm, so the arms differ only in the programs. Run with
//! no model loaded in either instance.

use std::time::Instant;

use astronomical_config::AstronomicalConfig;
use astronomical_ipc_protocol::NeuralEnginePrefillStatus;
use astronomical_model_serving::Qwen3_5ResidentModel;
use serial_test::serial;

use super::neural_engine_mlp_prefill::{load_named_dense_model, prefill_only_tokens_per_second};

const JOURNEY_NAME: &str = "neural-engine-token-share";
const ROUNDS_PER_ARM: usize = 3;

#[derive(Clone, Copy)]
enum Arm {
    GpuAlone,
    MlpPrograms,
    MlpProgramsAndOutputGates,
}

/// The median of the arm's passes, after one pass that lets the driver wire the programs.
fn arm_tokens_per_second(model: &Qwen3_5ResidentModel, prompt_token_ids: &[u32], arm: Arm) -> f64 {
    model.set_neural_engine_mlp_prefill_paused(matches!(arm, Arm::GpuAlone));
    model.set_neural_engine_output_gate_paused(!matches!(arm, Arm::MlpProgramsAndOutputGates));
    let _warming_pass = prefill_only_tokens_per_second(model, prompt_token_ids);
    let mut passes = Vec::with_capacity(ROUNDS_PER_ARM);
    for _ in 0..ROUNDS_PER_ARM {
        passes.push(prefill_only_tokens_per_second(model, prompt_token_ids));
    }
    passes.sort_by(f64::total_cmp);
    passes[passes.len() / 2]
}

fn sweep_shares(dense_model_id: &str, neural_engine_token_counts: &[u32]) {
    let development_config =
        AstronomicalConfig::load_from_development_location().expect("the config loads");
    let (mut model, prompt_token_ids) = load_named_dense_model(&development_config, dense_model_id);
    let mut chunking = crate::common::standard_worker_chunking_configuration();
    chunking.experimental_neural_engine_prefill_enabled = true;
    let sweep_started_at = Instant::now();
    for (share_index, neural_engine_token_count) in neural_engine_token_counts.iter().enumerate() {
        model.release_neural_engine_mlp_prefill();
        let status = model
            .prepare_neural_engine_prefill_with_token_count(&chunking, *neural_engine_token_count);
        assert!(
            matches!(status, NeuralEnginePrefillStatus::Active { .. }),
            "the programs must load for this sweep: {status:?}"
        );
        let gpu = arm_tokens_per_second(&model, &prompt_token_ids, Arm::GpuAlone);
        let mlp = arm_tokens_per_second(&model, &prompt_token_ids, Arm::MlpPrograms);
        let both = arm_tokens_per_second(&model, &prompt_token_ids, Arm::MlpProgramsAndOutputGates);
        let finished_count = share_index + 1;
        let remaining_seconds = sweep_started_at.elapsed().as_secs_f64() / finished_count as f64
            * (neural_engine_token_counts.len() - finished_count) as f64;
        eprintln!(
            "[{JOURNEY_NAME}] MEASURED model={dense_model_id} ne_tokens={neural_engine_token_count} gpu={gpu:.0} mlp={mlp:.0} ({:.3}x) both={both:.0} ({:.3}x) eta_s={remaining_seconds:.0}",
            mlp / gpu,
            both / gpu,
        );
    }
}

#[tokio::test]
#[ignore = "loads a real 0.8B model and compiles several program sets"]
#[serial]
async fn should_sweep_the_token_share_of_the_smallest_model() {
    let _gpu_exclusive = crate::common::direct_mlx_test_guard().await;
    sweep_shares("Qwen3.5-0.8B-MLX-4bit", &[128, 256, 448, 704, 960, 1216]);
}

#[tokio::test]
#[ignore = "loads a real 2B model and compiles several program sets"]
#[serial]
async fn should_sweep_the_token_share_of_the_two_billion_parameter_model() {
    let _gpu_exclusive = crate::common::direct_mlx_test_guard().await;
    sweep_shares("Qwen3.5-2B-4bit", &[448, 704, 960, 1216]);
}

#[tokio::test]
#[ignore = "loads a real 9B model and compiles several program sets"]
#[serial]
async fn should_sweep_the_token_share_of_the_nine_billion_parameter_model() {
    let _gpu_exclusive = crate::common::direct_mlx_test_guard().await;
    sweep_shares("Ornith-1.5-9B-MLX-4bit-vision", &[704, 960, 1216]);
}

/// Twice at the production share, because one disturbed arm decides nothing: whether the
/// output gates earn their memory on the model they should help most.
#[tokio::test]
#[ignore = "loads a real 9B model and its programs"]
#[serial]
async fn should_compare_output_gates_on_the_nine_billion_parameter_model() {
    let _gpu_exclusive = crate::common::direct_mlx_test_guard().await;
    sweep_shares("Ornith-1.5-9B-MLX-4bit-vision", &[960, 960]);
}
