use astronomical_model_serving::{
    Qwen3_5MoEArtifactValidator, Qwen3_5MoEModel, RequestDecoderStateStack,
};
use astronomical_runtime_integration::MlxRuntime;

use crate::common::qwen3_5_moe::certified_ornith_config;

const SAY_HI_PROMPT_TOKEN_IDS: [u32; 15] = [
    248_045, 846, 198, 44_240, 15_131, 13, 248_046, 198, 248_045, 74_455, 198, 248_068, 271,
    248_069, 271,
];

#[tokio::test]
#[ignore = "loads and executes the complete pinned 22 GB Ornith artifact"]
async fn should_match_the_certified_first_greedy_token() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let model_directory = crate::common::configured_ornith_model_artifact_directory();
    let validated_artifact = Qwen3_5MoEArtifactValidator::new()
        .validate(&model_directory, 20_480)
        .expect("the pinned Ornith artifact should validate before native loading");
    let mlx_memory_limits =
        crate::common::sample_model_artifact_qualification_mlx_memory_limits().await;
    let runtime = MlxRuntime::initialize(mlx_memory_limits)
        .expect("the direct MLX runtime should initialize");
    let qwen3_5_moe_model =
        Qwen3_5MoEModel::load(runtime, validated_artifact, &model_directory, false)
            .expect("the complete Ornith model should bind from validated descriptors");
    let mut request_decoder_state =
        RequestDecoderStateStack::empty_from_config(&certified_ornith_config());

    qwen3_5_moe_model
        .prefill_chunck(
            &SAY_HI_PROMPT_TOKEN_IDS[..SAY_HI_PROMPT_TOKEN_IDS.len() - 1],
            0,
            &mut request_decoder_state,
        )
        .expect("the prompt prefix should materialize decoder state without unused logits");
    let final_position_logits = qwen3_5_moe_model
        .forward_chunk(
            &SAY_HI_PROMPT_TOKEN_IDS[SAY_HI_PROMPT_TOKEN_IDS.len() - 1..],
            (SAY_HI_PROMPT_TOKEN_IDS.len() - 1) as u32,
            &mut request_decoder_state,
        )
        .expect("the final prompt token should produce logits");
    let first_token_id = qwen3_5_moe_model
        .greedy_token_id(&final_position_logits)
        .expect("the final logits should produce one greedy token");

    assert_eq!(first_token_id, 12_675);
}
