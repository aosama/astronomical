use astronomical_model_serving::{
    ORNITH_1_0_35B_OPTIQ_4BIT_MODEL_ID, ORNITH_1_0_35B_OPTIQ_4BIT_REVISION,
    PersistentPromptCacheModelContract, qwen3_5_decoder_cache_layout,
};

use crate::common::qwen3_5_moe::{
    certified_ornith_config, persistent_visual_embedding_model_contract,
};

#[test]
fn should_derive_persistent_prompt_cache_tensor_shapes_from_certified_model_metadata() {
    let ornith_config = certified_ornith_config();

    let persistent_prompt_cache_model_contract = PersistentPromptCacheModelContract::new(
        ORNITH_1_0_35B_OPTIQ_4BIT_MODEL_ID.to_owned(),
        ORNITH_1_0_35B_OPTIQ_4BIT_REVISION.to_owned(),
        qwen3_5_decoder_cache_layout(&ornith_config)
            .expect("the certified Ornith configuration should build a decoder-cache layout"),
    );
    let persistent_visual_embedding_model_contract = persistent_visual_embedding_model_contract();

    assert_eq!(
        persistent_prompt_cache_model_contract.model_id(),
        ORNITH_1_0_35B_OPTIQ_4BIT_MODEL_ID
    );
    assert_eq!(
        persistent_prompt_cache_model_contract.model_revision(),
        ORNITH_1_0_35B_OPTIQ_4BIT_REVISION
    );
    assert_eq!(
        persistent_prompt_cache_model_contract
            .decoder_cache_layout()
            .layer_count(),
        40
    );
    assert_eq!(
        persistent_prompt_cache_model_contract
            .decoder_cache_layout()
            .sequence_tensor_layouts()
            .first()
            .expect("the certified layout should contain sequence state")
            .tensor_layout()
            .dimensions(),
        &[1, 2, 0, 256]
    );
    assert_eq!(
        persistent_prompt_cache_model_contract
            .decoder_cache_layout()
            .boundary_tensor_layouts()
            .first()
            .expect("the certified layout should contain boundary state")
            .tensor_layout()
            .dimensions(),
        &[1, 3, 8_192]
    );
    assert_eq!(
        persistent_visual_embedding_model_contract.visual_embedding_shape(1_560),
        [1_560, 2_048]
    );
    assert_eq!(
        persistent_visual_embedding_model_contract.maximum_visual_embedding_token_count(),
        16_384
    );
    assert_eq!(
        persistent_prompt_cache_model_contract
            .decoder_cache_layout()
            .sequence_tensor_count(),
        20
    );
    assert_eq!(
        persistent_prompt_cache_model_contract
            .decoder_cache_layout()
            .boundary_tensor_count(),
        60
    );
}
