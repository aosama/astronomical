//! Contracts for finding the decoder layer a canonical tensor belongs to.

use super::streamed_layer_binding::decoder_layer_index_of;

#[test]
fn should_find_the_layer_of_a_decoder_layer_tensor() {
    assert_eq!(
        decoder_layer_index_of("language_model.model.layers.31.mlp.down_proj.weight"),
        Some(31)
    );
    assert_eq!(
        decoder_layer_index_of("language_model.model.layers.0.input_layernorm.weight"),
        Some(0)
    );
}

#[test]
fn should_find_no_layer_for_tensors_outside_the_decoder_layers() {
    assert_eq!(
        decoder_layer_index_of("language_model.model.embed_tokens.weight"),
        None
    );
    assert_eq!(
        decoder_layer_index_of("language_model.lm_head.weight"),
        None
    );
    assert_eq!(
        decoder_layer_index_of("vision_tower.blocks.3.attn.proj.weight"),
        None
    );
}

#[test]
fn should_find_no_layer_for_a_malformed_layer_index() {
    assert_eq!(
        decoder_layer_index_of("language_model.model.layers.x.mlp.weight"),
        None
    );
    assert_eq!(
        decoder_layer_index_of("language_model.model.layers.7"),
        None
    );
}
