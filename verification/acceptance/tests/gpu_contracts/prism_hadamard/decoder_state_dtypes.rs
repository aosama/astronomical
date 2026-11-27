//! The precisions a folded ternary pack's decoder state is saved in must be
//! the precisions its forward really produces (0.2.136 regression).
//!
//! The `astro_ssd_prompt_cache` writes a block in the layout the model
//! predicts at load and checks the block against it before keeping it. On
//! Ternary Bonsai 2.5 the layout predicted float32 attention keys while the
//! forward stored float16, so the check refused the first block of every long
//! prompt and the request stopped. This journey reads a short prompt and
//! compares, layer by layer, the precision of every key and value tensor the
//! forward stored with the precision the layout expects.

use model_serving::AstroDecoderStateCacheState;
use serial_test::serial;

use super::shared_core::{empty_shared_core_state, loaded_shared_core};
use super::support::{pack_directory, pack_tokenizer, romeo_and_juliet_prompt_token_ids};

/// Long enough to fill every layer's state, short enough to read in seconds.
const PROMPT_TOKENS: usize = 64;

#[test]
#[serial]
#[ignore = "loads an installed folded ternary pack on the GPU; run through cargo test-prism-hadamard-direct with model_directory configured"]
fn should_save_attention_state_in_the_precision_the_forward_produces() {
    let shared_core = loaded_shared_core(&pack_directory());
    let prompt_token_ids: Vec<u32> =
        romeo_and_juliet_prompt_token_ids(&pack_tokenizer(), PROMPT_TOKENS)
            .into_iter()
            .map(|token_id| token_id as u32)
            .collect();
    let mut decoder_state = empty_shared_core_state(&shared_core);
    shared_core
        .forward_chunk(&prompt_token_ids, 0, &mut decoder_state)
        .expect("the shared core reads the prompt");

    let expected_dtype_by_tensor_name = shared_core
        .astro_decoder_state_cache_layout()
        .sequence_tensor_layouts()
        .into_iter()
        .map(|persisted_tensor_layout| {
            (
                persisted_tensor_layout.persistent_tensor_name(),
                format!("{:?}", persisted_tensor_layout.tensor_layout().dtype()),
            )
        })
        .collect::<std::collections::HashMap<_, _>>();
    let mismatches = attention_dtype_mismatches(&decoder_state, &expected_dtype_by_tensor_name);
    assert!(
        mismatches.is_empty(),
        "the layout's precisions must match the forward's: {mismatches:?}"
    );
}

/// Every attention tensor whose stored precision differs from the layout's,
/// as `name: layout expects X, forward stored Y`.
fn attention_dtype_mismatches(
    decoder_state: &model_serving::RequestDecoderStateStack,
    expected_dtype_by_tensor_name: &std::collections::HashMap<String, String>,
) -> Vec<String> {
    let mut mismatches = Vec::new();
    for (layer_index, layer_state) in decoder_state.layers().iter().enumerate() {
        let AstroDecoderStateCacheState::AppendOnlyAttention { attention } = layer_state else {
            continue;
        };
        for (role, stored) in [
            ("attention.keys", attention.keys_state()),
            ("attention.values", attention.values_state()),
        ] {
            let tensor_name = format!("layer_{layer_index}_{role}");
            let stored_dtype = format!(
                "{:?}",
                stored
                    .expect("the forward stored this layer's state")
                    .dtype()
            );
            let expected_dtype = expected_dtype_by_tensor_name
                .get(&tensor_name)
                .cloned()
                .unwrap_or_else(|| "nothing".to_owned());
            if !same_precision(&expected_dtype, &stored_dtype) {
                mismatches.push(format!(
                    "{tensor_name}: layout expects {expected_dtype}, forward stored {stored_dtype}"
                ));
            }
        }
    }
    mismatches
}

/// The layout and MLX can spell a precision differently (`BFloat16` and
/// `Bfloat16`); compare them case-insensitively.
fn same_precision(layout_dtype: &str, stored_dtype: &str) -> bool {
    layout_dtype.eq_ignore_ascii_case(stored_dtype)
}
