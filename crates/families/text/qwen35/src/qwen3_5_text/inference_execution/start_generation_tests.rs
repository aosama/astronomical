//! Beside-source coverage for the pure prompt facts `start_generation`
//! derives before it assembles a session. Assembling the session itself needs
//! a loaded model and runs under the serial engine lane.

use super::{count_image_pad_tokens, request_may_use_astro_ssd_prompt_cache};

const IMAGE_PAD_TOKEN_ID: u32 = 248_056;

#[test]
fn should_count_every_image_placeholder_in_the_prompt() {
    let prompt_token_ids = [
        11,
        IMAGE_PAD_TOKEN_ID,
        IMAGE_PAD_TOKEN_ID,
        12,
        IMAGE_PAD_TOKEN_ID,
    ];

    assert_eq!(
        count_image_pad_tokens(&prompt_token_ids, IMAGE_PAD_TOKEN_ID),
        3
    );
}

#[test]
fn should_count_no_placeholders_in_a_prompt_without_images() {
    assert_eq!(count_image_pad_tokens(&[1, 2, 3], IMAGE_PAD_TOKEN_ID), 0);
}

#[test]
fn should_count_no_placeholders_in_an_empty_prompt() {
    assert_eq!(count_image_pad_tokens(&[], IMAGE_PAD_TOKEN_ID), 0);
}

#[test]
fn should_use_the_astro_ssd_prompt_cache_when_it_is_available_and_no_embeddings_are_precomputed() {
    assert!(request_may_use_astro_ssd_prompt_cache(true, false));
}

#[test]
fn should_not_use_the_astro_ssd_prompt_cache_when_it_is_unavailable() {
    assert!(!request_may_use_astro_ssd_prompt_cache(false, false));
    assert!(!request_may_use_astro_ssd_prompt_cache(false, true));
}

#[test]
fn should_not_use_the_astro_ssd_prompt_cache_for_precomputed_visual_embeddings() {
    assert!(!request_may_use_astro_ssd_prompt_cache(true, true));
}
