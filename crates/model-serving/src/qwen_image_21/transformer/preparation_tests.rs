use super::*;

#[test]
fn should_append_one_image_slot_per_group_of_four_target_tokens() {
    // Eight target tokens form two 2×2 latent groups, so two image slots join
    // the vision-language slots.
    let slot_mask = joint_slot_mask(&[false, true], 8);
    assert_eq!(slot_mask, vec![false, true, true, true]);
}

#[test]
fn should_leave_the_slot_mask_unchanged_when_the_target_has_no_tokens() {
    assert_eq!(joint_slot_mask(&[false, true], 0), vec![false, true]);
    assert_eq!(joint_slot_mask(&[], 0), Vec::<bool>::new());
}

#[test]
fn should_floor_a_partial_target_slot_group() {
    // Six target tokens complete only one 2×2 slot group.
    assert_eq!(joint_slot_mask(&[], 6), vec![true]);
}

#[test]
fn should_expand_image_slots_to_four_tokens_and_keep_text_tokens_single() {
    let joint_mask = expand_slots_to_joint_tokens(&[false, true]);
    assert_eq!(
        joint_mask,
        vec![false, true, true, true, true],
        "each image slot stands for a 2×2 group of latent tokens"
    );
    assert_eq!(joint_mask.len(), 1 + IMG_TOKENS_PER_SLOT);
}

#[test]
fn should_expand_an_empty_slot_mask_to_an_empty_joint_mask() {
    assert_eq!(expand_slots_to_joint_tokens(&[]), Vec::<bool>::new());
}

#[test]
fn should_keep_an_all_text_sequence_at_one_token_per_slot() {
    assert_eq!(
        expand_slots_to_joint_tokens(&[false, false, false]),
        vec![false, false, false]
    );
}

#[test]
fn should_embed_one_row_per_timestep_plus_the_causal_condition_row() {
    let sinusoidal = SinusoidalTimesteps::new(
        TIMESTEP_EMBEDDING_WIDTH,
        SINUSOIDAL_MAX_PERIOD,
        SINUSOIDAL_TIME_FACTOR,
    );
    let timesteps = [0.25_f32, 0.5];
    let width = TIMESTEP_EMBEDDING_WIDTH;

    let embedding_rows = timestep_embedding_rows(&sinusoidal, &timesteps, 2);

    assert_eq!(embedding_rows.len(), 3 * width);
    assert_eq!(
        &embedding_rows[0..width],
        sinusoidal.embed(f64::from(timesteps[0])).as_slice()
    );
    assert_eq!(
        &embedding_rows[width..2 * width],
        sinusoidal.embed(f64::from(timesteps[1])).as_slice()
    );
    assert_eq!(
        &embedding_rows[2 * width..3 * width],
        sinusoidal.embed(0.0).as_slice()
    );
}

#[test]
fn should_emit_only_the_causal_condition_row_without_timesteps() {
    let sinusoidal = SinusoidalTimesteps::new(
        TIMESTEP_EMBEDDING_WIDTH,
        SINUSOIDAL_MAX_PERIOD,
        SINUSOIDAL_TIME_FACTOR,
    );
    let embedding_rows = timestep_embedding_rows(&sinusoidal, &[], 0);
    assert_eq!(embedding_rows.len(), TIMESTEP_EMBEDDING_WIDTH);
    assert_eq!(embedding_rows, sinusoidal.embed(0.0));
}

#[test]
fn should_keep_the_rows_independent_of_the_batch_capacity_hint() {
    let sinusoidal = SinusoidalTimesteps::new(
        TIMESTEP_EMBEDDING_WIDTH,
        SINUSOIDAL_MAX_PERIOD,
        SINUSOIDAL_TIME_FACTOR,
    );
    assert_eq!(
        timestep_embedding_rows(&sinusoidal, &[0.5], 7),
        timestep_embedding_rows(&sinusoidal, &[0.5], 0),
        "batch only sizes the capacity hint, never the embedded values"
    );
}
