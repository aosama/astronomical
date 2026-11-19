use super::contains_token_id_outside_vocabulary;

const VOCABULARY_SIZE: u32 = 151_936;

#[test]
fn should_accept_token_ids_inside_the_vocabulary() {
    assert!(!contains_token_id_outside_vocabulary(
        &[0, 1, 151_834, VOCABULARY_SIZE - 1],
        VOCABULARY_SIZE
    ));
}

#[test]
fn should_reject_a_token_id_equal_to_the_vocabulary_size() {
    // The boundary itself is already outside: IDs are zero-based indexes.
    assert!(contains_token_id_outside_vocabulary(
        &[151_834, VOCABULARY_SIZE],
        VOCABULARY_SIZE
    ));
}

#[test]
fn should_reject_a_token_id_far_beyond_the_vocabulary_size() {
    assert!(contains_token_id_outside_vocabulary(
        &[u32::MAX],
        VOCABULARY_SIZE
    ));
}

#[test]
fn should_accept_an_empty_token_id_slice() {
    assert!(!contains_token_id_outside_vocabulary(&[], VOCABULARY_SIZE));
}

#[test]
fn should_short_circuit_on_the_first_out_of_vocabulary_token_id() {
    // The offending token may appear anywhere; the first one decides.
    assert!(contains_token_id_outside_vocabulary(
        &[151_834, 151_835, VOCABULARY_SIZE + 7, VOCABULARY_SIZE + 9],
        VOCABULARY_SIZE
    ));
}
