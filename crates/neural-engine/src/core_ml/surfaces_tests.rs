use super::check_element_count;

#[test]
fn should_reject_a_buffer_of_the_wrong_length() {
    assert!(check_element_count(4, 4).is_ok());
    assert!(check_element_count(4, 3).is_err());
}
