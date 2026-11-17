use super::*;

#[test]
fn should_write_one_in_the_singular() {
    assert_eq!(counted_noun(1, "example", "examples"), "1 example");
}

#[test]
fn should_write_zero_and_many_in_the_plural() {
    assert_eq!(counted_noun(0, "example", "examples"), "0 examples");
    assert_eq!(
        counted_noun(2, "different text", "different texts"),
        "2 different texts"
    );
}
