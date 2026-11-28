//! Unit tests for keeping float32 matrix multiplies float32 on every backend.
//! The numeric proof is the whole-model gradient journey
//! (`cargo test-laya-whole-model-gradient`), which fails on an Apple GPU with
//! neural accelerators when TF32 is allowed.

use super::keep_float32_matrix_multiplies_float32;

#[test]
fn should_give_the_same_answer_on_every_call() {
    keep_float32_matrix_multiplies_float32().expect("the first call succeeds");
    keep_float32_matrix_multiplies_float32().expect("a repeated call succeeds");
}
