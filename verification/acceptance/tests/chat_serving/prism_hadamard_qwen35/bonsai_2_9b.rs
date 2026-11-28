//! Text and picture replies from Ternary Bonsai 2 9B.
//!
//! The pack is the configured `model_directory`. Each test is one journey:
//! `cargo test-bonsai-2-9b-text` and `cargo test-bonsai-2-9b-picture`.
//! Thinking is off, and the prediction head is left unset.

use serial_test::serial;

use super::small_pack_reply::{answer_red_picture, write_greedy_text_reply};

#[tokio::test(flavor = "multi_thread")]
#[ignore = "loads Ternary Bonsai 2 9B and streams one greedy text reply"]
#[serial]
async fn should_write_a_greedy_text_reply_from_the_bonsai_2_9b() {
    write_greedy_text_reply("9b").await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "loads Ternary Bonsai 2 9B and answers one picture"]
#[serial]
async fn should_answer_a_picture_from_the_bonsai_2_9b() {
    answer_red_picture("9b").await;
}
