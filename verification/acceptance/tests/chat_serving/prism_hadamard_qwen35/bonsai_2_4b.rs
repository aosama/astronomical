//! Text and picture replies from Ternary Bonsai 2 4B.
//!
//! The pack is the configured `model_directory`. Each test is one journey:
//! `cargo test-bonsai-2-4b-text` and `cargo test-bonsai-2-4b-picture`.
//! Thinking is off, and the prediction head is left unset.

use serial_test::serial;

use super::small_pack_reply::{answer_red_picture, write_greedy_text_reply};

#[tokio::test(flavor = "multi_thread")]
#[ignore = "loads Ternary Bonsai 2 4B and streams one greedy text reply"]
#[serial]
async fn should_write_a_greedy_text_reply_from_the_bonsai_2_4b() {
    write_greedy_text_reply("4b").await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "loads Ternary Bonsai 2 4B and answers one picture"]
#[serial]
async fn should_answer_a_picture_from_the_bonsai_2_4b() {
    answer_red_picture("4b").await;
}
