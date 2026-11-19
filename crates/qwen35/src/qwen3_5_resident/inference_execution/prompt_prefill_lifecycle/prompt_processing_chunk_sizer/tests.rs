use std::time::Duration;

use super::Qwen3_5ResidentPromptProcessingChunkSizer;
use tokio::time::timeout;

#[tokio::test]
async fn should_process_resident_chunks_at_the_fixed_boundary() {
    timeout(Duration::from_secs(5), async {
        let chunk_sizer =
            Qwen3_5ResidentPromptProcessingChunkSizer::for_fixed_prompt_processing_chunk_size_tokens(
                2_048,
            )
            .expect("the resident chunk size should construct");

        assert_eq!(chunk_sizer.next_prompt_processing_chunk_end(0, 5_000), 2_048);
        assert_eq!(chunk_sizer.next_prompt_processing_chunk_end(4_096, 5_000), 5_000);
        assert_eq!(chunk_sizer.prompt_processing_operation_bound_tokens(), 2_048);
    })
    .await
    .expect("resident chunk planning should finish within five seconds");
}

#[tokio::test]
async fn should_reject_a_zero_resident_chunk_size() {
    timeout(Duration::from_secs(5), async {
        assert!(
            Qwen3_5ResidentPromptProcessingChunkSizer::for_fixed_prompt_processing_chunk_size_tokens(
                0
            )
            .is_err()
        );
    })
    .await
    .expect("resident chunk validation should finish within five seconds");
}
