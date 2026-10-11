use std::time::Duration;

use astronomical_model_serving::Qwen3_5StreamingPromptProcessingChunkSizer;
use tokio::time::timeout;

#[tokio::test]
async fn should_process_streaming_chunks_and_an_exact_terminal_remainder() {
    timeout(Duration::from_secs(5), async {
        let chunk_sizer =
            Qwen3_5StreamingPromptProcessingChunkSizer::for_ssd_streaming_chunk_size_tokens(2_048)
                .expect("the streaming chunk size should construct");

        assert_eq!(
            chunk_sizer.next_prompt_processing_chunk_end(0, 5_000),
            2_048
        );
        assert_eq!(
            chunk_sizer.next_prompt_processing_chunk_end(4_096, 5_000),
            5_000
        );
    })
    .await
    .expect("streaming chunk planning should finish within five seconds");
}

#[tokio::test]
async fn should_bound_the_next_chunk_by_proven_executable_capacity() {
    timeout(Duration::from_secs(5), async {
        let chunk_sizer =
            Qwen3_5StreamingPromptProcessingChunkSizer::for_ssd_streaming_chunk_size_tokens(2_048)
                .expect("the streaming chunk size should construct");

        assert_eq!(
            chunk_sizer
                .next_prompt_processing_chunk_end_with_maximum_executable_capacity(0, 5_000, 512,),
            512
        );
        assert_eq!(
            Qwen3_5StreamingPromptProcessingChunkSizer::next_smaller_executable_chunk_size_tokens(
                512
            ),
            Some(256)
        );
    })
    .await
    .expect("streaming capacity planning should finish within five seconds");
}

#[tokio::test]
async fn should_reject_a_zero_streaming_chunk_size() {
    timeout(Duration::from_secs(5), async {
        assert!(
            Qwen3_5StreamingPromptProcessingChunkSizer::for_ssd_streaming_chunk_size_tokens(0)
                .is_err()
        );
    })
    .await
    .expect("streaming chunk validation should finish within five seconds");
}

#[tokio::test]
async fn should_use_the_streaming_chunk_size_for_admission_and_execution() {
    timeout(Duration::from_secs(5), async {
        let chunk_sizer =
            Qwen3_5StreamingPromptProcessingChunkSizer::for_ssd_streaming_chunk_size_tokens(4_096)
                .expect("the larger streaming chunk should construct");

        assert_eq!(
            chunk_sizer.prompt_processing_operation_bound_tokens(),
            4_096
        );
        assert_eq!(
            chunk_sizer.maximum_prompt_processing_chunk_size_tokens(),
            4_096
        );
    })
    .await
    .expect("streaming admission planning should finish within five seconds");
}

#[tokio::test]
async fn should_fold_a_short_streaming_remainder_into_the_current_forward() {
    timeout(Duration::from_secs(5), async {
        let chunk_sizer =
            Qwen3_5StreamingPromptProcessingChunkSizer::for_ssd_streaming_chunk_size_tokens(2_048)
                .expect("the streaming chunk size should construct");

        assert_eq!(
            chunk_sizer.next_prompt_processing_chunk_end(0, 4_401),
            2_048
        );
        assert_eq!(
            chunk_sizer.next_prompt_processing_chunk_end(2_048, 4_401),
            4_401
        );
    })
    .await
    .expect("streaming tail planning should finish within five seconds");
}

#[tokio::test]
async fn should_keep_full_streaming_chunks_when_the_remainder_fills_another_chunk() {
    timeout(Duration::from_secs(5), async {
        let chunk_sizer =
            Qwen3_5StreamingPromptProcessingChunkSizer::for_ssd_streaming_chunk_size_tokens(2_048)
                .expect("the streaming chunk size should construct");

        assert_eq!(
            chunk_sizer.next_prompt_processing_chunk_end(0, 10_000),
            2_048
        );
        assert_eq!(
            chunk_sizer.next_prompt_processing_chunk_end(2_048, 10_000),
            4_096
        );
        assert_eq!(
            chunk_sizer.next_prompt_processing_chunk_end(4_096, 10_000),
            6_144
        );
        assert_eq!(
            chunk_sizer.next_prompt_processing_chunk_end(6_144, 10_000),
            10_000
        );
    })
    .await
    .expect("streaming chunk planning should finish within five seconds");
}
