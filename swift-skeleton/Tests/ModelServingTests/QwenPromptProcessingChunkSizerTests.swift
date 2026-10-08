import Foundation

import Testing

import ModelServing

/// Behavior coverage for deterministic Qwen prompt chunking, port of
/// crates/model-serving/tests/hermetic/qwen_prompt_processing_chunk_sizer.rs.
@Suite
final class QwenPromptProcessingChunkSizerTests {

    @Test
    func should_process_fixed_chunks_and_an_exact_terminal_remainder() throws {
        let chunkSizer: Qwen35PromptProcessingChunkSizer =
            try Qwen35PromptProcessingChunkSizer(fixedPromptProcessingChunkSizeTokens: 2_048)

        #expect(chunkSizer.nextPromptProcessingChunkEnd(
            chunkStartTokenPosition: 0,
            finalPromptEndTokenPositionExclusive: 5_000) == 2_048)
        #expect(chunkSizer.nextPromptProcessingChunkEnd(
            chunkStartTokenPosition: 4_096,
            finalPromptEndTokenPositionExclusive: 5_000) == 5_000)
    }

    @Test
    func should_use_the_ssd_streaming_fixed_size_only_while_experts_are_paged() throws {
        let chunkSizer: Qwen35PromptProcessingChunkSizer =
            try Qwen35PromptProcessingChunkSizer(
                fixedPromptProcessingChunkSizeTokens: 2_048,
                fixedSsdStreamingPromptProcessingChunkSizeTokens: 256)

        #expect(chunkSizer.nextPromptProcessingChunkEndForExpertResidency(
            chunkStartTokenPosition: 0,
            finalPromptEndTokenPositionExclusive: 5_000,
            sparseExpertsArePaged: true) == 256)
        #expect(chunkSizer.nextPromptProcessingChunkEndForExpertResidency(
            chunkStartTokenPosition: 0,
            finalPromptEndTokenPositionExclusive: 5_000,
            sparseExpertsArePaged: false) == 2_048)
    }

    @Test
    func should_bound_the_next_chunk_by_proven_executable_capacity() throws {
        let chunkSizer: Qwen35PromptProcessingChunkSizer =
            try Qwen35PromptProcessingChunkSizer(fixedPromptProcessingChunkSizeTokens: 2_048)

        #expect(chunkSizer.nextPromptProcessingChunkEndWithMaximumExecutableCapacity(
            chunkStartTokenPosition: 0,
            finalPromptEndTokenPositionExclusive: 5_000,
            sparseExpertsArePaged: false,
            maximumExecutableChunkSizeTokens: 512) == 512)
        #expect(
            Qwen35PromptProcessingChunkSizer.nextSmallerExecutableChunkSizeTokens(
                attemptedChunkSizeTokens: 512) == 256)
    }

    @Test
    func should_reject_invalid_resident_and_ssd_streaming_chunk_sizes() throws {
        #expect(throws: Qwen35PromptProcessingChunkSizerError.mustBePositive) {
            try Qwen35PromptProcessingChunkSizer(fixedPromptProcessingChunkSizeTokens: 0)
        }
        #expect(throws: Qwen35PromptProcessingChunkSizerError.mustBePositive) {
            try Qwen35PromptProcessingChunkSizer(
                fixedPromptProcessingChunkSizeTokens: 2_048,
                fixedSsdStreamingPromptProcessingChunkSizeTokens: 0)
        }
        let largerPagedChunkSizer: Qwen35PromptProcessingChunkSizer =
            try Qwen35PromptProcessingChunkSizer(
                fixedPromptProcessingChunkSizeTokens: 2_048,
                fixedSsdStreamingPromptProcessingChunkSizeTokens: 4_096)
        #expect(largerPagedChunkSizer.nextPromptProcessingChunkEndForExpertResidency(
            chunkStartTokenPosition: 0,
            finalPromptEndTokenPositionExclusive: 9_000,
            sparseExpertsArePaged: true) == 4_096)
        #expect(largerPagedChunkSizer.nextPromptProcessingChunkEndForExpertResidency(
            chunkStartTokenPosition: 0,
            finalPromptEndTokenPositionExclusive: 9_000,
            sparseExpertsArePaged: false) == 2_048)
    }

    @Test
    func should_default_paged_chunks_larger_than_resident_chunks() throws {
        let chunkSizer: Qwen35PromptProcessingChunkSizer =
            try Qwen35PromptProcessingChunkSizer(
                fixedPromptProcessingChunkSizeTokens: 2_048,
                fixedSsdStreamingPromptProcessingChunkSizeTokens: 8_192)
        #expect(chunkSizer.nextPromptProcessingChunkEndForExpertResidency(
            chunkStartTokenPosition: 0,
            finalPromptEndTokenPositionExclusive: 20_000,
            sparseExpertsArePaged: true) == 8_192)
        #expect(chunkSizer.nextPromptProcessingChunkEndForExpertResidency(
            chunkStartTokenPosition: 0,
            finalPromptEndTokenPositionExclusive: 20_000,
            sparseExpertsArePaged: false) == 2_048)
    }

    @Test
    func should_fold_a_short_paged_remainder_instead_of_paying_a_second_leftover_stream()
        throws {
        let chunkSizer: Qwen35PromptProcessingChunkSizer =
            try Qwen35PromptProcessingChunkSizer(
                fixedPromptProcessingChunkSizeTokens: 2_048,
                fixedSsdStreamingPromptProcessingChunkSizeTokens: 2_048)

        #expect(chunkSizer.nextPromptProcessingChunkEndForExpertResidency(
            chunkStartTokenPosition: 0,
            finalPromptEndTokenPositionExclusive: 4_401,
            sparseExpertsArePaged: true) == 2_048)
        #expect(chunkSizer.nextPromptProcessingChunkEndForExpertResidency(
            chunkStartTokenPosition: 2_048,
            finalPromptEndTokenPositionExclusive: 4_401,
            sparseExpertsArePaged: true) == 4_401)
        #expect(chunkSizer.nextPromptProcessingChunkEndForExpertResidency(
            chunkStartTokenPosition: 2_048,
            finalPromptEndTokenPositionExclusive: 4_401,
            sparseExpertsArePaged: false) == 4_096)
    }

    @Test
    func should_keep_full_paged_chunks_when_the_remainder_fills_another_configured_chunk()
        throws {
        let chunkSizer: Qwen35PromptProcessingChunkSizer =
            try Qwen35PromptProcessingChunkSizer(
                fixedPromptProcessingChunkSizeTokens: 2_048,
                fixedSsdStreamingPromptProcessingChunkSizeTokens: 2_048)

        #expect(chunkSizer.nextPromptProcessingChunkEndForExpertResidency(
            chunkStartTokenPosition: 0,
            finalPromptEndTokenPositionExclusive: 10_000,
            sparseExpertsArePaged: true) == 2_048)
        #expect(chunkSizer.nextPromptProcessingChunkEndForExpertResidency(
            chunkStartTokenPosition: 2_048,
            finalPromptEndTokenPositionExclusive: 10_000,
            sparseExpertsArePaged: true) == 4_096)
        #expect(chunkSizer.nextPromptProcessingChunkEndForExpertResidency(
            chunkStartTokenPosition: 4_096,
            finalPromptEndTokenPositionExclusive: 10_000,
            sparseExpertsArePaged: true) == 6_144)
        #expect(chunkSizer.nextPromptProcessingChunkEndForExpertResidency(
            chunkStartTokenPosition: 6_144,
            finalPromptEndTokenPositionExclusive: 10_000,
            sparseExpertsArePaged: true) == 10_000)
    }
}
