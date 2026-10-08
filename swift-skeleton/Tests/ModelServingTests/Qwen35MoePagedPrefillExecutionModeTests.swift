import Foundation;

import Testing;

import ModelServing;

/// Paged MoE execution must resolve multi-token host routes before each
/// layer executes while the one-token production decode route may defer,
/// port of crates/model-serving/tests/hermetic/paged_route_materialization.rs.
@Suite
final class Qwen35MoePagedPrefillExecutionModeTests {

    @Test
    func shouldResolveMultiTokenRoutesBeforeExecutingEachPagedLayer() {
        let productionMode = Qwen35MoePagedPrefillExecutionMode.productionDefault;

        for prefillTokenCount in [2, 512, 1_024, 2_048, 4_096] {
            #expect(
                !productionMode.shouldDeferHostRouteMaterialization(tokenCount: prefillTokenCount),
                "multi-token prefill must not execute with a holey expert snapshot");
        }
    }

    @Test
    func shouldDeferOnlyTheOneTokenProductionDecodeRoute() {
        #expect(
            Qwen35MoePagedPrefillExecutionMode.productionDefault
                .shouldDeferHostRouteMaterialization(tokenCount: 1));
        #expect(
            !Qwen35MoePagedPrefillExecutionMode.compactPromptDiagnostic
                .shouldDeferHostRouteMaterialization(tokenCount: 1));
        #expect(
            !Qwen35MoePagedPrefillExecutionMode.tokenLocalDiagnostic
                .shouldDeferHostRouteMaterialization(tokenCount: 1));
    }
}
