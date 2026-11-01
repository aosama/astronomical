// Contract tests for the Experts view pure renderers.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const EXPERT_GIGABYTE_BYTES = 1073741824;
const expertsRenderScriptPath = path.join(__dirname, "experts-render.js");
const expertsRenderScript = fs.readFileSync(expertsRenderScriptPath, "utf8");

function createExpertsRenderContext() {
    const scriptContext = vm.createContext({ console: { log() {} } });
    vm.runInContext(expertsRenderScript, scriptContext, {
        filename: expertsRenderScriptPath
    });
    return scriptContext;
}

const sampleTopology = [
    {
        layer_index: 2,
        retained_expert_ids: [1, 4],
        retained_payload_bytes: EXPERT_GIGABYTE_BYTES,
        is_complete_layer: false,
        covered_demand: 20
    }
];

test("should_format_gigabytes_on_the_apple_gigabyte", () => {
    const context = createExpertsRenderContext();
    assert.equal(context.formatExpertGigabytes(0), "0 GB");
    assert.equal(context.formatExpertGigabytes(-1), "0 GB");
    assert.equal(context.formatExpertGigabytes(EXPERT_GIGABYTE_BYTES / 2), "0.50 GB");
    assert.equal(context.formatExpertGigabytes(12 * EXPERT_GIGABYTE_BYTES), "12.00 GB");
});

test("should_format_sub_minute_durations_in_tenths", () => {
    const context = createExpertsRenderContext();
    assert.equal(context.formatExpertDuration(0), "0s");
    assert.equal(context.formatExpertDuration(0.04), "0s");
    assert.equal(context.formatExpertDuration(1.24), "1.2s");
    assert.equal(context.formatExpertDuration(42), "42s");
    assert.equal(context.formatExpertDuration(95), "1m 35s");
});

test("should_name_the_three_residency_phrases", () => {
    const context = createExpertsRenderContext();
    assert.equal(context.expertMemoryModePhrase("resident"), "Fully in memory");
    assert.equal(context.expertMemoryModePhrase("paged"), "RAM + SSD streaming");
    assert.equal(
        context.expertMemoryModePhrase("hybrid"),
        "Hot experts in RAM, the rest on the SSD"
    );
    assert.equal(context.expertMemoryModePhrase("other"), "");
});

test("should_write_the_headline_for_each_evidence_case", () => {
    const context = createExpertsRenderContext();
    assert.equal(
        context.expertHeadline({}),
        "Load a sparse model. This page shows which expert weights stay on the SSD."
    );
    assert.match(
        context.expertHeadline({
            modelId: "qwen",
            expertMemoryMode: "resident",
            completeExpertPayloadByteCount: 10 * EXPERT_GIGABYTE_BYTES,
            residentExpertPayloadBytes: 10 * EXPERT_GIGABYTE_BYTES
        }),
        /qwen fits\. Its expert library is in RAM/
    );
    const streamingHeadline = context.expertHeadline({
        modelId: "qwen",
        expertMemoryMode: "hybrid",
        completeExpertPayloadByteCount: 20 * EXPERT_GIGABYTE_BYTES,
        residentExpertPayloadBytes: 8 * EXPERT_GIGABYTE_BYTES,
        totalLayerCount: 40,
        ceilingBytes: 11 * EXPERT_GIGABYTE_BYTES
    });
    assert.match(streamingHeadline, /keeps 20\.00 GB of expert weights/);
    assert.match(streamingHeadline, /8\.00 GB are in RAM across 40 layers/);
    assert.match(streamingHeadline, /12\.00 GB stayed on the SSD/);
    assert.match(streamingHeadline, /layer map shows which layers/);
    assert.match(streamingHeadline, /ceiling is 11\.00 GB/);
    assert.match(
        context.expertHeadline({
            modelId: "qwen",
            expertMemoryMode: "hybrid",
            completeExpertPayloadByteCount: 32 * EXPERT_GIGABYTE_BYTES,
            residentExpertPayloadBytes: 32 * EXPERT_GIGABYTE_BYTES,
            totalLayerCount: 40
        }),
        /qwen has its expert library in RAM\. 32\.00 GB across 40 layers\./
    );
    const unknownHeadline = context.expertHeadline({
        modelId: "qwen",
        expertMemoryMode: "paged",
        completeExpertPayloadByteCount: 0,
        residentExpertPayloadBytes: EXPERT_GIGABYTE_BYTES,
        ceilingBytes: 11 * EXPERT_GIGABYTE_BYTES
    });
    assert.match(unknownHeadline, /qwen is loaded\. RAM \+ SSD streaming\./);
    assert.doesNotMatch(unknownHeadline, /stayed on the SSD/);
});

test("should_hide_the_ssd_remainder_when_the_library_size_is_unknown", () => {
    const context = createExpertsRenderContext();
    assert.equal(context.expertPayloadLeftOnSsdBytes(0, 5), null);
    assert.equal(context.expertPayloadLeftOnSsdBytes(10, 10), 0);
    assert.equal(context.expertPayloadLeftOnSsdBytes(10, 11), 0);
    assert.equal(context.expertPayloadLeftOnSsdBytes(10, 4), 6);
});

test("should_include_streaming_layers_absent_from_the_topology", () => {
    const context = createExpertsRenderContext();
    const layerRows = context.expertLayerRows(sampleTopology, 3, 8);
    assert.equal(layerRows.length, 3);
    assert.equal(layerRows[0].layerIndex, 0);
    assert.equal(layerRows[0].isStreaming, true);
    assert.equal(layerRows[0].retainedCount, 0);
    assert.equal(layerRows[1].layerIndex, 1);
    assert.equal(layerRows[1].isStreaming, true);
    assert.equal(layerRows[2].layerIndex, 2);
    assert.equal(layerRows[2].isStreaming, false);
    assert.equal(layerRows[2].expertCapacity, 8);
    assert.equal(layerRows[2].retainedCount, 2);
    assert.equal(context.expertLayerRows([], 0, 256).length, 0);
    assert.equal(context.expertLayerRows(null, 2, 0)[0].expertCapacity, 0);
});

test("should_summarize_which_layers_are_in_ram_and_which_are_on_the_ssd", () => {
    const context = createExpertsRenderContext();
    assert.equal(
        context.expertLayerOverview([]),
        "Layers appear here once a sparse model is loaded."
    );
    assert.equal(context.expertLayerOverview(null), "Layers appear here once a sparse model is loaded.");
    const allInRam = context.expertLayerRows([
        { layer_index: 0, retained_expert_ids: [0, 1], is_complete_layer: true, covered_demand: 1 },
        { layer_index: 1, retained_expert_ids: [0, 1], is_complete_layer: true, covered_demand: 9 }
    ], 2, 2);
    assert.equal(context.expertLayerOverview(allInRam), "All 2 layers are in RAM.");
    const allOnSsd = context.expertLayerRows([], 2, 4);
    assert.equal(context.expertLayerOverview(allOnSsd), "All 2 layers are still on the SSD.");
    const oneSplit = context.expertLayerRows(sampleTopology, 3, 8);
    assert.equal(
        context.expertLayerOverview(oneSplit),
        "Layer 2 still has 6 of 8 experts on the SSD. Still entirely on the SSD: layer 0, layer 1."
    );
    const mostlyInRam = context.expertLayerRows([
        { layer_index: 0, retained_expert_ids: [0, 1], is_complete_layer: true },
        { layer_index: 1, retained_expert_ids: [0], is_complete_layer: false }
    ], 2, 2);
    assert.equal(
        context.expertLayerOverview(mostlyInRam),
        "1 of 2 layers is entirely in RAM. Layer 1 still has 1 of 2 experts on the SSD."
    );
    assert.equal(context.expertLayerOnSsdCount(oneSplit[2]), 6);
    assert.equal(context.expertLayerOnSsdCount(oneSplit[0]), 8);
    assert.equal(context.expertLayerResidence(oneSplit[2]), "split");
    assert.equal(context.expertLayerResidence({ expertCapacity: 0, retainedCount: 0, isStreaming: true }), "ssd");
});

test("should_describe_the_selected_layer_as_ram_and_ssd_counts", () => {
    const context = createExpertsRenderContext();
    const layerRows = context.expertLayerRows(sampleTopology, 3, 8);
    assert.equal(context.expertLayerDetailSentence(null), "");
    assert.equal(
        context.expertLayerDetailSentence(layerRows[0]),
        "Layer 0 · all 8 experts on the SSD"
    );
    assert.equal(
        context.expertLayerDetailSentence(layerRows[2]),
        "Layer 2 · 2 in RAM · 6 on the SSD"
    );
    const fullLayer = context.expertLayerRows([
        { layer_index: 0, retained_expert_ids: [0, 1, 2, 3], is_complete_layer: true }
    ], 1, 4)[0];
    assert.equal(context.expertLayerDetailSentence(fullLayer), "Layer 0 · all 4 experts in RAM");
    assert.equal(
        context.expertLayerDetailSentence({ layerIndex: 4, expertCapacity: 0, retainedCount: 0, isStreaming: true }),
        "Layer 4 · on the SSD"
    );
});

test("should_keep_session_piles_in_bytes_and_separate_complete_layer_hits", () => {
    const context = createExpertsRenderContext();
    const piles = context.expertSessionPiles({
        mandatory_prefill_source_bytes: 80,
        mandatory_decode_source_bytes: 9,
        avoided_complete_layer_source_bytes: 120,
        retained_route_assignment_count: 3000
    });
    assert.equal(piles.promptBytes, 80);
    assert.equal(piles.generationBytes, 9);
    assert.equal(piles.avoidedCompleteLayerBytes, 120);
    const emptyPiles = context.expertSessionPiles(null);
    assert.equal(emptyPiles.promptBytes, 0);
    assert.equal(emptyPiles.generationBytes, 0);
    assert.equal(emptyPiles.avoidedCompleteLayerBytes, 0);
});

test("should_describe_generation_warmth_and_mixed_routes", () => {
    const context = createExpertsRenderContext();
    assert.equal(
        context.expertGenerationSentence(null),
        "No generated tokens have been classified yet."
    );
    assert.match(
        context.expertGenerationSentence({
            hot_route_fully_covered_token_count: 700,
            hot_route_partially_covered_token_count: 200,
            hot_route_missed_token_count: 100
        }),
        /700 generated tokens had every routed expert already warm\. 300 still touched the SSD\./
    );
    assert.match(
        context.expertMixedRouteSentence({ hot_route_partially_covered_token_count: 200 }),
        /200 of those tokens were mixed/
    );
    assert.equal(context.expertMixedRouteSentence({ hot_route_partially_covered_token_count: 0 }), "");
});

test("should_render_mode_transitions_in_product_phrases", () => {
    const context = createExpertsRenderContext();
    const transitionLine = context.modeTransitionLine({
        observed_at_unix_millis: 0,
        previous_mode: "paged",
        next_mode: "hybrid"
    });
    assert.match(transitionLine, /RAM \+ SSD streaming → Hot experts in RAM, the rest on the SSD/);
    const firstTransitionLine = context.modeTransitionLine({
        observed_at_unix_millis: 0,
        previous_mode: null,
        next_mode: "resident"
    });
    assert.match(firstTransitionLine, /Start → Fully in memory/);
    assert.equal(context.modeTransitionLine(null), "");
});

test("should_explain_the_estimate_from_nanoseconds", () => {
    const context = createExpertsRenderContext();
    const liveLine = context.expertMethodLiveLine({
        storage_route_assignment_count: 100,
        expert_source_read_elapsed_nanoseconds: 20_000_000_000,
        retained_route_assignment_count: 1000,
        storage_time_saved_estimate_nanoseconds: 1_500_000_000
    });
    assert.match(liveLine, /200\.0 ms/);
    assert.match(liveLine, /1,000/);
    assert.match(liveLine, /1\.5s/);
    assert.equal(
        context.expertMethodLiveLine({
            storage_route_assignment_count: 0,
            expert_source_read_elapsed_nanoseconds: 0
        }),
        "No storage assignment has been timed yet, so no estimate is published."
    );
    assert.equal(
        context.expertMethodLiveLine(null),
        "No storage assignment has been timed yet, so no estimate is published."
    );
});

test("should_reconcile_memory_segments_without_exceeding_active_memory", () => {
    const context = createExpertsRenderContext();
    const segments = context.reconciledExpertMemorySegments(
        {
            active_memory_bytes: 10,
            expert_payload_bytes: 6,
            model_core_payload_bytes: 3,
            context_state_payload_bytes: 4
        },
        16
    );
    assert.equal(segments.expertBytes, 6);
    assert.equal(segments.modelCoreBytes, 3);
    assert.equal(segments.contextStateBytes, 1);
    assert.equal(segments.runtimeWorkBytes, 0);
    assert.equal(segments.availableBytes, 6);
    assert.equal(context.expertBarPercent(0, 0), 0);
    assert.equal(context.expertLayerFillPercent(4, 0), 0);
});

test("should_copy_the_daemon_placement_sentences", () => {
    const context = createExpertsRenderContext();
    const missingPlacement = context.expertPlacementCopy({});
    assert.equal(missingPlacement.stockSentence, "");
    assert.equal(missingPlacement.replySentence, "");
    const placement = context.expertPlacementCopy({
        expert_placement: {
            stock_sentence: "All 40 layers are in RAM.",
            reply_sentence: "This reply has read experts from the SSD."
        }
    });
    assert.equal(placement.stockSentence, "All 40 layers are in RAM.");
    assert.equal(placement.replySentence, "This reply has read experts from the SSD.");
    const blankReply = context.expertPlacementCopy({
        expert_placement: { stock_sentence: "Fully in memory." }
    });
    assert.equal(blankReply.stockSentence, "Fully in memory.");
    assert.equal(blankReply.replySentence, "");
});

test("should_move_layer_columns_from_live_counts_and_wait_for_identities", () => {
    const context = createExpertsRenderContext();
    const layerRows = context.expertLayerRowsFromLiveCounts(
        [0, 2, 8],
        [{
            layer_index: 1,
            retained_expert_ids: [3, 9],
            retained_payload_bytes: 10,
            covered_demand: 4,
            is_complete_layer: false
        }],
        8
    );
    assert.equal(layerRows.length, 3);
    assert.equal(layerRows[0].isStreaming, true);
    assert.equal(layerRows[0].retainedCount, 0);
    assert.equal(layerRows[0].identitiesDeferred, false);
    assert.equal(layerRows[1].retainedCount, 2);
    assert.equal(layerRows[1].retainedExpertIds[0], 3);
    assert.equal(layerRows[1].retainedExpertIds[1], 9);
    assert.equal(layerRows[1].identitiesDeferred, false);
    assert.equal(layerRows[1].isCompleteLayer, false);
    assert.equal(layerRows[2].retainedCount, 8);
    assert.equal(layerRows[2].isCompleteLayer, true);
    assert.equal(layerRows[2].identitiesDeferred, true);
    assert.equal(layerRows[2].retainedExpertIds.length, 0);
    assert.equal(
        context.expertLayerIdentitiesNote(layerRows[2]),
        " Individual experts for this layer appear when the reply finishes."
    );
    assert.equal(context.expertLayerIdentitiesNote(layerRows[1]), "");
    assert.equal(
        context.expertLayerCaption(layerRows[2]),
        "Layer 2 · all 8 experts in RAM Individual experts for this layer appear when the reply finishes."
    );
    const fallbackRows = context.expertLayerRowsForObservatory(null, [], 2, 4);
    assert.equal(fallbackRows.length, 2);
    assert.equal(fallbackRows[0].isStreaming, true);
    const nonFiniteRow = context.expertLayerRowsFromLiveCounts([Number.NaN, -1], [], 4);
    assert.equal(nonFiniteRow[0].retainedCount, 0);
    assert.equal(nonFiniteRow[1].retainedCount, 0);
});

test("should_hold_session_totals_open_only_while_a_reply_is_running", () => {
    const context = createExpertsRenderContext();
    assert.equal(
        context.expertSessionOpenSentence("generating"),
        "These totals close when this reply finishes."
    );
    assert.equal(
        context.expertSessionOpenSentence("prompt_processing"),
        "These totals close when this reply finishes."
    );
    assert.equal(
        context.expertSessionOpenSentence("generation_preparation"),
        "These totals close when this reply finishes."
    );
    assert.equal(context.expertSessionOpenSentence("idle"), "");
    assert.equal(context.expertSessionOpenSentence("image_generation"), "");
    assert.equal(context.expertSessionOpenSentence(""), "");
    assert.equal(context.expertReplyIsOpen("generating"), true);
    assert.equal(context.expertReplyIsOpen("idle"), false);
});
