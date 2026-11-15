// Memory page tests: the headline, the Mac's memory bar and its wire limit,
// the four facts, and the per-token strip.

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const memoryPageScriptPath = path.join(__dirname, "memory-page.js");
const memoryPageScript = fs.readFileSync(memoryPageScriptPath, "utf8");
const GIGABYTE = 1073741824;

function createMemoryPageContext() {
    const scriptContext = vm.createContext({ document: { getElementById() { return null; } } });
    vm.runInContext(memoryPageScript, scriptContext, { filename: memoryPageScriptPath });
    return scriptContext;
}

function presentationFor(statusDocument, expertStatsDocument) {
    const scriptContext = createMemoryPageContext();
    scriptContext.statusDocument = statusDocument;
    scriptContext.expertStatsDocument = expertStatsDocument;
    return JSON.parse(vm.runInContext(
        "JSON.stringify(memoryPagePresentation(statusDocument, expertStatsDocument))", scriptContext));
}

function streamingStatusDocument() {
    return {
        ready_model_id: "sparse-122b",
        ready_model_size_bytes: 37.8 * GIGABYTE,
        machine_mlx_memory_ceiling_bytes: 48 * GIGABYTE,
        serving_session: { average_generation_tok_per_second: 36.4 },
        mlx_memory_snapshot: {
            active_memory_bytes: 30 * GIGABYTE,
            mlx_buffer_cache_bytes: GIGABYTE,
            memory_ledger: {
                core_weight_bytes: 6 * GIGABYTE,
                held_kept_weight_bytes: 22 * GIGABYTE,
                held_context_bytes: GIGABYTE,
                system_wired_bytes: 36 * GIGABYTE,
                system_wire_limit_bytes: 38 * GIGABYTE
            }
        }
    };
}

function streamingExpertStatsDocument() {
    return {
        residency: { resident_expert_payload_bytes: 22 * GIGABYTE, complete_expert_payload_byte_count: 32 * GIGABYTE },
        last_reply: {
            storage_read_expert_counts_by_token: [40, 0, 0, 10],
            mlx_buffer_cache_sampled_token_count: 4,
            mlx_buffer_cache_empty_token_count: 1,
            routed_expert_count_per_token: 384
        }
    };
}

test("should_state_the_model_the_mac_and_the_speed_in_the_headline", () => {
    const presentation = presentationFor(streamingStatusDocument(), streamingExpertStatsDocument());

    assert.equal(presentation.headline, "37.8 GB model, 48 GB Mac, 36 tokens per second");
    assert.equal(presentation.modelLine, "sparse-122b");
});

test("should_split_the_mac_memory_bar_and_mark_the_wire_limit", () => {
    const presentation = presentationFor(streamingStatusDocument(), streamingExpertStatsDocument());

    const partTexts = Object.fromEntries(presentation.bar.parts.map((barPart) => [barPart.key, barPart.text]));
    assert.deepEqual(partTexts, {
        other: "5.0 GB", fixed: "6.0 GB", experts: "22.0 GB", conversation: "1.0 GB", working: "2.0 GB"
    });
    assert.equal(presentation.bar.parts[2].widthPercent.toFixed(2), "45.83");
    assert.equal(presentation.bar.wireLimitPercent.toFixed(2), "79.17");
    assert.equal(presentation.bar.wireLimitText, "macOS wire limit · 38.0 GB");
    assert.deepEqual(presentation.bar.parts.filter((barPart) => barPart.isEmpty).map((barPart) => barPart.key), []);
});

test("should_report_the_four_facts_of_the_last_reply", () => {
    const presentation = presentationFor(streamingStatusDocument(), streamingExpertStatsDocument());

    assert.deepEqual(presentation.facts, [
        { label: "Under the wire limit", value: "2.0 GB" },
        { label: "Expert library in memory", value: "69%" },
        { label: "From the SSD per token", value: "13 of 384 experts" },
        { label: "Buffers reused", value: "3 of 4" }
    ]);
});

test("should_draw_each_token_to_the_scale_of_the_experts_it_routed_to", () => {
    const presentation = presentationFor(streamingStatusDocument(), streamingExpertStatsDocument());

    assert.deepEqual(presentation.tokens.map((token) => token.storageSharePercent.toFixed(2)),
        ["10.42", "0.00", "0.00", "2.60"]);
    assert.equal(presentation.tokenNote,
        "Each token used 384 experts. On average 97% of them were already in memory; the rest were read from the SSD.");
});

test("should_scale_tokens_to_the_largest_read_when_the_routed_width_is_unknown", () => {
    const expertStatsDocument = streamingExpertStatsDocument();
    delete expertStatsDocument.last_reply.routed_expert_count_per_token;

    const presentation = presentationFor(streamingStatusDocument(), expertStatsDocument);

    assert.equal(presentation.tokens[0].storageSharePercent, 100);
    assert.equal(presentation.facts[2].value, "13 experts");
    assert.equal(presentation.tokenNote, "Finish a reply on a model that streams experts to see every token here.");
});

test("should_say_so_when_wired_memory_is_over_the_limit", () => {
    const statusDocument = streamingStatusDocument();
    statusDocument.mlx_memory_snapshot.memory_ledger.system_wired_bytes = 39.5 * GIGABYTE;

    const presentation = presentationFor(statusDocument, null);

    assert.deepEqual(presentation.facts[0], { label: "Over the wire limit", value: "1.5 GB" });
});

test("should_leave_out_the_wire_limit_marker_before_macos_was_read", () => {
    const statusDocument = streamingStatusDocument();
    statusDocument.mlx_memory_snapshot.memory_ledger.system_wired_bytes = 0;
    statusDocument.mlx_memory_snapshot.memory_ledger.system_wire_limit_bytes = 0;

    const presentation = presentationFor(statusDocument, null);

    assert.equal(presentation.bar.wireLimitPercent, null);
    assert.equal(presentation.bar.parts[0].text, "0.0 GB");
    assert.equal(presentation.bar.parts[0].isEmpty, true);
    assert.equal(presentation.facts[0].value, "—");
});

test("should_show_an_invitation_and_no_bar_when_no_model_is_loaded", () => {
    const presentation = presentationFor({ machine_mlx_memory_ceiling_bytes: 48 * GIGABYTE }, null);

    assert.equal(presentation.modelLine, "No model loaded");
    assert.equal(presentation.headline, "48 GB Mac");
    assert.equal(presentation.bar, null);
    assert.deepEqual(presentation.tokens, []);
    assert.deepEqual(presentation.facts.map((fact) => fact.value), ["—", "—", "—", "—"]);
});

test("should_survive_missing_documents", () => {
    const presentation = presentationFor(null, null);

    assert.equal(presentation.headline, "Load a model to see how it fits in this Mac's memory");
    assert.equal(presentation.bar, null);
});
