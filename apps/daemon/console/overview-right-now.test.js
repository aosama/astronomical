// Home's "Right now" card tests: the sentence while a reply is written,
// after one and before any, the second sentence for a streamed, a fitting
// and an unmeasured model, the memory and SSD split, and the hidden card.

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const memoryPageScriptPath = path.join(__dirname, "memory-page.js");
const rightNowScriptPath = path.join(__dirname, "overview-right-now.js");
const GIGABYTE = 1073741824;

// The card reads memory-page.js's helpers, so both scripts load into one
// context the way the page loads them.
function presentationFor(statusDocument, expertStatsDocument) {
    const scriptContext = vm.createContext({ document: { getElementById() { return null; } } });
    vm.runInContext(fs.readFileSync(memoryPageScriptPath, "utf8"), scriptContext, { filename: memoryPageScriptPath });
    vm.runInContext(fs.readFileSync(rightNowScriptPath, "utf8"), scriptContext, { filename: rightNowScriptPath });
    scriptContext.statusDocument = statusDocument;
    scriptContext.expertStatsDocument = expertStatsDocument;
    return JSON.parse(vm.runInContext(
        "JSON.stringify(overviewRightNowPresentation(statusDocument, expertStatsDocument))", scriptContext));
}

function streamingStatusDocument() {
    return {
        ready_model_id: "Ornith-1.5-35B-A3B-OptiQ-4bit-vision-DFLASH2",
        ready_model_size_bytes: 22.9 * GIGABYTE,
        mlx_memory_ceiling_bytes: 8 * GIGABYTE,
        activity: "idle",
        serving_session: { average_generation_tok_per_second: 37.4 },
        mlx_memory_snapshot: { memory_ledger: { core_weight_bytes: 2.4 * GIGABYTE } }
    };
}

function streamingExpertStatsDocument() {
    return {
        residency: { resident_expert_payload_bytes: 4.8 * GIGABYTE, complete_expert_payload_byte_count: 18.25 * GIGABYTE },
        last_reply: { storage_read_expert_counts_by_token: [64, 64], routed_expert_count_per_token: 320 }
    };
}

test("should_say_the_model_is_answering_at_the_live_speed_while_a_reply_is_written", () => {
    const statusDocument = streamingStatusDocument();
    statusDocument.activity = "generating";
    statusDocument.progress = { phase: "generation", processed_tokens: 350, elapsed_ms: 10000 };

    const presentation = presentationFor(statusDocument, streamingExpertStatsDocument());

    assert.equal(presentation.headline, "This 22.9 GB model is answering inside your 8 GB limit, at 35 tokens a second.");
    assert.equal(presentation.modelLine, "Ornith-1.5-35B-A3B-OptiQ-4bit-vision-DFLASH2");
});

test("should_say_the_model_last_answered_at_the_session_speed_when_idle", () => {
    const presentation = presentationFor(streamingStatusDocument(), streamingExpertStatsDocument());

    assert.equal(presentation.headline, "This 22.9 GB model last answered inside your 8 GB limit, at 37 tokens a second.");
    assert.equal(presentation.detail, "80% of what each word needs is already in memory. The rest is read from your SSD as it goes.");
});

test("should_say_the_model_is_loaded_before_its_first_reply", () => {
    const statusDocument = streamingStatusDocument();
    statusDocument.serving_session = null;

    const presentation = presentationFor(statusDocument, { residency: streamingExpertStatsDocument().residency });

    assert.equal(presentation.headline, "This 22.9 GB model is loaded inside your 8 GB limit.");
    assert.equal(presentation.detail, "Astronomical keeps what each word needs most in memory and reads the rest from your SSD.");
});

test("should_say_a_model_smaller_than_the_limit_fits_whole", () => {
    const statusDocument = { ...streamingStatusDocument(), ready_model_size_bytes: 5.6 * GIGABYTE };

    const presentation = presentationFor(statusDocument, null);

    assert.equal(presentation.detail, "The whole model fits in your limit, so nothing has to be read from your SSD.");
    assert.equal(presentation.split, null, "no split is drawn without the expert figures behind it");
});

test("should_split_the_model_between_memory_and_the_ssd", () => {
    const presentation = presentationFor(streamingStatusDocument(), streamingExpertStatsDocument());

    assert.equal(presentation.split.inMemoryText, "In memory · 7.2 GB");
    assert.equal(presentation.split.onSsdText, "On your SSD, read as needed · 15.7 GB");
    assert.equal(presentation.split.inMemoryPercent.toFixed(1), "31.4");
});

test("should_never_report_more_in_memory_than_the_whole_model", () => {
    const expertStatsDocument = streamingExpertStatsDocument();
    expertStatsDocument.residency.resident_expert_payload_bytes = 30 * GIGABYTE;

    const presentation = presentationFor(streamingStatusDocument(), expertStatsDocument);

    assert.equal(presentation.split.inMemoryPercent, 100);
    assert.equal(presentation.split.onSsdText, "On your SSD, read as needed · 0.0 GB");
});

test("should_hide_the_card_without_a_model_a_size_or_a_limit", () => {
    assert.equal(presentationFor(null, null), null);
    assert.equal(presentationFor({ ...streamingStatusDocument(), ready_model_id: null }, null), null);
    assert.equal(presentationFor({ ...streamingStatusDocument(), ready_model_size_bytes: 0 }, null), null);
    assert.equal(presentationFor({ ...streamingStatusDocument(), mlx_memory_ceiling_bytes: 0 }, null), null);
});
