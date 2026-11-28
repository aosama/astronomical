// Copy as image picture tests: the sentence at the top and its article, the
// three figures, the footer, and that the picture is drawn with every text
// and every token column in place.

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const memoryPageScriptPath = path.join(__dirname, "memory-page.js");
const memoryPageImageScriptPath = path.join(__dirname, "memory-page-image.js");
const GIGABYTE = 1073741824;

// The picture reads memory-page.js's helpers, so both scripts load into one
// context the way the page loads them.
function createMemoryPageImageContext() {
    const scriptContext = vm.createContext({ document: { getElementById() { return null; } } });
    vm.runInContext(fs.readFileSync(memoryPageScriptPath, "utf8"), scriptContext, { filename: memoryPageScriptPath });
    vm.runInContext(fs.readFileSync(memoryPageImageScriptPath, "utf8"), scriptContext, { filename: memoryPageImageScriptPath });
    return scriptContext;
}

function evaluate(expression, bindings) {
    const scriptContext = createMemoryPageImageContext();
    Object.assign(scriptContext, bindings);
    return JSON.parse(vm.runInContext("JSON.stringify(" + expression + ")", scriptContext));
}

function streamingStatusDocument() {
    return {
        ready_model_id: "Ornith-1.5-35B-A3B-OptiQ-4bit-vision-DFLASH2",
        ready_model_size_bytes: 22.9 * GIGABYTE,
        mlx_memory_ceiling_bytes: 8 * GIGABYTE,
        serving_session: { average_generation_tok_per_second: 37.2 },
        application: { version: "0.2.138" }
    };
}

function streamingExpertStatsDocument() {
    return { last_reply: { storage_read_expert_counts_by_token: [64, 66], routed_expert_count_per_token: 320 } };
}

// Records every text and rectangle drawn.
function recordingDrawingContext() {
    return {
        drawnTexts: [],
        filledRectangles: [],
        fillRect(rectangleLeft, rectangleTop, rectangleWidth, rectangleHeight) {
            this.filledRectangles.push({ rectangleTop, rectangleHeight, fillStyle: this.fillStyle });
        },
        fillText(drawnText) { this.drawnTexts.push(drawnText); },
        measureText(measuredText) { return { width: measuredText.length * 10 }; }
    };
}

test("should_say_an_before_limits_spoken_with_a_vowel_and_a_before_the_rest", () => {
    const limitPhrases = evaluate("[8, 11, 16, 18, 24, 80, 128].map(memoryPageImageLimitPhrase)", {});

    assert.deepEqual(limitPhrases, [
        "an 8 GB limit", "an 11 GB limit", "a 16 GB limit", "an 18 GB limit",
        "a 24 GB limit", "an 80 GB limit", "a 128 GB limit"
    ]);
});

test("should_state_the_model_size_inside_the_limit_at_the_top", () => {
    const headline = evaluate("memoryPageImageHeadline(statusDocument, 'page headline')", { statusDocument: streamingStatusDocument() });

    assert.equal(headline, "A 22.9 GB model in an 8 GB limit");
});

test("should_fall_back_to_the_page_headline_without_a_size_or_a_limit", () => {
    const statusWithoutLimit = streamingStatusDocument();
    delete statusWithoutLimit.mlx_memory_ceiling_bytes;

    assert.equal(evaluate("memoryPageImageHeadline(statusDocument, 'page headline')", { statusDocument: statusWithoutLimit }), "page headline");
    assert.equal(evaluate("memoryPageImageHeadline(null, 'page headline')", {}), "page headline");
});

test("should_show_the_speed_the_share_in_memory_and_no_cloud_accounts", () => {
    const figures = evaluate("memoryPageImageFigures(statusDocument, expertStatsDocument)",
        { statusDocument: streamingStatusDocument(), expertStatsDocument: streamingExpertStatsDocument() });

    assert.deepEqual(figures.map((figure) => figure.value), ["37", "80%", "0"]);
    assert.deepEqual(figures.map((figure) => figure.isAccent), [true, false, false]);
});

test("should_show_a_dash_for_a_figure_not_known_yet", () => {
    const figures = evaluate("memoryPageImageFigures({}, null)", {});

    assert.deepEqual(figures.map((figure) => figure.value), ["—", "—", "0"]);
});

test("should_name_the_model_and_the_app_version_in_the_footer", () => {
    assert.equal(evaluate("memoryPageImageFooter(statusDocument)", { statusDocument: streamingStatusDocument() }),
        "Ornith-1.5-35B-A3B-OptiQ-4bit-vision-DFLASH2 · Astronomical 0.2.138");
    assert.equal(evaluate("memoryPageImageFooter(null)", {}), "No model loaded · Astronomical");
});

test("should_draw_the_headline_the_figures_the_footer_and_one_column_per_token", () => {
    const scriptContext = createMemoryPageImageContext();
    const drawingContext = recordingDrawingContext();
    scriptContext.imageCanvas = { getContext: () => drawingContext };
    scriptContext.statusDocument = streamingStatusDocument();
    scriptContext.expertStatsDocument = streamingExpertStatsDocument();
    vm.runInContext("drawMemoryPageImage(imageCanvas, statusDocument, expertStatsDocument, "
        + "{ background: 'bg', ink: 'ink', inkDim: 'dim', accent: 'violet', memoryServed: 'blue', storageRead: 'violet' })", scriptContext);

    assert.deepEqual(drawingContext.drawnTexts, [
        "Astronomical", "A 22.9 GB model in an 8 GB limit",
        "37", "tokens a second", "80%", "of each token's experts already in memory", "0", "cloud accounts",
        "Ornith-1.5-35B-A3B-OptiQ-4bit-vision-DFLASH2 · Astronomical 0.2.138", "getastronomical.com"
    ]);
    const tokenRectangles = drawingContext.filledRectangles.filter((rectangle) => rectangle.rectangleTop >= 350);
    assert.equal(tokenRectangles.length, 4, "two tokens, each drawn as its memory part and its SSD part");
    assert.equal(tokenRectangles[1].fillStyle, "violet");
    assert.equal(tokenRectangles[1].rectangleHeight, 30, "64 of 320 experts read from the SSD fill a fifth of 150 pixels");
});
