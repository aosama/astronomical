// Memory page picture tests: where the legend entries go, and that every
// bar part is drawn even when the legend leaves it out.

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const memoryPageImageScriptPath = path.join(__dirname, "memory-page-image.js");
const memoryPageImageScript = fs.readFileSync(memoryPageImageScriptPath, "utf8");

function createMemoryPageImageContext() {
    const scriptContext = vm.createContext({});
    vm.runInContext(memoryPageImageScript, scriptContext, { filename: memoryPageImageScriptPath });
    return scriptContext;
}

function legendPlacementsFor(legendTextWidths) {
    const scriptContext = createMemoryPageImageContext();
    scriptContext.legendTextWidths = legendTextWidths;
    return JSON.parse(vm.runInContext(
        "JSON.stringify(memoryPageImageLegendPlacements(legendTextWidths, 300))", scriptContext));
}

// Records every rectangle drawn so a test can count the bar's parts.
function recordingDrawingContext() {
    return {
        filledRectangles: [],
        fillRect(rectangleLeft, rectangleTop, rectangleWidth, rectangleHeight) {
            this.filledRectangles.push({ rectangleTop, rectangleWidth, fillStyle: this.fillStyle });
        },
        fillText() {},
        measureText(measuredText) { return { width: measuredText.length * 10 }; }
    };
}

test("should_place_legend_entries_on_one_line_when_they_fit", () => {
    const legendPlacements = legendPlacementsFor([200, 200, 200]);

    assert.deepEqual(legendPlacements, [{ left: 64, top: 300 }, { left: 316, top: 300 }, { left: 568, top: 300 }]);
});

test("should_start_a_new_line_when_the_next_entry_would_cross_the_margin", () => {
    const legendPlacements = legendPlacementsFor([300, 300, 300, 300]);

    assert.deepEqual(legendPlacements.map((legendPlacement) => legendPlacement.top), [300, 300, 300, 330]);
    assert.equal(legendPlacements[3].left, 64);
});

test("should_keep_an_entry_wider_than_the_picture_on_its_own_line_start", () => {
    const legendPlacements = legendPlacementsFor([2000, 100]);

    assert.deepEqual(legendPlacements, [{ left: 64, top: 300 }, { left: 64, top: 330 }]);
});

test("should_place_nothing_for_an_empty_legend", () => {
    assert.deepEqual(legendPlacementsFor([]), []);
});

test("should_draw_every_bar_part_on_one_row_even_when_the_legend_wraps_or_skips_it", () => {
    const scriptContext = createMemoryPageImageContext();
    const drawingContext = recordingDrawingContext();
    scriptContext.drawingContext = drawingContext;
    scriptContext.barPresentation = {
        parts: ["other", "fixed", "experts", "conversation", "working"].map((partKey) => ({
            key: partKey,
            label: "A long legend label for " + partKey,
            text: partKey === "conversation" ? "0.0 GB" : "9.9 GB",
            isEmpty: partKey === "conversation",
            widthPercent: 10
        }))
    };
    scriptContext.palette = {
        track: "track", inkDim: "ink",
        parts: { other: "other", fixed: "fixed", experts: "experts", conversation: "conversation", working: "working" }
    };

    vm.runInContext("drawMemoryPageImageBar(drawingContext, barPresentation, palette, 228)", scriptContext);

    const barRectangles = drawingContext.filledRectangles.filter((filledRectangle) => filledRectangle.rectangleTop === 228);
    assert.deepEqual(barRectangles.map((barRectangle) => barRectangle.fillStyle),
        ["track", "other", "fixed", "experts", "conversation", "working"]);
    const legendSwatchCount = drawingContext.filledRectangles.length - barRectangles.length;
    assert.equal(legendSwatchCount, 4);
});
