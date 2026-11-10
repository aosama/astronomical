// Observatory palette tests: console.css promises that every text colour keeps
// at least 4.5:1 on --ground, --surface and --sunk in both appearances, and
// that the dark palette, which CSS forces it to write twice, is the same in
// both places. These tests hold the stylesheet to both promises.

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");

const MINIMUM_TEXT_CONTRAST = 4.5;
const TEXT_TOKENS = ["--ink", "--ink-2", "--ink-3", "--accent-text", "--danger", "--data", "--memory-context", "--ok", "--warn"];
const GROUND_TOKENS = ["--ground", "--surface", "--sunk"];

const consoleStylesheet = fs.readFileSync(path.join(__dirname, "console.css"), "utf8");

function hexTokensOfBlockStarting(blockOpening) {
    const blockStart = consoleStylesheet.indexOf(blockOpening);
    assert.notEqual(blockStart, -1, `console.css must contain ${blockOpening}`);
    const blockEnd = consoleStylesheet.indexOf("}", blockStart);
    const hexByToken = new Map();
    for (const declaration of consoleStylesheet.slice(blockStart, blockEnd).matchAll(/(--[a-z0-9-]+):\s*(#[0-9a-f]{6});/g)) {
        hexByToken.set(declaration[1], declaration[2]);
    }
    return hexByToken;
}

const lightPalette = hexTokensOfBlockStarting(":root {");
const darkPaletteFollowingTheMac = hexTokensOfBlockStarting(':root:not([data-theme="light"]) {');
const darkPaletteChosenByHand = hexTokensOfBlockStarting(':root[data-theme="dark"] {');

function relativeLuminance(hexColour) {
    const linearChannels = [1, 3, 5].map((channelStart) => {
        const channel = parseInt(hexColour.slice(channelStart, channelStart + 2), 16) / 255;
        return channel <= 0.03928 ? channel / 12.92 : ((channel + 0.055) / 1.055) ** 2.4;
    });
    return 0.2126 * linearChannels[0] + 0.7152 * linearChannels[1] + 0.0722 * linearChannels[2];
}

function contrastRatio(firstHexColour, secondHexColour) {
    const luminances = [relativeLuminance(firstHexColour), relativeLuminance(secondHexColour)];
    return (Math.max(...luminances) + 0.05) / (Math.min(...luminances) + 0.05);
}

function pairsBelowMinimum(palette) {
    const failingPairs = [];
    const checkedPairs = [["--accent-ink", "--accent"]];
    for (const textToken of TEXT_TOKENS) {
        for (const groundToken of GROUND_TOKENS) {
            checkedPairs.push([textToken, groundToken]);
        }
    }
    for (const [textToken, groundToken] of checkedPairs) {
        assert.ok(palette.has(textToken) && palette.has(groundToken), `${textToken} and ${groundToken} must both be hex tokens`);
        const ratio = contrastRatio(palette.get(textToken), palette.get(groundToken));
        if (ratio < MINIMUM_TEXT_CONTRAST) {
            failingPairs.push(`${textToken} on ${groundToken}: ${ratio.toFixed(2)}`);
        }
    }
    return failingPairs;
}

test("should_keep_every_text_colour_readable_on_every_ground_in_the_light_palette", () => {
    assert.deepEqual(pairsBelowMinimum(lightPalette), []);
});

test("should_keep_every_text_colour_readable_on_every_ground_in_the_dark_palette", () => {
    assert.deepEqual(pairsBelowMinimum(darkPaletteChosenByHand), []);
});

test("should_write_the_same_dark_palette_for_a_dark_mac_and_for_a_chosen_dark", () => {
    assert.deepEqual([...darkPaletteFollowingTheMac], [...darkPaletteChosenByHand]);
});

test("should_report_the_light_accent_as_too_weak_for_text_on_the_dark_ground", () => {
    // The reason the dark palette has its own lifted accent.
    assert.ok(contrastRatio(lightPalette.get("--accent"), darkPaletteChosenByHand.get("--ground")) < MINIMUM_TEXT_CONTRAST);
});

test("should_measure_the_extremes_of_contrast_as_one_and_twenty_one", () => {
    assert.equal(contrastRatio("#777777", "#777777"), 1);
    assert.ok(Math.abs(contrastRatio("#000000", "#ffffff") - 21) < 0.001);
});
