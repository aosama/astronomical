import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { test } from "node:test";

/**
 * tokens.css promises that text holds at least 4.5:1 on every ground in both
 * appearances. These tests hold the stylesheet to that, so a later colour
 * tweak that breaks reading contrast fails here instead of on someone's screen.
 */

const MINIMUM_TEXT_CONTRAST = 4.5;
const DARK_APPEARANCE_MARKER = "@media (prefers-color-scheme: dark)";
const TEXT_TOKENS = ["--ink", "--ink-2", "--ink-3", "--accent-text", "--danger"];
const GROUND_TOKENS = ["--ground", "--sidebar", "--surface", "--sunk"];

const tokensStylesheet = fs.readFileSync(path.join(__dirname, "..", "src", "tokens.css"), "utf8");
const darkAppearanceStart = tokensStylesheet.indexOf(DARK_APPEARANCE_MARKER);
const lightPalette = hexTokensIn(tokensStylesheet.slice(0, darkAppearanceStart));
const darkPalette = hexTokensIn(tokensStylesheet.slice(darkAppearanceStart));

function hexTokensIn(stylesheetPart: string): Map<string, string> {
  const hexByToken = new Map<string, string>();
  for (const declaration of stylesheetPart.matchAll(/(--[a-z0-9-]+):\s*(#[0-9a-f]{6});/g)) {
    hexByToken.set(declaration[1]!, declaration[2]!);
  }
  return hexByToken;
}

function relativeLuminance(hexColour: string): number {
  const linearChannels = [1, 3, 5].map((channelStart) => {
    const channel = parseInt(hexColour.slice(channelStart, channelStart + 2), 16) / 255;
    return channel <= 0.03928 ? channel / 12.92 : ((channel + 0.055) / 1.055) ** 2.4;
  });
  return 0.2126 * linearChannels[0]! + 0.7152 * linearChannels[1]! + 0.0722 * linearChannels[2]!;
}

function contrastRatio(firstHexColour: string, secondHexColour: string): number {
  const luminances = [relativeLuminance(firstHexColour), relativeLuminance(secondHexColour)];
  return (Math.max(...luminances) + 0.05) / (Math.min(...luminances) + 0.05);
}

function pairsBelowMinimum(palette: Map<string, string>): string[] {
  const failingPairs: string[] = [];
  const checkedPairs: Array<[string, string]> = [["--accent-ink", "--accent"], ["--ink", "--bubble-user"]];
  for (const textToken of TEXT_TOKENS) {
    for (const groundToken of GROUND_TOKENS) {
      checkedPairs.push([textToken, groundToken]);
    }
  }
  for (const [textToken, groundToken] of checkedPairs) {
    const textHex = palette.get(textToken);
    const groundHex = palette.get(groundToken);
    assert.ok(textHex && groundHex, `${textToken} and ${groundToken} must both be hex tokens`);
    const ratio = contrastRatio(textHex, groundHex);
    if (ratio < MINIMUM_TEXT_CONTRAST) {
      failingPairs.push(`${textToken} on ${groundToken}: ${ratio.toFixed(2)}`);
    }
  }
  return failingPairs;
}

test("should_keep_every_text_colour_readable_on_every_ground_in_light_appearance", () => {
  assert.deepEqual(pairsBelowMinimum(lightPalette), []);
});

test("should_keep_every_text_colour_readable_on_every_ground_in_dark_appearance", () => {
  assert.deepEqual(pairsBelowMinimum(darkPalette), []);
});

test("should_report_the_icon_violet_as_too_weak_for_text_on_the_dark_ground", () => {
  // The reason dark appearance has its own lifted accent.
  const iconVioletOnDarkGround = contrastRatio(lightPalette.get("--accent")!, darkPalette.get("--ground")!);
  assert.ok(iconVioletOnDarkGround < MINIMUM_TEXT_CONTRAST);
});

test("should_measure_the_extremes_of_contrast_as_one_and_twenty_one", () => {
  assert.equal(contrastRatio("#777777", "#777777"), 1);
  assert.ok(Math.abs(contrastRatio("#000000", "#ffffff") - 21) < 0.001);
});
