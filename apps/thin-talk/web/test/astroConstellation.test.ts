import assert from "node:assert/strict";
import { test } from "node:test";
import fs from "node:fs";
import path from "node:path";
import { ASTRO_CONSTELLATION_LINK_COUNT, astroConstellationHtml } from "../src/astro/AstroConstellation";
import { AstroFigure } from "../src/astro/AstroFigure";
import { installDom, waitFor } from "./support/dom";
import { bootJourney, MET_ASTRO_PROFILE, waitUntilReady } from "./support/journey";

const astroStylesheet = (): string => fs.readFileSync(path.join(process.cwd(), "src", "astro.css"), "utf8");

test("should_draw_six_stars_joined_by_seven_lines_in_order", () => {
  const constellationHtml = astroConstellationHtml();

  assert.equal((constellationHtml.match(/<circle class="astro-constellation__star"/g) ?? []).length, 6);
  assert.equal((constellationHtml.match(/<line class="astro-constellation__link"/g) ?? []).length, ASTRO_CONSTELLATION_LINK_COUNT);
  for (let linkOrder = 0; linkOrder < ASTRO_CONSTELLATION_LINK_COUNT; linkOrder += 1) {
    assert.match(constellationHtml, new RegExp(`--link-order:${linkOrder};`), `line ${linkOrder} waits its turn`);
  }
  assert.match(constellationHtml, /aria-hidden="true"/, "decoration: the words beside it say what Astro is doing");
});

// The person's look (Rainbow, Aurora, Sunset) used to colour the trail under
// the character; it now colours the lines, so the choice still shows.
test("should_colour_its_lines_with_the_four_colours_of_the_chosen_look", () => {
  const constellationHtml = astroConstellationHtml();

  for (let lookColour = 1; lookColour <= 4; lookColour += 1) {
    assert.match(constellationHtml, new RegExp(`stroke:var\\(--astro-trail-${lookColour}\\)`));
  }
  assert.doesNotMatch(constellationHtml, /--astro-trail-5/, "only the look's four colours");
});

test("should_draw_the_constellation_and_no_character_picture_or_trail_in_the_figure", () => {
  installDom();
  const figure = new AstroFigure(document, "astro-home__astro");

  assert.ok(figure.element.querySelector("svg.astro-constellation"));
  assert.equal(figure.element.querySelector("img"), null);
  assert.equal(figure.element.querySelector(".astro-figure__trail"), null);
});

test("should_draw_its_lines_while_charging_and_hold_still_under_reduce_motion", () => {
  const stylesheet = astroStylesheet();
  const reducedMotionRules = stylesheet.slice(stylesheet.indexOf("/* Reduce Motion: the constellation"));

  assert.match(stylesheet, /\.astro-figure--charging \.astro-constellation__link,[^{]*\{\s*animation: astro-draw-link/);
  assert.match(reducedMotionRules, /\.astro-constellation__link,\s*\.astro-constellation__star,[^{]*\{\s*animation: none;/);
  assert.match(reducedMotionRules, /\.astro-constellation__link \{\s*stroke-dashoffset: 0;/, "every line drawn");
});

function typeDraft(draftText: string): void {
  const field = document.querySelector<HTMLTextAreaElement>("#composer-input");
  assert.ok(field !== null);
  field.value = draftText;
  field.dispatchEvent(new window.Event("input", { bubbles: true }));
}

function messageField(): HTMLElement {
  const found = document.getElementById("composer");
  assert.ok(found !== null);
  return found;
}

// The field is where people type; a figure beside it only crowded the words.
test("should_keep_astro_out_of_the_message_field_while_its_skill_buttons_stay", async () => {
  bootJourney(undefined, { astroProfile: MET_ASTRO_PROFILE });
  await waitUntilReady();
  await waitFor(() => document.querySelector<HTMLElement>("#skill-launchers")?.hidden === false, "the skill buttons");

  assert.equal(messageField().querySelector("svg.astro-constellation"), null, "an empty field");
  typeDraft("Why is the sky blue?");
  assert.equal(messageField().querySelector("svg.astro-constellation"), null, "a draft to send");
  assert.equal(document.getElementById("composer-astro"), null, "no place kept for it");
});

test("should_draw_lines_in_the_newest_answer_only_while_astro_thinks_or_works", () => {
  const stylesheet = astroStylesheet();
  const drawingRule = stylesheet.slice(0, stylesheet.indexOf("animation: astro-draw-link"));
  const drawingSelectors = drawingRule.slice(drawingRule.lastIndexOf("}") + 1);

  for (const drawingMood of ["thinking", "working"]) {
    assert.match(drawingSelectors, new RegExp(`\\.message--latest \\.astro-avatar\\[data-mood="${drawingMood}"\\] \\.astro-constellation__link`));
  }
  for (const stillMood of ["talking", "done", "stopped"]) {
    assert.doesNotMatch(drawingSelectors, new RegExp(`data-mood="${stillMood}"`), `${stillMood} keeps the figure drawn`);
  }
  assert.doesNotMatch(drawingSelectors, /(^|,)\s*\.astro-avatar/, "an older answer's Astro stays still");
});

test("should_draw_stars_in_astros_ink_so_they_show_on_the_dark_sky_too", () => {
  assert.match(astroStylesheet(), /\.astro-constellation__star \{\s*fill: var\(--astro-ink\);/);
});
