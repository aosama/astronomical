import assert from "node:assert/strict";
import { test } from "node:test";
import fs from "node:fs";
import path from "node:path";
import { ASTRO_GALAXY_ARM_STAR_COUNT, astroGalaxyHtml } from "../src/astro/AstroGalaxy";
import { AstroFigure } from "../src/astro/AstroFigure";
import { installDom } from "./support/dom";

const astroStylesheet = (): string => fs.readFileSync(path.join(process.cwd(), "src", "astro.css"), "utf8");

const GRID_SIZE = 24;

interface DrawnStar {
  readonly centreX: number;
  readonly centreY: number;
  readonly radius: number;
  readonly brightness: number;
}

function numberAttribute(starMarkup: string, attributeName: string): number {
  const found = new RegExp(` ${attributeName}="([0-9.]+)"`).exec(starMarkup);
  assert.ok(found !== null, `${attributeName} on ${starMarkup}`);
  return Number(found[1]);
}

function drawnArmStars(): DrawnStar[] {
  const armStarMarkups = astroGalaxyHtml().match(/<circle class="astro-galaxy__star"[^>]*>/g) ?? [];
  return armStarMarkups.map((starMarkup) => ({
    centreX: numberAttribute(starMarkup, "cx"),
    centreY: numberAttribute(starMarkup, "cy"),
    radius: numberAttribute(starMarkup, "r"),
    brightness: numberAttribute(starMarkup, "fill-opacity"),
  }));
}

test("should_draw_one_core_and_two_arms_of_stars_as_decoration", () => {
  const galaxyHtml = astroGalaxyHtml();

  assert.equal((galaxyHtml.match(/<circle class="astro-galaxy__core"/g) ?? []).length, 1);
  assert.equal(drawnArmStars().length, ASTRO_GALAXY_ARM_STAR_COUNT);
  assert.equal(ASTRO_GALAXY_ARM_STAR_COUNT, 14, "two arms of seven");
  assert.match(galaxyHtml, /aria-hidden="true"/, "decoration: the words beside it say what Astro is doing");
});

// Lines between the stars made the earlier figure read as a spider's web.
test("should_join_no_two_stars_with_a_line", () => {
  assert.doesNotMatch(astroGalaxyHtml(), /<(line|path|polyline|polygon)\b/);
});

test("should_shrink_and_dim_the_stars_along_each_arm_and_keep_the_last_one_visible", () => {
  const armStars = drawnArmStars();
  const starsPerArm = ASTRO_GALAXY_ARM_STAR_COUNT / 2;

  for (const armStart of [0, starsPerArm]) {
    for (let starIndexInArm = 1; starIndexInArm < starsPerArm; starIndexInArm += 1) {
      const innerStar = armStars[armStart + starIndexInArm - 1]!;
      const outerStar = armStars[armStart + starIndexInArm]!;
      assert.ok(outerStar.radius <= innerStar.radius, `star ${starIndexInArm} is no larger than the one before`);
      assert.ok(outerStar.brightness < innerStar.brightness, `star ${starIndexInArm} is dimmer than the one before`);
    }
    const lastStar = armStars[armStart + starsPerArm - 1]!;
    assert.ok(lastStar.radius >= 0.5, "the last star does not vanish at the small size");
    assert.ok(lastStar.brightness > 0.3, "the last star still shows");
  }
});

// The arms turn around the core, so every star has to stay on the grid and
// each arm has to mirror the other through the centre.
test("should_keep_every_star_on_the_grid_and_the_arms_balanced_around_the_core", () => {
  const armStars = drawnArmStars();
  const starsPerArm = ASTRO_GALAXY_ARM_STAR_COUNT / 2;

  for (const armStar of armStars) {
    for (const nearEdgeOfStar of [armStar.centreX - armStar.radius, armStar.centreY - armStar.radius]) {
      assert.ok(nearEdgeOfStar >= 0, "inside the top and left edges");
    }
    for (const farEdgeOfStar of [armStar.centreX + armStar.radius, armStar.centreY + armStar.radius]) {
      assert.ok(farEdgeOfStar <= GRID_SIZE, "inside the bottom and right edges");
    }
  }
  for (let starIndexInArm = 0; starIndexInArm < starsPerArm; starIndexInArm += 1) {
    const firstArmStar = armStars[starIndexInArm]!;
    const secondArmStar = armStars[starsPerArm + starIndexInArm]!;
    assert.ok(Math.abs(firstArmStar.centreX + secondArmStar.centreX - GRID_SIZE) < 0.02);
    assert.ok(Math.abs(firstArmStar.centreY + secondArmStar.centreY - GRID_SIZE) < 0.02);
  }
});

// Silver is the default look: with no colour look picked, no look colour is
// defined, so every arm star falls back to Astro's ink.
test("should_tint_arm_stars_with_a_looks_four_colours_and_fall_back_to_astros_ink", () => {
  const galaxyHtml = astroGalaxyHtml();

  for (let lookColour = 1; lookColour <= 4; lookColour += 1) {
    assert.match(galaxyHtml, new RegExp(`fill:var\\(--astro-colour-look-${lookColour},var\\(--astro-ink\\)\\)`));
  }
  assert.doesNotMatch(galaxyHtml, /--astro-colour-look-5/, "only the look's four colours");
  assert.doesNotMatch(galaxyHtml, /astro-galaxy__core"[^>]*--astro-colour-look/, "the core stays ink in every look");
  assert.match(astroStylesheet(), /\.astro-galaxy__core,\s*\.astro-galaxy__star \{\s*fill: var\(--astro-ink\);/);
});

test("should_draw_the_galaxy_and_no_character_picture_in_the_figure", () => {
  installDom();
  const figure = new AstroFigure(document, "astro-home__astro");

  assert.ok(figure.element.querySelector("svg.astro-galaxy"));
  assert.equal(figure.element.querySelector("img"), null);
});

test("should_turn_its_arms_while_charging_and_hold_still_under_reduce_motion", () => {
  const stylesheet = astroStylesheet();
  const reducedMotionRules = stylesheet.slice(stylesheet.indexOf("/* Reduce Motion: the galaxy"));

  assert.match(stylesheet, /\.astro-figure--charging \.astro-galaxy__arms,[^{]*\{\s*animation: astro-galaxy-turn/);
  assert.match(reducedMotionRules, /\.astro-galaxy__arms,\s*\.astro-galaxy__star,[^{]*\{\s*animation: none;/);
});

test("should_turn_the_arms_in_the_newest_answer_only_while_astro_thinks_or_works", () => {
  const stylesheet = astroStylesheet();
  const turningRule = stylesheet.slice(0, stylesheet.indexOf("animation: astro-galaxy-turn"));
  const turningSelectors = turningRule.slice(turningRule.lastIndexOf("}") + 1);

  for (const turningMood of ["thinking", "working"]) {
    assert.match(turningSelectors, new RegExp(`\\.message--latest \\.astro-avatar\\[data-mood="${turningMood}"\\] \\.astro-galaxy__arms`));
  }
  for (const stillMood of ["talking", "done", "stopped"]) {
    assert.doesNotMatch(turningSelectors, new RegExp(`data-mood="${stillMood}"`), `${stillMood} keeps the galaxy still`);
  }
  assert.doesNotMatch(turningSelectors, /(^|,)\s*\.astro-avatar/, "an older answer's Astro stays still");
});

// A mark that never stops moving beside every answer pulls the eye off the words.
test("should_hold_the_galaxy_still_at_rest", () => {
  const stylesheet = astroStylesheet();

  assert.doesNotMatch(stylesheet, /(^|\n)\.astro-galaxy__(arms|star|core)[^{]*\{[^}]*animation:/, "no animation without a working state");
});
