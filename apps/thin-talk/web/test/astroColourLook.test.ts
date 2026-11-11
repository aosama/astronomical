import assert from "node:assert/strict";
import { test } from "node:test";
import fs from "node:fs";
import path from "node:path";
import { AstroColourLook, applyAstroColourLook } from "../src/astro/AstroColourLook";
import { installDom } from "./support/dom";

const astroStylesheet = (): string => fs.readFileSync(path.join(process.cwd(), "src", "astro.css"), "utf8");

function colourLookClasses(colourLookOwner: HTMLElement): string[] {
  return [...colourLookOwner.classList].filter((className) => className.startsWith("astro-colour-look--"));
}

test("should_put_the_picked_colour_looks_class_on_the_owner", () => {
  installDom();
  const colourLookOwner = document.createElement("div");

  applyAstroColourLook(colourLookOwner, AstroColourLook.AURORA);

  assert.deepEqual(colourLookClasses(colourLookOwner), ["astro-colour-look--aurora"]);
});

test("should_set_no_class_for_silver_because_the_stylesheet_draws_it_unaided", () => {
  installDom();
  const colourLookOwner = document.createElement("div");

  applyAstroColourLook(colourLookOwner, AstroColourLook.SILVER);

  assert.deepEqual(colourLookClasses(colourLookOwner), []);
});

test("should_keep_one_look_at_a_time_and_leave_other_classes_alone", () => {
  installDom();
  const colourLookOwner = document.createElement("div");
  colourLookOwner.className = "app--astro";

  applyAstroColourLook(colourLookOwner, AstroColourLook.SUNSET);
  applyAstroColourLook(colourLookOwner, AstroColourLook.RAINBOW);
  assert.deepEqual(colourLookClasses(colourLookOwner), ["astro-colour-look--rainbow"]);
  applyAstroColourLook(colourLookOwner, AstroColourLook.SILVER);
  assert.deepEqual(colourLookClasses(colourLookOwner), [], "back to silver clears every look");
  assert.ok(colourLookOwner.classList.contains("app--astro"), "unrelated classes survive");
});

test("should_define_look_colours_only_inside_a_colour_look_so_silver_has_none", () => {
  const stylesheet = astroStylesheet();
  const rootRules = stylesheet.slice(stylesheet.indexOf(":root {"), stylesheet.indexOf(".astro-colour-look--rainbow {"));

  assert.doesNotMatch(rootRules, /--astro-colour-look-\d:/, "nothing defined for the default look");
  for (const colourLook of ["rainbow", "aurora", "sunset"]) {
    assert.match(stylesheet, new RegExp(`\\.astro-colour-look--${colourLook} \\{\\s*--astro-colour-look-1:`));
  }
  assert.doesNotMatch(stylesheet, /\.astro-colour-look--silver/, "silver needs no class");
});

// With no look colours defined, a silver download meter would be grey; it
// falls back to Astro's action colour, the app's violet, at every stop.
test("should_fill_the_download_meter_with_astros_action_colour_under_silver", () => {
  const stylesheet = astroStylesheet();
  const meterRule = stylesheet.slice(stylesheet.indexOf(".download-meter__fill {"));
  const meterBackground = meterRule.slice(0, meterRule.indexOf("}"));

  for (let lookColour = 1; lookColour <= 4; lookColour += 1) {
    assert.match(meterBackground, new RegExp(`var\\(--astro-colour-look-${lookColour}, var\\(--astro-action\\)\\)`));
  }
  assert.doesNotMatch(meterBackground, /var\(--astro-colour-look-\d\)/, "no stop without a fallback");
});
